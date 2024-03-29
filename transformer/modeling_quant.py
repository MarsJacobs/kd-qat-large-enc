# coding=utf-8
# 2020.04.20 - Add&replace quantization modules
#              Huawei Technologies Co., Ltd <zhangwei379@huawei.com>
# Copyright (c) 2020, Huawei Technologies Co., Ltd.  All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.w
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os

import torch
from torch import nn
from torch.autograd import Variable
from .configuration import BertConfig
from .utils_quant import QuantizeLinear, QuantizeEmbedding, SymQuantizer, TwnQuantizer

logger = logging.getLogger(__name__)

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
#WEIGHTS_NAME = "FFN_GT_KD_AUG.bin"
from torch.nn import CrossEntropyLoss, MSELoss

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return torch.sum((- targets_prob * student_likelihood), dim=-1).mean()

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        self.word_embeddings = QuantizeEmbedding(config.vocab_size, config.hidden_size, padding_idx = 0,config=config)
        # self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # position_embeddings and token_type_embeddings are kept in fp32 anyway
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config, i):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.i = i
        self.config = config

        self.act_quant_flag = True
        self.weight_quant_flag = True
        self.output_bertviz = False
        
        # ================================================================================  #
        # Weight Quant Setting
        # ================================================================================ #
        
        self.query = QuantizeLinear(config.hidden_size, self.all_head_size,config=config, name=f"layer_{self.i}_{self.__class__.__name__}_query", weight_flag=self.weight_quant_flag, input_bit=config.input_bits)
        self.key = QuantizeLinear(config.hidden_size, self.all_head_size,config=config, name=f"layer_{self.i}_{self.__class__.__name__}_key", weight_flag=self.weight_quant_flag, input_bit=config.input_bits)
        self.value = QuantizeLinear(config.hidden_size, self.all_head_size,config=config, name=f"layer_{self.i}_{self.__class__.__name__}_value", weight_flag=self.weight_quant_flag, input_bit=config.input_bits)

        # self.query = nn.Linear(config.hidden_size, self.all_head_size)
        # self.key = nn.Linear(config.hidden_size, self.all_head_size)
        # self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.input_bits = config.input_bits
        self.act_quantizer = SymQuantizer
        
        # Default Min-Max 8 bit Activation Quantization
        self.act_quantizer = SymQuantizer
        self.register_buffer('clip_query', torch.Tensor([-config.clip_val, config.clip_val]))
        self.register_buffer('clip_key', torch.Tensor([-config.clip_val, config.clip_val]))
        self.register_buffer('clip_value', torch.Tensor([-config.clip_val, config.clip_val]))
        self.register_buffer('clip_attn', torch.Tensor([-config.clip_val, config.clip_val]))

        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, teacher_probs=None):
        
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)        
        
        query_layer = self.act_quantizer.apply(query_layer, self.clip_query, self.input_bits, True)
        key_layer = self.act_quantizer.apply(key_layer, self.clip_key, self.input_bits, True)
            
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        st_attention_probs = nn.Softmax(dim=-1)(attention_scores)
    
        attention_prob = st_attention_probs # attention probs to return (for append)
        attention_probs = self.dropout(st_attention_probs)
            
        # quantize both attention probs and value layer for dot product
        attention_probs = self.act_quantizer.apply(attention_probs, self.clip_attn, self.input_bits, True)
        value_layer = self.act_quantizer.apply(value_layer, self.clip_value, self.input_bits, True)
                
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer_ = context_layer

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
       
        if self.output_bertviz:
            attn_data = {
                'attn': attention_prob,
                'queries': query_layer,
                'keys': key_layer
            }
            attention_prob = attn_data
        return context_layer, attention_scores, attention_prob, context_layer_ , value_layer

class BertAttention(nn.Module):
    def __init__(self, config, i):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config, i)
        self.output = BertSelfOutput(config, i)
        self.config = config
        self.norm = BertNormOutput(config)
        self.i = i
        self.output_norm = False

    def forward(self, input_tensor, attention_mask):

        self_output, layer_att, layer_probs, layer_context, layer_value  = self.self(input_tensor, attention_mask)
        attention_output, self_output_hs = self.output(self_output, input_tensor)
        
        if self.output_norm:
            norms_outputs = self.norm(
                input_tensor,
                layer_probs,
                layer_value,
                self.output.dense
            )
            return attention_output, layer_att, layer_probs, (layer_context, attention_output, norms_outputs)

        return attention_output, layer_att, layer_probs, (layer_context, attention_output, self_output_hs)


class BertSelfOutput(nn.Module):
    def __init__(self, config, i):
        super(BertSelfOutput, self).__init__()

        # self.act_quant_flag = False
        # self.weight_quant_flag = False
        self.config = config
        self.i = i
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)

        self.weight_quant_flag = True
        self.act_quant_flag = True
        
        self.dense = QuantizeLinear(config.hidden_size, config.hidden_size,config=config, name=f"layer_{i}_{self.__class__.__name__}_output", weight_flag=self.weight_quant_flag, act_flag=self.act_quant_flag, input_bit=config.input_bits)            
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor, teacher_probs=None):
        
        hidden_states = self.dense(hidden_states)
        self_output_hs = hidden_states
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states ,self_output_hs


class BertIntermediate(nn.Module):
    def __init__(self, config, i):
        super(BertIntermediate, self).__init__()
        self.i = i
        self.act_quant_flag = True
        self.weight_quant_flag = True

        self.weight_quant_flag = True
        self.act_quant_flag = True
        self.dense = QuantizeLinear(config.hidden_size, config.intermediate_size,config=config, name=f"layer_{self.i}_{self.__class__.__name__}", weight_flag=self.weight_quant_flag, act_flag=self.act_quant_flag, input_bit=config.input_bits)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = gelu(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config, i):
        super(BertOutput, self).__init__()
        self.i = i

        self.act_quant_flag = True
        self.weight_quant_flag = True
    
        self.dense = QuantizeLinear(config.intermediate_size, config.hidden_size,config=config, name=f"layer_{self.i}_{self.__class__.__name__}", weight_flag=self.weight_quant_flag, act_flag=self.act_quant_flag, input_bit=config.input_bits)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config, i):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config, i)
        self.intermediate = BertIntermediate(config, i)
        self.output = BertOutput(config, i)

    def forward(self, hidden_states, attention_mask, teacher_probs=None):

        attention_output, layer_att, layer_probs, layer_value = self.attention(
            hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output, layer_att, layer_probs, layer_value


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config, i)
                                    for i in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask):
        all_encoder_layers = [hidden_states]
        all_encoder_atts = []
        all_encoder_probs = []
        all_encoder_attns = []

        for _, layer_module in enumerate(self.layer):
            hidden_states, layer_att, layer_probs, layer_attn = layer_module(
                hidden_states, attention_mask)
            all_encoder_layers.append(hidden_states)
            all_encoder_atts.append(layer_att)
            all_encoder_probs.append(layer_probs)
            all_encoder_attns.append(layer_attn)

        return all_encoder_layers, all_encoder_atts, all_encoder_probs, all_encoder_attns


class BertPooler(nn.Module):
    def __init__(self, config, recurs=None):
        super(BertPooler, self).__init__()

        self.act_quant_flag = False
        self.weight_quant_flag = False
        self.weight_quant_flag = True
        self.dense = QuantizeLinear(config.hidden_size, config.hidden_size,config=config, name=f"{self.__class__.__name__}", weight_flag=self.weight_quant_flag, act_flag=self.act_quant_flag, input_bit=config.input_bits)
        # self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.activation = nn.Tanh()
        self.config = config

    def forward(self, hidden_states):
        pooled_output = hidden_states[-1][:, 0]
        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)

        return pooled_output


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Params:
            pretrained_model_name_or_path:
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            config: BertConfig instance
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        config = kwargs.get('config', None)
        kwargs.pop('config', None)
        
        if config is None:
            # Load config
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
            config = BertConfig.from_json_file(config_file)

        #logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(
                pretrained_model_name_or_path, WEIGHTS_NAME)
            # logger.info("Loading model {}".format(weights_path))
            state_dict = torch.load(weights_path, map_location='cpu')

        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        
        load(model, prefix=start_prefix)
        
        return model


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # extended_attention_mask = extended_attention_mask.to(
        #     dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers, attention_scores, attention_probs, attention_blocks = self.encoder(embedding_output,
                                                  extended_attention_mask)

        pooled_output = self.pooler(encoded_layers)
        return encoded_layers, attention_scores, attention_probs, attention_blocks, pooled_output

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels = 2):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)
        self.config = config
        
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, 
                token_type_ids=None,
                attention_mask=None, 
                seq_lengths=None):
        
        encoded_layers, student_atts, attention_probs, attention_blocks, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, student_atts, encoded_layers, attention_probs, attention_blocks

class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)
        
    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
        teacher_outputs=None
    ):
        sequence_output, att_output, attention_probs, attention_values, pooled_output = self.bert(
            input_ids,token_type_ids,attention_mask, teacher_probs=teacher_outputs)

        last_sequence_output = sequence_output[-1]

        logits = self.qa_outputs(last_sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        logits = (start_logits, end_logits)
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss, att_output, sequence_output

        return logits, att_output, sequence_output, attention_probs, attention_values


class BertNormOutput(nn.Module): # This class is added by Goro Kobayashi
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

    def forward(self, hidden_states, attention_probs, value_layer, dense):
        # hidden_states: (batch, seq_length, all_head_size)
        # attention_probs: (batch, num_heads, seq_length, seq_length)
        # value_layer: (batch, num_heads, seq_length, head_size)
        # dense: nn.Linear(all_head_size, all_head_size)

        with torch.no_grad():
            # value_layer is converted to (batch, seq_length, num_heads, 1, head_size)
            value_layer = value_layer.permute(0, 2, 1, 3).contiguous()
            value_shape = value_layer.size()
            value_layer = value_layer.view(value_shape[:-1] + (1, value_shape[-1],))

            # dense weight is converted to (num_heads, head_size, all_head_size)
            dense = dense.weight
            dense = dense.view(self.all_head_size, self.num_attention_heads, self.attention_head_size)
            dense = dense.permute(1, 2, 0).contiguous()

            # Make transformed vectors f(x) from Value vectors (value_layer) and weight matrix (dense).
            transformed_layer = value_layer.matmul(dense)
            transformed_shape = transformed_layer.size() #(batch, seq_length, num_heads, 1, all_head_size)
            transformed_layer = transformed_layer.view(transformed_shape[:-2] + (transformed_shape[-1],))
            transformed_layer = transformed_layer.permute(0, 2, 1, 3).contiguous() 
            transformed_shape = transformed_layer.size() #(batch, num_heads, seq_length, all_head_size)
            transformed_norm = torch.norm(transformed_layer, dim=-1)

            # Make weighted vectors αf(x) from transformed vectors (transformed_layer) and attention weights (attention_probs).
            weighted_layer = torch.einsum('bhks,bhsd->bhksd', attention_probs, transformed_layer) #(batch, num_heads, seq_length, seq_length, all_head_size)
            weighted_norm = torch.norm(weighted_layer, dim=-1)

            # Sum each αf(x) over all heads: (batch, seq_length, seq_length, all_head_size)
            summed_weighted_layer = weighted_layer.sum(dim=1)

            # Calculate L2 norm of summed weighted vectors: (batch, seq_length, seq_length)
            summed_weighted_norm = torch.norm(summed_weighted_layer, dim=-1)

            del transformed_shape
            
            # outputs: ||f(x)||, ||αf(x)||, ||Σαf(x)||
            outputs = (transformed_norm,
                    weighted_norm,
                    summed_weighted_norm,
                    transformed_layer,
                    )
            del weighted_layer, summed_weighted_layer
        torch.cuda.empty_cache()
        return outputs