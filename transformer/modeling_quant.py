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
from .utils_quant import QuantizeLinear, QuantizeEmbedding, SymQuantizer, ClipLinear, ClipEmbedding, TwnQuantizer, QuantizeAct

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

        if config.quantize and config.emb_q:
            self.word_embeddings = QuantizeEmbedding(config.vocab_size, config.hidden_size, padding_idx = 0,config=config)
        elif config.clipping:
            self.word_embeddings = ClipEmbedding(config.vocab_size, config.hidden_size)
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
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

        self.act_quant_flag = False
        self.weight_quant_flag = False
        self.output_bertviz = False
        
        # ================================================================================  #
        # Weight Quant Setting
        # ================================================================================ #
        
        is_q_layer = True
        if config.layer_num != -1:
            is_q_layer = config.layer_num > i
        
        if self.config.quantize and config.qkv_q and is_q_layer:
            
            if self.config.quantize_weight:
                self.weight_quant_flag = True

            if self.config.qk_FP:
                self.query = nn.Linear(config.hidden_size, self.all_head_size)
                self.key = nn.Linear(config.hidden_size, self.all_head_size)
                self.value = QuantizeLinear(config.hidden_size, self.all_head_size,config=config, name=f"layer_{self.i}_{self.__class__.__name__}_value", weight_flag=self.weight_quant_flag, input_bit=config.input_bits)
            elif self.config.qkv_FP:
                self.query = nn.Linear(config.hidden_size, self.all_head_size)
                self.key = nn.Linear(config.hidden_size, self.all_head_size)
                self.value = nn.Linear(config.hidden_size, self.all_head_size)
            else:
                self.query = QuantizeLinear(config.hidden_size, self.all_head_size,config=config, map=self.config.map, name=f"layer_{self.i}_{self.__class__.__name__}_query", weight_flag=self.weight_quant_flag, input_bit=config.input_bits)
                self.key = QuantizeLinear(config.hidden_size, self.all_head_size,config=config, map=self.config.map, name=f"layer_{self.i}_{self.__class__.__name__}_key", weight_flag=self.weight_quant_flag, input_bit=config.input_bits)
                self.value = QuantizeLinear(config.hidden_size, self.all_head_size,config=config, name=f"layer_{self.i}_{self.__class__.__name__}_value", weight_flag=self.weight_quant_flag, input_bit=config.input_bits)


            
            
        elif config.clipping:
            self.query = ClipLinear(config.hidden_size, self.all_head_size, config=config)
            self.key = ClipLinear(config.hidden_size, self.all_head_size, config=config)
            self.value = ClipLinear(config.hidden_size, self.all_head_size, config=config)

        else:
            self.query = nn.Linear(config.hidden_size, self.all_head_size)
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # ================================================================================  #
        # ACT Quant Setting
        # ================================================================================ #
        if self.config.quantize_act:
            self.act_quant_flag = True

            self.query.act_flag = True
            self.value.act_flag = True
            self.key.act_flag = True

            self.input_bits = config.input_bits
            self.act_quantizer = SymQuantizer
            if self.config.act_quantizer == "ternary":
                # Default Min-Max 8 bit Activation Quantization
                self.act_quantizer = SymQuantizer
                self.register_buffer('clip_query', torch.Tensor([-config.clip_val, config.clip_val]))
                self.register_buffer('clip_key', torch.Tensor([-config.clip_val, config.clip_val]))
                self.register_buffer('clip_value', torch.Tensor([-config.clip_val, config.clip_val]))
                self.register_buffer('clip_attn', torch.Tensor([-config.clip_val, config.clip_val]))
                
            else:
                # Nbit Ternary Activation PACT Quantization 
                #self.act_quantizer = SymQuantizer
                self.q_act_quantizer = QuantizeAct(self.input_bits, name=f"layer_{self.i}_{self.__class__.__name__}_qq", two_sided=True, config=self.config, act_flag=self.act_quant_flag)
                self.k_act_quantizer = QuantizeAct(self.input_bits, name=f"layer_{self.i}_{self.__class__.__name__}_kk", two_sided=True, config=self.config, act_flag=self.act_quant_flag)
                self.qk_act_quantizer = QuantizeAct(self.input_bits, name=f"layer_{self.i}_{self.__class__.__name__}_qk", two_sided=False, config=self.config, act_flag=self.act_quant_flag)
                self.v_act_quantizer = QuantizeAct(self.input_bits, name=f"layer_{self.i}_{self.__class__.__name__}_vv", two_sided=True, config=self.config, act_flag=self.act_quant_flag)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, teacher_probs=None):
        # Stop Grad 
        if self.config.stop_grad and self.config.qk_FP:
            hidden_states_ = hidden_states.clone().detach()
            mixed_query_layer = self.query(hidden_states_)
            mixed_key_layer = self.key(hidden_states_)
            mixed_value_layer = self.value(hidden_states)
        elif self.config.stop_grad and self.config.qkv_FP:
            hidden_states_ = hidden_states.clone().detach()
            mixed_query_layer = self.query(hidden_states_)
            mixed_key_layer = self.key(hidden_states_)
            mixed_value_layer = self.value(hidden_states_)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        # Batch Size : 16, Max_len_seq : 64
        # q, k, v : 16, 64, 768
        # transpose for scores : 16, 64, 768 -> 16, 64, 12, 64 -> 16, 12(head), 64, 64 

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # Value Relation 
        attention_value = value_layer
        
        if self.config.quantize_act:
            if self.config.act_quantizer == "ternary":
                query_layer = self.act_quantizer.apply(query_layer, self.clip_query, self.input_bits, True)
                key_layer = self.act_quantizer.apply(key_layer, self.clip_key, self.input_bits, True)
                # query_layer = self.q_act_quantizer(query_layer)
                # key_layer = self.k_act_quantizer(key_layer)
            else:
                # query_layer = self.act_quantizer.apply(query_layer, self.clip_query, self.input_bits, True)
                # key_layer = self.act_quantizer.apply(key_layer, self.clip_key, self.input_bits, True)
                query_layer = self.q_act_quantizer(query_layer)
                key_layer = self.k_act_quantizer(key_layer)
            
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        st_attention_probs = nn.Softmax(dim=-1)(attention_scores)
    
        if self.config.teacher_attnmap and teacher_probs is not None:
            # Teacher Map Insertion
            tc_attention_probs = teacher_probs[0][self.i]
            attention_prob = st_attention_probs # attention probs to return (for append)
            attention_probs = self.dropout(tc_attention_probs)

        else:
            attention_prob = st_attention_probs # attention probs to return (for append)
            
            # EXP : PARKS (Step2 Option)
            if self.config.parks:
                if self.training:
                    tc_attention_probs = teacher_probs[self.i]
                    attention_probs = self.dropout(tc_attention_probs)
                else:
                    attention_probs = self.dropout(st_attention_probs)
            else:
                attention_probs = self.dropout(st_attention_probs)
        
        # quantize both attention probs and value layer for dot product
        if self.config.quantize_act:
            if self.config.act_quantizer == "ternary":
                attention_probs = self.act_quantizer.apply(attention_probs, self.clip_attn, self.input_bits, True)
                value_layer = self.act_quantizer.apply(value_layer, self.clip_value, self.input_bits, True)
                # attention_probs = self.qk_act_quantizer(attention_probs)
                # value_layer = self.v_act_quantizer(value_layer)
            else:
                # attention_probs = self.act_quantizer.apply(attention_probs, self.clip_attn, self.input_bits, True)
                # value_layer = self.act_quantizer.apply(value_layer, self.clip_value, self.input_bits, True)
                attention_probs = self.qk_act_quantizer(attention_probs)
                value_layer = self.v_act_quantizer(value_layer)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer_ = context_layer

        if self.config.teacher_context and teacher_probs is not None:
            context_layer = teacher_probs[1][self.i][0] # TI/CI - Layer Number - Context
            # context_layer = teacher_probs[1][self.i][1] # TI/CI - Layer Number - Output
        
        
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
        return context_layer, attention_scores, attention_prob, context_layer_, value_layer

class BertAttention(nn.Module):
    def __init__(self, config, i):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config, i)
        self.output = BertSelfOutput(config, i)
        self.config = config
        self.i = i

    def forward(self, input_tensor, attention_mask, teacher_probs=None):

        if self.training and self.config.teacher_input and self.config.num_hidden_layers > 12 and self.i == self.config.layer_thres_num:
            input_tensor = teacher_probs[2][self.i].clone().detach()  # Layer Input Intervention
        
        self_output, layer_att, layer_probs, layer_context, value_layer = self.self(input_tensor, attention_mask, teacher_probs=teacher_probs)
        attention_output, self_output_hs = self.output(self_output, input_tensor)
        
        # MSKIM norm based analysis
        return attention_output, layer_att, layer_probs, (layer_context, attention_output, value_layer, self_output_hs)


class BertSelfOutput(nn.Module):
    def __init__(self, config, i):
        super(BertSelfOutput, self).__init__()

        is_q_layer = True
        self.act_quant_flag = False
        self.weight_quant_flag = False

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)

        if config.layer_num != -1:
            is_q_layer = config.layer_num > i

        if config.quantize and config.qkv_q and is_q_layer:
            
            if config.quantize_weight:
                self.weight_quant_flag = True
            
            if config.quantize_act:
                self.act_quant_flag = True

            self.dense = QuantizeLinear(config.hidden_size, config.hidden_size,config=config, name=f"layer_{i}_{self.__class__.__name__}_output", weight_flag=self.weight_quant_flag, act_flag=self.act_quant_flag, input_bit=config.input_bits)
            
        elif config.clipping:
            self.dense = ClipLinear(config.hidden_size, config.hidden_size, config=config)
            
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        # Norm Based 
        # layer_value = layer_value.permute(0, 2, 1, 3)
        # layer_value = layer_value.reshape(layer_value.shape[0], layer_value.shape[1], -1)
        # norm_based = self.dense(layer_value)

        # new_size = norm_based.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # norm_based = norm_based.view(*new_size)
        # norm_based = norm_based.permute(0, 2, 1, 3)

        hidden_states = self.dense(hidden_states)
        self_output_hs = hidden_states
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states , self_output_hs


class BertIntermediate(nn.Module):
    def __init__(self, config, i):
        super(BertIntermediate, self).__init__()
        self.i = i
        is_q_layer = True
        self.act_quant_flag = False
        self.weight_quant_flag = False

        if config.layer_num != -1:
            is_q_layer = config.layer_num > i
        
        if config.quantize and config.ffn_q_1 and is_q_layer:

            if config.quantize_weight:
                self.weight_quant_flag = True
            
            if config.quantize_act:
                self.act_quant_flag = True

            self.dense = QuantizeLinear(config.hidden_size, config.intermediate_size,config=config, name=f"layer_{self.i}_{self.__class__.__name__}", weight_flag=self.weight_quant_flag, act_flag=self.act_quant_flag, input_bit=config.input_bits)

        elif config.clipping:
            self.dense = ClipLinear(config.hidden_size, config.intermediate_size, config=config)

        else:
            self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
            

    def forward(self, hidden_states):
        #torch.save(hidden_states, f"Q_layer_{self.i}_ffn1_input.pt")
        hidden_states = self.dense(hidden_states)
        #torch.save(hidden_states, f"Q_layer_{self.i}_ffn1_output.pt")
        hidden_states = gelu(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config, i):
        super(BertOutput, self).__init__()
        self.i = i
        is_q_layer = True
        self.act_quant_flag = False
        self.weight_quant_flag = False

        if config.layer_num != -1:
            is_q_layer = config.layer_num > i

        if config.quantize and config.ffn_q_2 and is_q_layer:

            if config.quantize_weight:
                self.weight_quant_flag = True
            
            if config.quantize_act:
                self.act_quant_flag = True

            self.dense = QuantizeLinear(config.intermediate_size, config.hidden_size,config=config, name=f"layer_{self.i}_{self.__class__.__name__}", weight_flag=self.weight_quant_flag, act_flag=self.act_quant_flag, input_bit=config.input_bits)
            
        elif config.clipping:
            self.dense = ClipLinear(config.intermediate_size, config.hidden_size, config=config)
        else:
            self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
            

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        #torch.save(hidden_states, f"Q_layer_{self.i}_ffn2_input.pt")
        hidden_states = self.dense(hidden_states)
        #torch.save(hidden_states, f"Q_layer_{self.i}_ffn2_output.pt")
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        #torch.save(hidden_states, f"Q_layer_{self.i}_ffn2_Layernorm_output.pt")
        
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config, i):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config, i)
        self.intermediate = BertIntermediate(config, i)
        self.output = BertOutput(config, i)

    def forward(self, hidden_states, attention_mask, teacher_probs=None):

        attention_output, layer_att, layer_probs, layer_value = self.attention(
            hidden_states, attention_mask, teacher_probs=teacher_probs)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output, layer_att, layer_probs, layer_value


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.layer = nn.ModuleList([BertLayer(config, i)
                                    for i in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, teacher_probs=None):
        all_encoder_layers = [hidden_states]
        all_encoder_atts = []
        all_encoder_probs = []
        all_encoder_values = []

        for _, layer_module in enumerate(self.layer):
            hidden_states, layer_att, layer_probs, layer_value = layer_module(
                hidden_states, attention_mask, teacher_probs=teacher_probs)
            all_encoder_layers.append(hidden_states)
            all_encoder_atts.append(layer_att)
            all_encoder_probs.append(layer_probs)
            all_encoder_values.append(layer_value)

        return all_encoder_layers, all_encoder_atts, all_encoder_probs, all_encoder_values


class BertPooler(nn.Module):
    def __init__(self, config, recurs=None):
        super(BertPooler, self).__init__()

        self.act_quant_flag = False
        self.weight_quant_flag = False
        
        if config.quantize and config.cls_q:

            if config.quantize_weight:
                self.weight_quant_flag = True
            
            if config.quantize_act:
                self.act_quant_flag = True

            self.dense = QuantizeLinear(config.hidden_size, config.hidden_size,config=config, name=f"{self.__class__.__name__}", weight_flag=self.weight_quant_flag, act_flag=self.act_quant_flag, input_bit=config.input_bits)
        elif config.clipping:
            self.dense = ClipLinear(config.hidden_size, config.hidden_size, config=config)
            #self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)

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
            logger.info("Loading model {}".format(weights_path))
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

        logger.info('loading model...')
        
        load(model, prefix=start_prefix)
        logger.info('done!')
        if len(missing_keys) > 0:
            # logger.info("Weights of {} not initialized from pretrained model: {}".format(
            #     model.__class__.__name__, missing_keys))
            pass
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))

        return model


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, teacher_probs=None):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # extended_attention_mask = extended_attention_mask.to(
        #     dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers, attention_scores, attention_probs, attention_values = self.encoder(embedding_output,
                                                  extended_attention_mask, teacher_probs=teacher_probs)

        pooled_output = self.pooler(encoded_layers)
        return encoded_layers, attention_scores, attention_probs, attention_values, pooled_output

class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels = 2):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)
        self.config = config
        
        # self.coeff = nn.Parameter(torch.zeros(config.num_hidden_layers))
        # self.coeff = nn.Parameter(torch.zeros(config.num_hidden_layers))
        # self.output_coeff = nn.Parameter(torch.ones(1)*2)
        # self.coeff = nn.Parameter(torch.zeros(config.num_hidden_layers, 2))
        self.coeff = nn.Parameter(torch.zeros(config.num_hidden_layers, 2))
        # self.coeff = nn.Parameter(torch.zeros(2))
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, 
                token_type_ids=None,
                attention_mask=None, 
                labels=None,
                output_mode=None,
                teacher_outputs=None,
                seq_lengths=None):
        
        encoded_layers, attention_scores, attention_probs, attention_values, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, teacher_probs=teacher_outputs)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # ================================================================================  #
        # Learnable Loss Coefficient
        # ================================================================================ #

        teacher_probs, teacher_values, teacher_reps, teacher_logits = teacher_outputs

        loss = 0.
        cls_loss = 0.
        output_loss = 0.
        attmap_loss = 0.
        rep_loss = 0.

        output_loss_list = []
        map_loss_list = []
        rep_loss_list = []
        map_coeff_list = []
        output_coeff_list = []

        # Pred Loss
        if output_mode == "classification":
            cls_loss = soft_cross_entropy(logits,teacher_logits)
        elif output_mode == "regression":
            cls_loss = MSELoss()(logits, teacher_logits)
        else:
            cls_loss = soft_cross_entropy(logits,teacher_logits)
    
        # Output Loss
        for i, (student_value, teacher_value) in enumerate(zip(attention_values, teacher_values)):    
            tmp_loss = MSELoss()(student_value[1], teacher_value[1]) # 1 : Attention Output 0 : Layer Context
            output_loss_list.append(tmp_loss)
            output_loss += tmp_loss

        # Attention Map Loss
        BATCH_SIZE = attention_probs[0].shape[0]
        NUM_HEADS = attention_probs[0].shape[1]
        MAX_SEQ = attention_probs[0].shape[2]
        
        mask = torch.zeros(BATCH_SIZE, NUM_HEADS, MAX_SEQ, MAX_SEQ, dtype=torch.float32)
        mask_seq = []
        
        for sent in range(BATCH_SIZE):
            s = seq_lengths[sent]
            mask[sent, :, :s, :s] = 1.0
        
        mask = mask.to("cuda")
        for i, (student_prob, teacher_prob) in enumerate(zip(attention_probs, teacher_probs)):            
                    
            # student_prob = student_prob["attn"]
            # teacher_prob = teacher_prob["attn"]

            # KLD(teacher || student)
            # = sum (p(t) log p(t)) - sum(p(t) log p(s))
            # = (-entropy) - (-cross_entropy)
            
            student = torch.clamp_min(student_prob, 1e-8)
            teacher = torch.clamp_min(teacher_prob, 1e-8)

            # p(t) log p(s) = negative cross entropy
            neg_cross_entropy = teacher * torch.log(student) * mask
            neg_cross_entropy = torch.sum(neg_cross_entropy, dim=-1)  # (b, h, s, s) -> (b, h, s)
            neg_cross_entropy = torch.sum(neg_cross_entropy, dim=-1) / seq_lengths.view(-1, 1)  # (b, h, s) -> (b, h)

            # p(t) log p(t) = negative entropy
            neg_entropy = teacher * torch.log(teacher) * mask
            neg_entropy = torch.sum(neg_entropy, dim=-1)  # (b, h, s, s) -> (b, h, s)
            neg_entropy = torch.sum(neg_entropy, dim=-1) / seq_lengths.view(-1, 1)  # (b, h, s) -> (b, h)

            kld_loss = neg_entropy - neg_cross_entropy
            
            # kld_loss_sum = torch.sum(kld_loss)
            kld_loss_mean = torch.mean(kld_loss)

            # Other Option (Cosine Similarity, MSE Loss)
            # attnmap_mse_loss = loss_mse(student, teacher)
            #kld_loss_sum = torch.nn.functional.cosine_similarity(student, teacher, -1).mean()
            

            #layer_attmap_loss[i].update(kld_loss_sum)
            # attmap_loss += attnmap_mse_loss
            map_loss_list.append(kld_loss_mean)
            attmap_loss +=kld_loss_mean

        # Rep Loss
        for i, (student_rep, teacher_rep) in enumerate(zip(encoded_layers, teacher_reps)):
            
            tmp_loss = MSELoss()(student_rep, teacher_rep)
            rep_loss_list.append(tmp_loss)
            rep_loss += tmp_loss

        # coeff = self.softmax(self.coeff/0.3)
        # loss = cls_loss + rep_loss + attmap_loss * coeff[0] + output_loss * coeff[1]
        loss += rep_loss_list[0] # Embedding Loss
        loss += cls_loss
        coeff = self.softmax(self.coeff/self.config.sm_temp)
        for i in range(self.config.num_hidden_layers):
              # map_coeff = torch.sigmoid(self.coeff[i])
              # output_coeff = 1 - map_coeff

              
              # Logging Coeff
            #   map_coeff_list.append(map_coeff)
            #   output_coeff_list.append(output_coeff)

              # layer_loss = rep_loss_list[i+1]*0.5 + map_coeff*map_loss_list[i] + output_coeff*output_loss_list[i]
              layer_loss = rep_loss_list[i+1] + map_loss_list[i]*coeff[i][0] + output_loss_list[i]*coeff[i][1]
              loss += layer_loss

        
        
        # loss = cls_loss + rep_loss + attmap_loss

        return logits, loss, cls_loss, rep_loss, output_loss, attmap_loss, coeff
        # if labels is not None:
        #     loss_fct = nn.CrossEntropyLoss()
        #     loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #     return loss, attention_scores, encoded_layers
        # else:
        #     return logits, attention_scores, encoded_layers, attention_probs, attention_values

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
        teacher_probs=None
    ):
        sequence_output, att_output, attention_probs, attention_values, pooled_output = self.bert(
            input_ids,token_type_ids,attention_mask, teacher_probs=teacher_probs)

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
