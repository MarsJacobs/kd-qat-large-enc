import torch
import torch.nn as nn
import sys
import logging

from transformers import SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST

from .pact_func import *
from .lsq import *

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()


class LSQ_Quantizer(torch.nn.Module):
    def __init__(self, bit_width):
        super(LSQ_Quantizer, self).__init__()
        
        self.bit_width = bit_width
        
        self.Qn = -2**(bit_width - 1)
        self.Qp = 2 ** (bit_width - 1) - 1
        
        self.clip_val = torch.nn.Parameter(torch.ones(1)*1)

    def grad_scale(self, x, scale):
        y_out = x
        y_grad = x * scale

        y = (y_out - y_grad).detach() + y_grad

        return y

    def round_pass(self, x):
        y_out = x.round()
        y_grad = x
        y = torch.detach(y_out - y_grad) + y_grad

        return y


    def forward(self, input: torch.Tensor):
        
        # scale_factor = torch.Tensor([1 / (x.numel() * self.Qp) ** 0.5]).to(x)
        # scale = torch.Tensor([self.clip_val * 2 / 2 ** (self.bit_width)]).to(input)
        # zero_point = torch.Tensor([0]).to(x)
        # bit_width = torch.Tensor([self.bit_width]).to(x)
        # scale = self.grad_scale(self.clip_val, scale_factor)
        x = input / (self.clip_val * 2 / 2 ** (self.bit_width - 1))
        x = x.clamp(self.Qn, self.Qp)

        x_bar = self.round_pass(x)
        x_hat = x_bar * (self.clip_val * 2 / 2 ** (self.bit_width - 1))
        return x_hat

    def __repr__(self):
        return '{0}(num_bits_act={1}, clip_val={2})'.\
            format(self.__class__.__name__, self.bit_width, self.clip_val.item())

class SymQuantizer(torch.autograd.Function):
    """
        uniform quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val=2.5, num_bits=2, layerwise=False):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        
        ctx.save_for_backward(input, clip_val)
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        # NOTE: dynamic scaling (max_input).
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        s = (2 ** (num_bits - 1) - 1) / max_input
        output = torch.round(input * s).div(s)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None


class AsymQuantizer(torch.autograd.Function):
    """
        min-max quantization
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)

        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        # input = torch.clamp(input, clip_val[0], clip_val[1])
        # NOTE: dynamic scaling gives better performance than static
        if layerwise:
            alpha = (input.max() - input.min()).detach()
            beta = input.min().detach()
        else:
            if input.ndimension() <= 3:
                # weight & hidden layer
                alpha = (input.max(dim=-1, keepdim=True)[0] - input.min(dim=-1, keepdim=True)[0]).expand_as(input).detach()
                beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                # TODO: attention score matrix, calculate alpha / beta per head
                tmp = input.view(input.shape[0], input.shape[1], -1)
                alpha = (tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1) - \
                            tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)).expand_as(input).detach()
                beta = tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        input_normalized = (input - beta) / (alpha + 1e-8)
        s = (2**num_bits - 1)
        quant_input = torch.round(input_normalized * s).div(s)
        output = quant_input * (alpha + 1e-8) + beta


        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None


class TwnQuantizer(torch.autograd.Function):
    """Ternary Weight Networks (TWN)
    Ref: https://arxiv.org/abs/1605.04711
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param input: tensor to be ternarized
        :return: quantized tensor
        """
        mean_scale = 0.7

        ctx.save_for_backward(input, clip_val)
    
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])

        
        if layerwise:
            m = input.norm(p=1).div(input.nelement())
            thres = mean_scale * m  
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = (mask * input).abs().sum() / mask.sum()
            result = alpha * pos - alpha * neg
        else: # row-wise only for embed / weight
            n = input[0].nelement()
            m = input.data.norm(p=1, dim=1).div(n)
            thres = (mean_scale * m).view(-1, 1).expand_as(input)
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
            result = alpha * pos - alpha * neg
        
        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None

class QuantizeAct(torch.nn.Module):
    def __init__(self, num_bit, name=None, two_sided=False, config=None, act_flag=False):
        super(QuantizeAct, self).__init__()
        self.act_quantizer = None
        self.num_bit = num_bit
        self.name = name
        self.two_sided = two_sided
        self.config = config
        self.act_flag = act_flag
        self.weight_flag = None

        if self.config.act_quantizer == "pact":
            if two_sided:
                self.act_quantizer = LearnedTwosidedClippedLinearQuantization(num_bits = self.num_bit,
                                                                                    init_clip_val = 8.0,
                                                                                    init_clip_valn = -8.0,
                                                                                    dequantize = True, 
                                                                                    inplace = False) 
            else:
                self.act_quantizer = LearnedClippedLinearQuantization(num_bits = self.num_bit,
                                                                                    init_act_clip_val = 8.0,
                                                                                    dequantize = True, 
                                                                                    inplace = False) 
        if self.config.act_quantizer == "lsq":
            self.act_quantizer = LSQ_Quantizer(bit_width = self.num_bit)
                                                                
    def clip_initialize(self, s_init):        
        if self.config.act_quantizer == "pact":
            init_clip_val = torch.Tensor([s_init[1]]).to(s_init[1])
            self.act_quantizer.clip_val = nn.Parameter(init_clip_val)

            if self.two_sided:
                init_clip_valn = torch.Tensor([s_init[0]]).to(s_init[0])
                self.act_quantizer.clip_valn = nn.Parameter(init_clip_valn)

        if self.config.act_quantizer == "lsq":
            init_clip_valn = torch.Tensor([s_init[0]]).to(s_init[0])
            init_clip_val = torch.Tensor([s_init[1]]).to(s_init[0])

            if torch.abs(init_clip_valn) >= torch.abs(init_clip_val):
                clip_val = torch.abs(init_clip_valn)
            else:
                clip_val = init_clip_val
            
            self.act_quantizer.clip_val = nn.Parameter(clip_val)
        
    
    def forward(self, input):
        if self.act_flag:
            input = self.act_quantizer(input)
        else:
            input = input
        return input

class QuantizeLinear(nn.Linear):

    def __init__(self,  *kargs,bias=True, config = None, map=False, name=None, weight_flag=False, act_flag=False, input_bit=None):
        super(QuantizeLinear, self).__init__(*kargs,bias=True)
        self.weight_bits = config.weight_bits
        self.mean_scale = config.mean_scale
        self.input_bits = input_bit

        self.name = name
        self.map = map
        self.config = config
        self.register_buffer('qweight', self.weight.clone().detach())

        self.act_quantizer = None
        self.weight_quantizer= None

        self.weight_flag = weight_flag
        self.act_flag = act_flag

        # Weight & Activation Quantization Setting
        self.clip_initialize()    
        

    def clip_initialize(self, s_init=None):
        config = self.config

        # For Gradual Quantization (Deprecated)
        if self.map:
            self.config.quantizer = "pact"
        else:
            self.config.quantizer = "ternary"

        # ================================================================================  #
        # Weight Quantizer Setting
        # ================================================================================ #
        if s_init == None:
            if self.weight_bits < 8:
                if self.config.quantizer == "ternary":
                    self.weight_quantizer = TwnQuantizer
                if self.config.quantizer == "pact":
                    if config.clip_method == "minmax":
                        init_clip_val = self.weight.max() * config.clip_ratio
                        init_clip_valn = self.weight.min() * config.clip_ratio
                    elif config.clip_method == "std":
                        init_clip_val = self.weight.std() * config.clip_ratio
                        init_clip_valn = self.weight.std() * -1*config.clip_ratio
                    elif config.clip_method == "lsq":
                        init_clip_val = self.weight.abs().mean() * 2 * config.clip_ratio
                        init_clip_valn = self.weight.abs().mean()* -2 *config.clip_ratio
                    else:
                        raise ValueError("[MS] PACT : Choose Clip Value init method")
                    # EXP : Gradual Quantization 4bit Setting (on Going)
                    self.weight_quantizer = LearnedTwosidedClippedLinearQuantization(num_bits = self.weight_bits,
                                                                            init_clip_val = init_clip_val,
                                                                            init_clip_valn = init_clip_valn,
                                                                            dequantize = True, 
                                                                            inplace = False) 
                if self.config.quantizer == 'lsq':
                    self.weight_quantizer = quantization(weight = self.weight, config=config, tag='wt', bit=self.weight_bits)
            else:
                self.weight_quantizer = SymQuantizer

            self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))\
            
            # ================================================================================  #
            # Activation Quantizer
            # ================================================================================ #
            if self.config.quantize_act:
                if self.config.act_quantizer == "pact":
                    
                    init_clip_val = 5.0
                    init_clip_valn = -5.0
                        
                    self.act_quantizer = LearnedTwosidedClippedLinearQuantization(num_bits = self.input_bits,
                                                                                init_clip_val = init_clip_val,
                                                                                init_clip_valn = init_clip_valn,
                                                                                dequantize = True, 
                                                                                inplace = False) 
                elif self.config.act_quantizer == "lsq":
                    self.act_quantizer = LSQ_Quantizer(bit_width=self.input_bits)
                else:
                    self.act_quantizer = SymQuantizer
                
                self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))

        # ================================================================================  #
        # Activation Quantizer init Clip Val setting (Revisited)
        # ================================================================================ # 
        else:
            if self.config.act_quantizer == "pact":
                init_clip_valn = torch.Tensor([s_init[0]]).to(s_init[0])
                init_clip_val = torch.Tensor([s_init[1]]).to(s_init[0])
                
                self.act_quantizer.clip_val = nn.Parameter(init_clip_val)
                self.act_quantizer.clip_valn = nn.Parameter(init_clip_valn)

            elif self.config.act_quantizer == "lsq":
                init_clip_valn = torch.Tensor([s_init[0]]).to(s_init[0])
                init_clip_val = torch.Tensor([s_init[1]]).to(s_init[0])

                if torch.abs(init_clip_valn) >= torch.abs(init_clip_val):
                    clip_val = torch.abs(init_clip_valn)
                else:
                    clip_val = init_clip_val
                
                self.act_quantizer.clip_val = nn.Parameter(clip_val)

            else:
                self.act_quantizer = SymQuantizer

    def forward(self, input):
        # quantize weight
        if self.weight_flag:
            if self.config.quantizer == "ternary" and self.map != True:
                weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)
            else:
                weight = self.weight_quantizer(self.weight, layerwise=True)
        else:
            weight = self.weight
        
        # quantize input
        if self.act_flag:
            if self.config.act_quantizer != "ternary":
                q_input = self.act_quantizer(input)
            else:
                q_input = self.act_quantizer.apply(input, self.act_clip_val, self.input_bits, True)
        else:
            q_input = input
        
        # nn.Linear w/ Quantized input and output
        out = nn.functional.linear(q_input, weight)
        
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out

    def __repr__(self):
        return '{0}(num_bits_weight={1}, num_bits_act={2}, w_quant_fn={3}, a_quant_fn={4})'.format(self.__class__.__name__, self.weight_bits, self.input_bits, self.weight_quantizer, self.act_quantizer)

    # MSKIM get Ternary Weight Quantization's Threshold and Alpha Value
    def get_thres_alpha(self):

        weight = self.weight.clone().detach()
        new_weight = torch.where(weight < self.weight_clip_val[1], weight, self.weight_clip_val[1])
        new_weight = torch.where(weight > self.weight_clip_val[0], weight, self.weight_clip_val[0])
        m = weight.norm(p=1).div(weight.nelement())
        thres = 0.7 * m
        mask = (weight.abs() > thres).float()
        alpha = (mask * weight).abs().sum() / mask.sum()

        return thres, alpha




class QuantizeEmbedding(nn.Embedding):

    def __init__(self,  *kargs,padding_idx=None, config = None):
        super(QuantizeEmbedding, self).__init__(*kargs, padding_idx = padding_idx)
        self.weight_bits = config.weight_bits
        self.layerwise = False
        self.mean_scale = config.mean_scale
        self.config = config
        self.register_buffer('qweight', self.weight.clone().detach())

        self.clip_initialize()

        self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
    
    def clip_initialize(self):
        config = self.config
        if self.weight_bits == 2 or self.weight_bits == 4:
            if self.config.quantizer == "ternary":
                self.weight_quantizer = TwnQuantizer
            if self.config.quantizer == "pact":
                
                if config.clip_method == "minmax":
                    init_clip_val = self.weight.max() * config.clip_ratio
                    init_clip_valn = self.weight.min() * config.clip_ratio
                elif config.clip_method == "std":
                    init_clip_val = self.weight.std() * config.clip_ratio
                    init_clip_valn = self.weight.std() * -1*config.clip_ratio
                elif config.clip_method == "lsq":
                    init_clip_val = self.weight.abs().mean() * 2 * config.clip_ratio
                    init_clip_valn = self.weight.abs().mean() * -2 *config.clip_ratio
                else:
                    raise ValueError("[MS] PACT : Choose Clip Value init method")
                self.weight_quantizer = LearnedTwosidedClippedLinearQuantization(num_bits = self.weight_bits,
                                                                         init_clip_val = init_clip_val,
                                                                         init_clip_valn = init_clip_valn,
                                                                         dequantize = True, 
                                                                         inplace = False)   
                                                                            
            if self.config.quantizer == 'lsq':
                self.weight_quantizer = quantization(weight = self.weight, config=self.config)

        self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))

    def forward(self, input):
        if self.config.quantizer == "ternary":
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, self.layerwise)
        else:
            weight = self.weight_quantizer(self.weight, layerwise=self.layerwise)
        
        out = nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out

    def __repr__(self):
        return '{0}(num_bits_weight={1}, w_quant_fn={2})'.format(self.__class__.__name__, self.weight_bits, self.weight_quantizer)

    def get_thres_alpha(self):

        weight = self.weight.clone().detach()
        new_weight = torch.where(weight < self.weight_clip_val[1], weight, self.weight_clip_val[1])
        new_weight = torch.where(weight > self.weight_clip_val[0], weight, self.weight_clip_val[0])

        n = weight[0].nelement()
        m = weight.norm(p=1, dim=1).div(n)
        thres = (0.7 * m).view(-1,1).expand_as(weight)
        mask = (weight.abs() > thres).float()
        alpha = ((mask * weight).abs().sum(dim=1) / mask.sum(dim=1)).view(-1,1)

        return thres, alpha

class ClipLinear(nn.Linear):

    def __init__(self,  *kargs,bias=True, config = None):
        super(ClipLinear, self).__init__(*kargs,bias=True)

    def forward(self, input):
        
        # Clippinng weight
        weight = self.weight
        
        m = weight.norm(p=1).div(weight.nelement())
        
        thres = m
        mask = (weight.abs() > thres).float()
        alpha = (mask * weight).abs().sum() / mask.sum()
        

        weight = torch.where(weight < alpha, weight, alpha)
        weight = torch.where(weight > -1*alpha, weight, -1*alpha)
        
        out = nn.functional.linear(input, weight)
        
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out

class ClipEmbedding(nn.Embedding):

    def __init__(self,  *kargs,padding_idx=None, config = None):
        super(ClipEmbedding, self).__init__(*kargs,padding_idx = padding_idx)

    def forward(self, input):
        
        # Clippinng weight
        weight = self.weight
        
        m = weight.norm(p=1).div(weight.nelement())
        
        thres = m * 0.7
        mask = (weight.abs() > thres).float()
        alpha = (mask * weight).abs().sum() / mask.sum() 

        weight = torch.where(weight < alpha, weight, alpha)
        weight = torch.where(weight > -1*alpha, weight, -1*alpha)
        
        out = nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)

        return out
