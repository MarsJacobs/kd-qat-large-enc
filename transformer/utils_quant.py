import torch
import torch.nn as nn
import sys
import logging

from .pact_func import *
from .lsq import *

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()

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

class QuantizeLinear(nn.Linear):

    def __init__(self,  *kargs,bias=True, config = None, map=False):
        super(QuantizeLinear, self).__init__(*kargs,bias=True)
        self.quantize_act = config.quantize_act
        self.weight_bits = config.weight_bits
        self.quantize_act = config.quantize_act
        self.mean_scale = config.mean_scale
        
        self.map = map

        self.config = config
        
        self.register_buffer('qweight', self.weight.clone().detach())

        # Weight Quantization Setting
        self.clip_initialize()
        
        if self.quantize_act:
            self.input_bits = config.input_bits
            
            if self.input_bits == 8:
                # Default Min-Max 8 bit Activation Quantization
                self.act_quantizer = SymQuantizer
            else:
                # 2bit Ternary Activation Quantization 
                self.act_quantizer = TwnQuantizer
            
            self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))

    def clip_initialize(self):
        config = self.config
        
        if self.map:
            self.config.quantizer = "pact"
        else:
            self.config.quantizer = "ternary"

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
                    init_clip_valn = self.weight.abs().mean()* -2 *config.clip_ratio
                else:
                    raise ValueError("[MS] PACT : Choose Clip Value init method")
                # EXP : Gradual Quantization 4bit Setting (on Going)
                self.weight_quantizer = LearnedTwosidedClippedLinearQuantization(num_bits = 4,
                                                                         init_clip_val = init_clip_val,
                                                                         init_clip_valn = init_clip_valn,
                                                                         dequantize = True, 
                                                                         inplace = False) 
            if self.config.quantizer == 'lsq':
                self.weight_quantizer = quantization(weight = self.weight, config=config)
        else:
            self.weight_quantizer = SymQuantizer

        self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))

    def forward(self, input):
        # quantize weight
        if self.config.quantizer == "ternary" and self.map != True:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)
        else:
            weight = self.weight_quantizer(self.weight, layerwise=True)
    
        # quantize input
        if self.quantize_act:
            input = self.act_quantizer.apply(input, self.act_clip_val, self.input_bits, True)
        
        # nn.Linear w/ Quantized input and output
        out = nn.functional.linear(input, weight)
        
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out

    def __repr__(self):
        return '{0}(num_bits_weight={1}, w_quant_fn={2})'.format(self.__class__.__name__, self.weight_bits, self.weight_quantizer)

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
