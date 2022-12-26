import torch
import torch.nn as nn
import sys
import logging

from transformers import SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST
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

class TwnQuantizer_mx(torch.autograd.Function):
    """Ternary Weight Networks (TWN)
    Ref: https://arxiv.org/abs/1605.04711
    """
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise):
        """
        :param input: tensor to be ternarized
        :return: quantized tensor
        """
        mean_scale = 1

        ctx.save_for_backward(input, clip_val)
    
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        
        if layerwise:
            # import pdb; pdb.set_trace()
            m = input.norm(p=1).div(input.nelement())
            thres_mask = mean_scale * m  
            mask = (input.abs() > thres_mask).float()
            thres = (mask * input).abs().sum() / mask.sum()
            
            step_size = (thres * 2) / (2 ** num_bits - 1)
            input = torch.where(input < thres, input, thres)
            input = torch.where(input > -1*thres, input, thres*-1)
            result = torch.round(input / step_size) * step_size
            
            
        else: # row-wise only for embed / weight
            n = input[0].nelement()
            m = input.data.norm(p=1, dim=1).div(n)
            thres_mask = (mean_scale * m).view(-1, 1).expand_as(input)
            mask = (input.abs() > thres_mask).float()
            thres = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)

            step_size = (thres * 2) / (2 ** num_bits - 1)
            input = torch.where(input < thres, input, thres)
            input = torch.where(input > -1*thres, input, thres*-1)
            result = torch.round(input / step_size) * step_size
            # result = alpha * pos - alpha * neg
        
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

    def __init__(self,  *kargs,bias=True, config = None, map=False, name=None, weight_flag=False, act_flag=False, input_bit=None):
        super(QuantizeLinear, self).__init__(*kargs,bias=True)
        self.weight_bits = config.weight_bits
        self.input_bits = input_bit

        self.name = name
        self.config = config

        self.weight_flag = True
        self.act_flag = True

        # Weight & Activation Quantization Setting
        self.clip_initialize()    
        

    def clip_initialize(self):
        config = self.config

        # ================================================================================  #
        # Quantizer Setting
        # ================================================================================ #
        if self.weight_bits < 8:
            # [MSKIM] Mixed Preicision Test 
            if self.weight_bits >= 4:
                self.weight_quantizer = TwnQuantizer_mx
            else:
                self.weight_quantizer = TwnQuantizer
        else:
            self.weight_quantizer = SymQuantizer

        self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))\
            
        self.act_quantizer = SymQuantizer        
        self.register_buffer('act_clip_val', torch.tensor([-config.clip_val, config.clip_val]))


    def forward(self, input):
        # quantize weight
        if self.weight_flag:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, True)
        else:
            weight = self.weight
        
        # quantize input
        if self.act_flag:
            q_input = self.act_quantizer.apply(input, self.act_clip_val, self.input_bits, True)
        else:
            q_input = input
        
        out = nn.functional.linear(q_input, weight)
        
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out

class QuantizeEmbedding(nn.Embedding):

    def __init__(self,  *kargs,padding_idx=None, config = None):
        super(QuantizeEmbedding, self).__init__(*kargs, padding_idx = padding_idx)
        self.weight_bits = config.weight_bits
        self.layerwise = False
        self.config = config

        self.weight_flag = True
        self.clip_initialize()

        self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))
    
    def clip_initialize(self):
        config = self.config
        if self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
        if self.weight_bits >= 4:
            self.weight_quantizer = TwnQuantizer_mx

        self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))

    def forward(self, input):

        if self.weight_flag:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, self.layerwise)
        else:
            weight = self.weight
        
        out = nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out