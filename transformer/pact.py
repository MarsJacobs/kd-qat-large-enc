from collections import namedtuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction, Function
from collections import OrderedDict

import os.path
import sys
sys.path.append(" . ")
from pact_func import *
import pdb
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Q_Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Q_Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                         stride, padding, dilation, groups, bias)                                
        self.num_bit = 32
        self.weight_old = None
        self.quantize_weight = None
        self.q_idx = None
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        self.conv_input = 0
 
        ######### Static Padding Added ########
        image_size=224
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def initialize(self, n_bit, clip_init_val, clip_init_valn):
            #print('initializing num_bit_weight to %d' %(nbit))  
        self.num_bit = n_bit
        self.quantize_weight = LearnedTwosidedClippedLinearQuantization( num_bits = self.num_bit,
                                                                         init_clip_val = clip_init_val, 
                                                                         init_clip_valn = clip_init_valn,
                                                                         dequantize = True, 
                                                                         inplace = False) 
                
    def _weight_quant(self):
        weight = self.quantize_weight(self.weight)
        self.q_idx = self.quantize_weight.q_idx
        return weight

    def forward(self, input):
        #input = self.static_padding(input)
        self.conv_input = input
        if self.num_bit < 32:
            self.quantize_weight.num_bits = self.num_bit
            qweight = self._weight_quant()

        else:
            qweight = self.weight

        output = F.conv2d(input, qweight, self.bias,self.stride, self.padding, self.dilation, self.groups)
        #print("Conv2d", type(output))
        return output
       
    def __repr__(self):
        return '{0}(num_bits_weight={1}, w_quant_fn={2})'.format(self.__class__.__name__, self.num_bit, self.quantize_weight)

class PACT_Embedding(nn.Embedding):
    def __init__(self, vocab_size, embedding_dim, padding_idx):
        super(Q_Embedding, self).__init__(vocab_size, embedding_dim, padding_idx)
        self.num_bit = 32
        self.quantize_weight = None

        self.weight_buffer = AverageMeter()
        self.qweight_buffer = AverageMeter()

    def initialize(self, n_bit, clip_init_val, clip_init_valn):
        self.num_bit = n_bit
        self.quantize_weight = LearnedTwosidedClippedLinearQuantization(num_bits= self.num_bit,
                                                                        init_clip_val=clip_init_val,
                                                                        init_clip_valn=clip_init_valn,
                                                                        dequantize=True,
                                                                        inplace = False)
        
    def forward(self, input):
        if self.num_bit < 32:
            qweight = self.quantize_weight(self.weight)

            self.weight_buffer.update(self.weight)
            self.qweight_buffer.update(qweight)
        else:
            qweight = self.weight
        
        return F.embedding(input, qweight)
            

        
class PACT_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(Q_Linear, self).__init__(in_features, out_features, bias=True)                                
        self.num_bit = 32
        self.weight_old = None
        self.quantize_weight = None
        self.q_idx = None

        self.weight_buffer = AverageMeter()
        self.qweight_buffer = AverageMeter()

    def initialize(self, n_bit, clip_init_val, clip_init_valn):
            #print('initializing num_bit_weight to %d' %(nbit))  
        self.num_bit = n_bit
        self.quantize_weight = LearnedTwosidedClippedLinearQuantization( num_bits = self.num_bit,
                                                                         init_clip_val = clip_init_val, 
                                                                         init_clip_valn = clip_init_valn,
                                                                         dequantize = True, 
                                                                         inplace = False)                                                                     
                
    def _weight_quant(self):
        qweight = self.quantize_weight(self.weight)
        return qweight

    def forward(self, input):
        if self.num_bit < 32:

            self.quantize_weight.num_bits = self.num_bit
            qweight = self._weight_quant()

            self.weight_buffer.update(self.weight)
            self.qweight_buffer.update(qweight)
            # qbias = self.quantize_weight(self.bias)
            # qinput = self.quantize_act(input)

        else:
            qweight = self.weight

        qoutput = F.linear(input, qweight, self.bias)
        #sqoutput = F.linear(qinput, qweight, qbias)
        return qoutput
       
    def __repr__(self):
        return '{0}(num_bits_weight={1}, w_quant_fn={2})'.format(self.__class__.__name__, self.num_bit, self.quantize_weight)


def initialize(model, a_bit, w_bit, act=False, weight=False, clip_init_val = None, clip_init_valn = None, weight_init_val = None):
    
    for name, module in model.named_modules():

        if isinstance(module, (Q_ReLU, Q_ReLU6, Q_Swish, Q_Sym)) and act:
            module.initialize(a_bit, clip_init_val, clip_init_valn)
            module.num_bit = a_bit

        if isinstance(module, (Q_Conv2d,Q_Linear)) and weight:
            module.initialize(w_bit, weight_init_val, -1*weight_init_val)
            module.num_bit = w_bit

    model.cuda()

    # def initialize_hook(module, input, output):
    #     if isinstance(module, (Q_ReLU, Q_ReLU6, Q_Swish, Q_Sym)) and act:
    #         module.initialize(n_bit, clip_init_val, clip_init_valn)

    #     if isinstance(module, (Q_Conv2d,Q_Linear)) and weight:
    #         module.initialize(n_bit, clip_init_val, clip_init_valn)

    # hooks = []
        
    # for name, module in model.named_modules():
    #     #pdb.set_trace()  
    #     if hasattr(module, 'num_bit'):
    #         hook = module.register_forward_hook(initialize_hook)
    #         hooks.append( (name,hook) )
    
    # model.train()
    # model.cpu()
    # for i, (input, target) in enumerate(loader):
    #     with torch.no_grad():
    #         if isinstance(model, nn.DataParallel):
    #             output = model.module(input)
    #         else:
    #             output = model(input)
    #     break

    # #pdb.set_trace()
    # model.cuda()
    # for _, hook in hooks:
    #     hook.remove()  


class QuantOps(object):
    initialize = initialize
    Conv2d = Q_Conv2d
    ReLU = Q_ReLU
    Swish = Q_Swish
    QLinear = Q_Linear
    ReLU6 = Q_ReLU6
    Sym = Q_Sym
    Sequential = Q_Sequential
    QEmbedding = Q_Embedding

