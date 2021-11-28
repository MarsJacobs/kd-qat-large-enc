import torch
import torch.nn as nn


def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)

def sawb_quantize_param(out, num_bits):
    dequantize=True
    inplace=False
    scale, zero_point = asymmetric_linear_quantization_params(num_bits, 0, 1, signed=False)
    clip_val = sawb_quantization_params(num_bits, out)
    out = out.mul(1/clip_val).clamp(-1, 1).mul(0.5).add(0.5)
    out = LinearQuantizeSTE.apply(out, scale, zero_point, dequantize, inplace)
    out = (2 * out - 1) * clip_val
    return out

def _prep_saturation_val_tensor(sat_val):
    is_scalar = not isinstance(sat_val, torch.Tensor)

    if not isinstance(sat_val, torch.Tensor):
        out = torch.tensor(sat_val)
    else:
        out = sat_val.clone().detach()

    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return is_scalar, out

def linear_quantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.mul_(scale).sub_(zero_point).round_()
        return input
    if isinstance(scale, torch.Tensor):
        return torch.round(scale.to(input.device) * input - zero_point.to(input.device)) # HACK for PACT
    else:
        return torch.round(scale * input - zero_point)


def linear_dequantize(input, scale, zero_point, inplace=False):
    if inplace:
        input.add_(zero_point).div_(scale)
        return input
    if isinstance(scale, torch.Tensor):
       # print('inside of dequantize: scale{}, zero_point{}, and input{},'.format(scale, zero_point, input)) 
        return (input + zero_point.to(input.device)) / scale.to(input.device) # HACK for PACT
    else:
        return (input + zero_point) / scale


def sawb_quantization_params(num_bits, out):
    with torch.no_grad():
        x = out.flatten()
        mu = x.abs().mean()
        std = x.mul(x).mean().sqrt()

       # dic_coeff = {2:(3.212, -2.178), 3:(7.509, -6.892), 4:(12.68, -12.80), 5:(17.74, -18.64)}
        dic_coeff = {2:(3.12, -2.064), 3:(7.509, -6.892), 4:(12.68, -12.80), 5:(17.74, -18.64)}
        if num_bits > 5:
            raise ValueError('SAWB not implemented for num_bits={}'.format(num_bits))
        coeff = dic_coeff[num_bits]
        clip_val = coeff[1] * mu + coeff[0] * std

        return clip_val

class LinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale, zero_point, dequantize, inplace):
        if inplace:
            ctx.mark_dirty(input)
        output = linear_quantize(input, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator
        return grad_output, None, None, None, None

class LearnedTwosidedClippedLinearQuantizeSTE(torch.autograd.Function):
    """two-sided pact quantization for activation"""
    @staticmethod
    def forward(ctx, input, clip_val, clip_valn, num_bits, dequantize, inplace):
        ctx.save_for_backward(input, clip_val, clip_valn)
        if inplace:
            ctx.mark_dirty(input)
        scale, zero_point = asymmetric_linear_quantization_params(num_bits, clip_valn.data, clip_val.data, integral_zero_point=False, signed=False)
        if isinstance(clip_val, torch.Tensor):
            output = torch.where(input>clip_val, torch.ones_like(input)*clip_val, input) 
            output = torch.where(output<clip_valn, torch.ones_like(input)*clip_valn, output) 
        else:
            output = clamp(input, clip_valn.data, clip_val.data, inplace)
        output = linear_quantize(output, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val, clip_valn = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input = torch.where(input<=clip_valn, torch.zeros_like(grad_input), grad_input) 
        grad_input = torch.where(input>=clip_val, torch.zeros_like(grad_input), grad_input) 


        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(input<clip_val, torch.zeros_like(grad_alpha), grad_alpha) 
        grad_alpha = grad_alpha.sum().expand_as(clip_val)

        grad_alphan = grad_output.clone()
        grad_alphan = torch.where(input>clip_valn, torch.zeros_like(grad_alphan), grad_alphan) 
        grad_alphan = grad_alphan.sum().expand_as(clip_valn)
        # Straight-through estimator for the scale factor calculation
        return grad_input, grad_alpha, grad_alphan, None, None, None

class LearnedTwosidedClippedLinearQuantization(nn.Module):
    def __init__(self, num_bits, init_clip_valn, init_clip_val, dequantize=True, inplace=False, CGGrad=False):
        """two-sided original PACT"""
        super(LearnedTwosidedClippedLinearQuantization, self).__init__()
        self.num_bits = num_bits
        
        if isinstance(init_clip_val, torch.Tensor) and isinstance(init_clip_valn, torch.Tensor):
            self.clip_val = nn.Parameter(init_clip_val)
            self.clip_valn = nn.Parameter(init_clip_valn)
        elif not isinstance(init_clip_val, torch.Tensor) and not isinstance(init_clip_valn, torch.Tensor):
            self.clip_val = nn.Parameter(torch.Tensor([init_clip_val]))
            self.clip_valn = nn.Parameter(torch.Tensor([init_clip_valn]))
        else:
            raise ValueError('[JC] SENQNN: init_clip_val and init_clip_valn in LearnedTwosidedClippedLinearQuantization should be the same instance type.')
             
        self.dequantize = dequantize
        self.inplace = inplace
        self.CGGrad = CGGrad

    def forward(self, input):
        if self.CGGrad:
            input = CGLearnedTwosidedClippedLinearQuantizeSTE.apply(input, self.clip_val, self.clip_valn, self.num_bits, self.dequantize, self.inplace)
            pass
        else:
            input = LearnedTwosidedClippedLinearQuantizeSTE.apply(input, self.clip_val, self.clip_valn, self.num_bits, self.dequantize, self.inplace)

        return input

    def __repr__(self):
        clip_str = ', pos-clip={}, neg-clip={}'.format(self.clip_val[0], self.clip_valn[0])
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(num_bits={1}{2}{3})'.format(self.__class__.__name__, self.num_bits, clip_str, inplace_str)

class LearnedClippedLinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, dequantize, inplace):
        ctx.save_for_backward(input, clip_val)
        if inplace:
            ctx.mark_dirty(input)
        scale, zero_point = asymmetric_linear_quantization_params(num_bits, 0, clip_val, signed=False)
        if isinstance(clip_val, torch.Tensor):
            if input.min() < 0:
                raise ValueError('[JC] SENQNN: input to ClippedLinearQuantization should be non-negative.')
            output = torch.where(input>clip_val, torch.ones_like(input)*clip_val, input) ##naigang: to combine last two lines for speedup
        else:
            output = clamp(input, 0, clip_val, inplace)
        output = linear_quantize(output, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        grad_input = grad_output.clone()
        #Naigang: modify last two lines for speedup
        grad_input = torch.where(input<0, torch.zeros_like(grad_input), grad_input) 
        grad_input = torch.where(input>clip_val, torch.zeros_like(grad_input), grad_input) 

        grad_alpha = grad_output.clone()
        grad_alpha = torch.where(input<clip_val, torch.zeros_like(grad_alpha), grad_alpha) 
        grad_alpha = grad_alpha.sum().expand_as(clip_val)
        return grad_input, grad_alpha, None, None, None

class TwosidedClippedLinearQuantization(nn.Module):
    def __init__(self, num_bits, clip_val, dequantize=True, inplace=False):
        """hard clip for two sided quantization"""
        super(TwosidedClippedLinearQuantization, self).__init__()
        self.num_bits = num_bits
        self.clip_val = clip_val
        self.scale, self.zero_point = symmetric_linear_quantization_params(num_bits, clip_val)
        self.dequantize = dequantize
        self.inplace = inplace

    def forward(self, input):
        input = clamp(input, 0, self.clip_val, self.inplace)
        input = LinearQuantizeSTE.apply(input, self.scale, self.zero_point, self.dequantize, self.inplace)
        return input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(num_bits={1}, clip_val={2}{3})'.format(self.__class__.__name__, self.num_bits, self.clip_val,
                                                           inplace_str)

class LearnedClippedLinearQuantization(nn.Module):
    def __init__(self, num_bits, init_act_clip_val, dequantize=True, inplace=False, CGGrad=False):
        """single sided original PACT"""
        super(LearnedClippedLinearQuantization, self).__init__()
        self.num_bits = num_bits
        self.clip_val = init_act_clip_val

        if isinstance(init_act_clip_val, torch.Tensor):
            self.clip_val = nn.Parameter(init_act_clip_val)
        elif isinstance(init_act_clip_val, nn.Parameter):
            self.clip_val = init_act_clip_val
        else:
            self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]))

        self.dequantize = dequantize
        self.inplace = inplace
        self.CGGrad = CGGrad

    def forward(self, input):

        input = LearnedClippedLinearQuantizeSTE.apply(input, self.clip_val, self.num_bits, self.dequantize, self.inplace)
        return input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        if isinstance(self.clip_val, torch.Tensor):
            c_val = self.clip_val[0]
        else:
            c_val = self.clip_val
        return '{0}(num_bits={1}, clip_val={2}{3})'.format(self.__class__.__name__, self.num_bits, c_val, inplace_str)




class Weight_LearnedClippedLinearQuantization(nn.Module):
    def __init__(self, num_bits, init_act_clip_val, dequantize=True, inplace=False, CGGrad=False):
        """single sided original PACT"""
        super(Weight_LearnedClippedLinearQuantization, self).__init__()
        self.num_bits = num_bits
        if isinstance(init_act_clip_val, torch.Tensor):
            self.clip_val = nn.Parameter(init_act_clip_val)
        else:
            self.clip_val = nn.Parameter(torch.Tensor([init_act_clip_val]))

        self.dequantize = dequantize
        self.inplace = inplace
        self.CGGrad = CGGrad

    def forward(self, input):
        if self.CGGrad:
           raise ValueError('No CGGread Mode yet')
            #input = CGLearnedClippedLinearQuantizeSTE.apply(input, self.clip_val, self.num_bits, self.dequantize, self.inplace)
        else:
            input = Weight_LearnedClippedLinearQuantizeSTE.apply(input, self.clip_val, self.num_bits, self.dequantize, self.inplace)
        return input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return '{0}(num_bits={1}, clip_val={2}{3})'.format(self.__class__.__name__, self.num_bits, self.clip_val[0], inplace_str)
    

def symmetric_linear_quantization_params(num_bits, saturation_val):
    is_scalar, sat_val = _prep_saturation_val_tensor(saturation_val)

    if any(sat_val < 0):
        raise ValueError('Saturation value must be >= 0')

    # Leave one bit for sign
    n = 2 ** (num_bits - 1) - 1

    # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
    # value to 'n', so the scale becomes 1
    sat_val[sat_val == 0] = n
    scale = n / sat_val
    zero_point = torch.zeros_like(scale)

    if is_scalar:
        # If input was scalar, return scalars
        return scale.item(), zero_point.item()
    return scale, zero_point

def asymmetric_linear_quantization_params(num_bits, saturation_min, saturation_max,
                                          integral_zero_point=True, signed=False):
    with torch.no_grad():                                      
        scalar_min, sat_min = _prep_saturation_val_tensor(saturation_min)
        scalar_max, sat_max = _prep_saturation_val_tensor(saturation_max)
        is_scalar = scalar_min and scalar_max

        if scalar_max and not scalar_min:
            sat_max = sat_max.to(sat_min.device)
        elif scalar_min and not scalar_max:
            sat_min = sat_min.to(sat_max.device)
       
#        print('device {}, sat_min {}'.format(sat_min.device.index, sat_min))
#        print('device {}, sat_max {}'.format(sat_min.device.index, sat_max))
        
       # if any(sat_min > sat_max):
       #     raise ValueError('saturation_min must be smaller than saturation_max, sat_min={}, sat_max={}'.format(sat_min, sat_max))

        n = 2 ** num_bits - 1

        # Make sure 0 is in the range
        sat_min = torch.min(sat_min, torch.zeros_like(sat_min))
        sat_max = torch.max(sat_max, torch.zeros_like(sat_max))

        diff = sat_max - sat_min
       # print('diff is :', diff)
        # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
        # value to 'n', so the scale becomes 1
        diff[diff == 0] = n

        scale = n / diff
        zero_point = scale * sat_min
        if integral_zero_point:
            zero_point = zero_point.round()
        if signed:
            zero_point += 2 ** (num_bits - 1)
        if is_scalar:
            return scale.item(), zero_point.item()
#        print('device {}, scale {}'.format(scale.device.index, scale))
#        print('device {}, zero_point {}'.format(zero_point.device.index, zero_point))
        return scale, zero_point


class Weight_LearnedClippedLinearQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, dequantize, inplace):
        ctx.save_for_backward(input, clip_val)
        if inplace:
            ctx.mark_dirty(input)

        #edit: change line location between clipping and finding stepsize ##########
        if isinstance(clip_val, torch.Tensor):
            output = torch.where(input>clip_val, torch.ones_like(input)*clip_val, input) ##naigang: to combine last two lines for speedup
            output = torch.where(input<-1*clip_val, torch.ones_like(input)*-1*clip_val, output) ##naigang: to combine last two lines for speedup
        else:
            output = clamp(input, -1*clip_val.data, clip_val, inplace)
        
        #edit: normalize weight
        output = output / (2*clip_val)
        
        #edit: shift normalized weight
        output = output + 0.5        

        #edit: calculate step size between 0 and 1, because weight is normalized        
        scale, zero_point = asymmetric_linear_quantization_params(num_bits, 0, 1, signed=False)

        output = linear_quantize(output, scale, zero_point, inplace)
        if dequantize:
            output = linear_dequantize(output, scale, zero_point, inplace)


        #edit: denormalization
        output = 2*clip_val*(output-0.5)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        grad_input = grad_output.clone()
        #Naigang: modify last two lines for speedup
        grad_input = torch.where(input>clip_val, torch.zeros_like(grad_input), grad_input) 
        #edit     
        grad_input = torch.where(input<-1*clip_val, torch.zeros_like(grad_input), grad_input)

        grad_alpha = grad_output.clone()
        #edit
        grad_alpha = torch.where(input>clip_val, grad_alpha, torch.zeros_like(grad_alpha)) 
        grad_alpha = torch.where(input<-1*clip_val, -1*grad_alpha, grad_alpha) 

        grad_alpha = grad_alpha.sum().expand_as(clip_val)
        return grad_input, grad_alpha, None, None, None


