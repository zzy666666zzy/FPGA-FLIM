import torch
import torch.nn as nn
from torch.autograd import Function
import math
from torch.nn.init import xavier_normal_

#Modified Kernel1 and Kernel2
class adder2d(nn.Module):

    def __init__(self,input_channel,output_channel,kernel1_size,kernel2_size, stride=1, padding=0, bias = False):
        super(adder2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel1_size = kernel1_size
        self.kernel2_size = kernel2_size
        
        self.Weight = torch.nn.Parameter(nn.init.normal_(torch.randn(output_channel,input_channel,kernel1_size,kernel2_size)))
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(nn.init.uniform_(torch.zeros(output_channel)))

    def forward(self, x):
        output = adder2d_function(x,self.Weight, self.stride, self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        return output

# 2-D adder functions, can be configured as 1-D
def adder2d_function(X, W, stride=1, padding=0):
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()

    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    h_out, w_out = int(h_out), int(w_out)
    X_col = torch.nn.functional.unfold(X.view(1, -1, h_x, w_x), (h_filter,w_filter), dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
    X_col = X_col.permute(1,2,0).contiguous().view(X_col.size(1),-1)
    W_col = W.view(n_filters, -1)
    
    
    out = adder.apply(W_col,X_col)
    
    out = out.view(n_filters, h_out, w_out, n_x)
    out = out.permute(3, 0, 1, 2).contiguous()
    
    return out

class adder(Function):
    @staticmethod #no need to initialize the class
    def forward(ctx, W_col, X_col):
        ctx.save_for_backward(W_col,X_col)
        W_col=W_col.unsqueeze(2)
        X_col=X_col.unsqueeze(0)
        output = -(W_col-X_col).abs()
        output = output.sum(1)
        return output

    @staticmethod
    def backward(ctx,grad_output):
        W_col,X_col = ctx.saved_tensors
        grad_W_col = ((X_col.unsqueeze(0)-W_col.unsqueeze(2))*grad_output.unsqueeze(1)).sum(2)
        grad_W_col = grad_W_col/grad_W_col.norm(p=2).clamp(min=1e-12)*math.sqrt(W_col.size(1)*W_col.size(0))/5
        grad_X_col = (-(X_col.unsqueeze(0)-W_col.unsqueeze(2)).clamp(-1,1)*grad_output.unsqueeze(1)).sum(0)
        
        return grad_W_col, grad_X_col
    
    