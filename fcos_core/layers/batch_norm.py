# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn


from qd.layers.batch_norm import FrozenBatchNorm2d
#class FrozenBatchNorm2d(nn.Module):
    #"""
    #BatchNorm2d where the batch statistics and the affine parameters
    #are fixed
    #"""

    #def __init__(self, num_features, eps=0):
        #super(FrozenBatchNorm2d, self).__init__()
        #self.register_buffer("weight", torch.ones(num_features))
        #self.register_buffer("bias", torch.zeros(num_features))
        #self.register_buffer("running_mean", torch.zeros(num_features))
        #self.register_buffer("running_var", torch.ones(num_features))
        #self.eps = eps
        #self.num_features = num_features

    #def forward(self, x):
        #scale = self.weight * self.running_var.rsqrt()
        #bias = self.bias - self.running_mean * scale
        #scale = scale.reshape(1, -1, 1, 1)
        #bias = bias.reshape(1, -1, 1, 1)
        #return x * scale + bias

    #def extra_repr(self):
        #return '{}, eps={}'.format(len(self.weight), self.eps)
