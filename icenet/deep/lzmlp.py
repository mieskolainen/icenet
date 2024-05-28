# Lipschitz MLP
# https://arxiv.org/abs/2202.08345
#
# m.mieskolainen@imperial.ac.uk, 2024

import torch
import torch.nn as nn
import math
import numpy as np

from icenet.deep.deeptools import Multiply
from icenet.deep.dmlp import *


class LipschitzLinear(torch.nn.Module):
    """ Lipschitz linear layer
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight       = torch.nn.Parameter(torch.empty((out_features, in_features), requires_grad=True))
        self.bias         = torch.nn.Parameter(torch.empty((out_features), requires_grad=True))
        self.c            = torch.nn.Parameter(torch.empty((1), requires_grad=True))
        self.softplus     = torch.nn.Softplus()
        self.initialize_parameters()

    def initialize_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

        # Lipschitz constant of initial weight
        W = self.weight.data
        W_abs_row_sum = torch.abs(W).sum(1)
        self.c.data   = W_abs_row_sum.max() # Approx initialization
    
    def get_lipschitz_constant(self):
        return self.softplus(self.c)

    def forward(self, input):
        lipc  = self.softplus(self.c)
        scale = lipc / torch.abs(self.weight).sum(1)
        scale = torch.clamp(scale, max=1.0)
        return torch.nn.functional.linear(input, self.weight * scale.unsqueeze(1), self.bias)


class LZMLP(torch.nn.Module):
    def __init__(self, D, C, out_dim=None, mlp_dim = [128, 64], activation='relu', batch_norm=False, dropout=0.0, last_tanh=False, last_tanh_scale=10.0, **kwargs):
        """
        Lipschitz MLP
        """
        super().__init__()

        self.D = D
        self.C = C
        
        if out_dim is None:
            self.out_dim = C
        else:
            self.out_dim = out_dim
        
        layers = []
        
        ## First layer
        layers.append(LipschitzLinear(D, mlp_dim[0]))
        layers.append(get_act(activation))
        
        if batch_norm:  layers.append(nn.BatchNorm1d(mlp_dim[0]))
        if dropout > 0: layers.append(nn.Dropout(dropout, inplace=False))
        
        ## Hidden layers
        if len(mlp_dim) >= 2:
            
            for i in range(len(mlp_dim)-1):
                layers.append(LipschitzLinear(mlp_dim[i], mlp_dim[i+1]))
                layers.append(get_act(activation))

                if batch_norm:  layers.append(nn.BatchNorm1d(mlp_dim[i+1]))
                if dropout > 0: layers.append(nn.Dropout(dropout, inplace=False))
        
        ## Output layer
        layers.append(LipschitzLinear(mlp_dim[-1], self.out_dim))

        self.mlp = nn.Sequential(*layers)

        # Add extra final squeezing activation and post-scale aka "soft clipping"
        if last_tanh:
            self.mlp.add_module("tanh",  nn.Tanh())
            self.mlp.add_module("scale", Multiply(last_tanh_scale))
    
    def get_lipschitz_loss(self):
        """ Lipschitz regularization loss
        """
        lc = 1.0
        for i in range(len(self.mlp)):
            if type(self.mlp[i]) is LipschitzLinear:
                lc = lc * self.mlp[i].get_lipschitz_constant()
        return lc
    
    def forward(self, x):
        return self.mlp(x)
    
    def softpredict(self,x) :
        """ Softmax probability
        """
        if self.out_dim > 1:
            return F.softmax(self.forward(x), dim=-1)
        else:
            return torch.sigmoid(self.forward(x))
    
    def binarypredict(self,x) :
        """ Return maximum probability class
        """
        if self.out_dim > 1:
            prob = list(self.softpredict(x).detach().numpy())
            return np.argmax(prob, axis=1)
        else:
            return np.round(self.softpredict(x).detach().numpy()).astype(int)

