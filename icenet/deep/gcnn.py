# Graph Convolution Network
#
# https://arxiv.org/abs/1609.02907
# Adapted from original code
# 
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GCN_layer(Module):
    """ Graph Convolution Network Layer
    """

    def __init__(self, D_in, D_out, bias=True):
        super(GCN_layer, self).__init__()

        # Input / output dimensions
        self.D_in  = D_in
        self.D_out = D_out

        self.weight = Parameter(torch.FloatTensor(D_in, D_out))
        if bias:
            self.bias = Parameter(torch.FloatTensor(D_out))
        else:
            self.register_param('bias', None)
        self.reset_param()

    def reset_param(self):
        """ 1 / sqrt(N) normalization """

        std = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, x, adj_matrix):
        """ Forward operator """

        support = torch.mm(x, self.weight)        # matrix multiplication
        output  = torch.spmm(adj_matrix, support) # sparse matrix multiplication
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
    def __repr__(self):
        return self.__class__.__name__ + ' ('+ str(self.D_in) + ' -> ' + str(self.D_out) + ')'


class GCN(nn.Module):
    """ Graph Convolution Network
    """

    def __init__(self, D, Z, C, out_dim=None, dropout=0.5):
        """ 
        Args:
            D       : Input dimension
            Z       : Latent (hidden) dimension
            C       : Output dimension
            dropout : Dropout parameter
        """
        super(GCN, self).__init__()

        self.D = D
        self.Z = Z
        self.C = C
        
        if out_dim is None:
            self.out_dim = C
        else:
            self.out_dim = out_dim
        
        self.gc1 = GCN_layer(self.D, self.Z)
        self.gc2 = GCN_layer(self.Z, self.out_dim)
        self.dropout = dropout

    def forward(self, x, adj_matrix):
        y = self.gc1(x, adj_matrix)
        y = F.relu(y)
        y = F.dropout(y, self.dropout, training=self.training)
        y = self.gc2(y, adj_matrix)
        
        return y
    
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
