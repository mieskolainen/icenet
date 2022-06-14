# Permutation Equivariant Networks
# 
# https://arxiv.org/abs/1703.06114
# https://arxiv.org/abs/1812.09902
# 
# 
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.nn.parameter import Parameter
from   torch.nn.modules.module import Module


class PEN_max(nn.Module):
    """ Permutation Equivariant Network (PEN) max-type layers.
    
    Multidimensional model.
    """
    def __init__(self, in_dim, out_dim):
        super(PEN_max, self).__init__()
        self.Gamma  = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        xm, _ = x.max(1, keepdim=True)
        xm    = self.Lambda(xm) 
        x     = self.Gamma(x)
        x     = x - xm
        return x


class PEN_mean(nn.Module):
    """ Permutation Equivariant Network (PEN) mean-type layers.

    Multidimensional model.
    """
    def __init__(self, in_dim, out_dim):
        super(PEN_mean, self).__init__()
        self.Gamma  = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x):
        xm = x.mean(1, keepdim=True)
        xm = self.Lambda(xm) 
        x  = self.Gamma(x)
        x  = x - xm
        return x


class PEN1_max(nn.Module):
    """ Permutation Equivariant Network (PEN) max-type layers.

    Single dimensional model.
    """
    def __init__(self, in_dim, out_dim):
        super(PEN1_max, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        xm, _ = x.max(1, keepdim=True)
        x     = self.Gamma(x - xm)
        return x


class PEN1_mean(nn.Module):
    """ Permutation Equivariant Network (PEN) mean-type layers.
    
    Single dimensional model.
    """
    def __init__(self, in_dim, out_dim):
        super(PEN1_mean, self).__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        xm = x.mean(1, keepdim=True)
        x  = self.Gamma(x - xm)
        return x


class DEPS(nn.Module):
    """ Permutation equivariant networks.
    """
    def __init__(self, D, z_dim, C=2, pool='max', dropout=0.5):
        """
        Args:
            D:        Input dimesion
            z_dim:    Latent dimension
            C:        Number of classes
            pool:     Pooling operation type: 'max','mean' or 'max1','mean1' (multi dimensional or single)
            dimtype:  'multi' or 'single'
            dropout:  Dropout regularization
        """
        super(DEPS, self).__init__()
        
        self.D       = D
        self.z_dim   = z_dim
        self.C       = C
        self.dropout = dropout
        
        if   pool == 'max':
            self.phi = nn.Sequential(
                PEN_max(self.D,     self.z_dim),
                nn.ReLU(inplace=True),
                PEN_max(self.z_dim, self.z_dim),
                nn.ReLU(inplace=True),
                PEN_max(self.z_dim, self.z_dim),
                nn.ReLU(inplace=True),
            )
        elif pool == 'mean':
            self.phi = nn.Sequential(
                PEN_mean(self.D,     self.z_dim),
                nn.ReLU(inplace=True),
                PEN_mean(self.z_dim, self.z_dim),
                nn.ReLU(inplace=True),
                PEN_mean(self.z_dim, self.z_dim),
                nn.ReLU(inplace=True),
            )
        elif pool == 'max1':
            self.phi = nn.Sequential(
                PEN1_max(self.D,     self.z_dim),
                nn.ReLU(inplace=True),
                PEN1_max(self.z_dim, self.z_dim),
                nn.ReLU(inplace=True),
                PEN1_max(self.z_dim, self.z_dim),
                nn.ReLU(inplace=True),
            )
        elif pool == 'mean1':
            self.phi = nn.Sequential(
                PEN1_mean(self.D,     self.z_dim),
                nn.ReLU(inplace=True),
                PEN1_mean(self.z_dim, self.z_dim),
                nn.ReLU(inplace=True),
                PEN1_mean(self.z_dim, self.z_dim),
                nn.ReLU(inplace=True),
            )

        self.ro = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(self.z_dim, self.C),
        )
        print(self)

    # Forward operator
    def forward(self, x):
        x = self.phi(x)
        x = x.mean(1)
        x = self.ro(x)
        return x
    
    # Returns softmax probability
    def softpredict(self,x) :
        if self.training:
            return F.log_softmax(self.forward(x), dim=-1) # Numerically more stable
        else:
            return F.softmax(self.forward(x), dim=-1)
    
    # Return class
    def binarypredict(self,x) :
        prob = list(self.softpredict(x).detach().numpy())
        return np.argmax(prob, axis=1)
