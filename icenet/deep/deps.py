# Permutation Equivariant Networks
# 
# https://arxiv.org/abs/1703.06114
# https://arxiv.org/abs/1812.09902
# 
# 
# m.mieskolainen@imperial.ac.uk, 2024

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
    def __init__(self, D, C, z_dim, out_dim=None, phi_layers=3, rho_layers=3, pool='max', dropout=0.1, **kwargs):
        """
        Args:
            D:        Input dimension
            C:        Number of classes
            z_dim:    Latent dimension
            out_dim:  Output dimension
            pool:     Pooling operation type: 'max','mean' or 'max1','mean1' (multi dimensional or single)
            dimtype:  'multi' or 'single'
            dropout:  Dropout regularization
        """
        super(DEPS, self).__init__()
        
        self.D       = D
        self.z_dim   = z_dim
        self.C       = C
        self.dropout = dropout
        
        if out_dim is None:
            self.out_dim = C
        else:
            self.out_dim = out_dim
        
        phi_channels = [D]
        for i in range(phi_layers - 1): phi_channels.append(z_dim)
        phi_channels.append(z_dim)

        rho_channels = [z_dim]
        for i in range(rho_layers - 1): rho_channels.append(z_dim)
        rho_channels.append(self.out_dim)

        if   pool == 'max':
            accumulator = PEN_max
        elif pool == 'mean':
            accumulator = PEN_mean
        elif pool == 'max1':
            accumulator = PEN1_max
        elif pool == 'mean1':
            accumulator = PEN1_mean

        # -------------------------------------------
        # Create phi-function
        self.phi = nn.Sequential(*[
            nn.Sequential(
                accumulator(phi_channels[i-1], phi_channels[i]),
                nn.ReLU(inplace=True)
            )
            for i in range(1,len(phi_channels))
        ])
        # -------------------------------------------

        # -------------------------------------------
        # Create rho-function
        self.rho = nn.Sequential(*[
            nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(rho_channels[i-1], rho_channels[i]),
                nn.ReLU(inplace=True),
            )
            for i in range(1,len(rho_channels) - 1)
        ])
        
        # N.B. Last layer without activation
        self.rho = nn.Sequential(
            self.rho,
            nn.Dropout(p=dropout),
            nn.Linear(rho_channels[-2], rho_channels[-1]),
        )
        # -------------------------------------------
    
    # Forward operator
    def forward(self, x):
        x = self.phi(x)
        x = x.mean(1)
        x = self.rho(x)
        return x
    
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
