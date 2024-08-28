# Deep MLP
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from icenet.deep.deeptools import Multiply


def get_act(act: str = 'relu'):
    """
    Returns torch activation function
    
    Args:
        act:  activation function type
    """
    if   act == 'relu':
        return nn.ReLU()
    elif act == 'tanh':
        return nn.Tanh()
    elif act == 'silu':
        return nn.SiLU()
    elif act == 'elu':
        return nn.ELU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'softplus':
        return nn.Softplus()
    else:
        raise Exception(f'Uknown act "{act}" chosen')

class LinearLayer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, skip_connections=False, activation: str='relu',
                 layer_norm: bool=False, batch_norm: bool=False, dropout: float=0.0,
                 act_after_norm=True):
        
        super(LinearLayer, self).__init__()
        
        self.layer = torch.nn.Linear(dim_in, dim_out)
        self.skip_connections = skip_connections
        self.act_after_norm = act_after_norm
        
        if not self.act_after_norm:  # Just for print() ordering
            self.act = get_act(activation)
        
        if layer_norm:
            self.ln = nn.LayerNorm(dim_out)
        if batch_norm:
            self.bn = nn.BatchNorm1d(dim_out)
        if dropout > 0:
            self.do = nn.Dropout(dropout, inplace=False)

        if self.act_after_norm:
            self.act = get_act(activation)
        
    def forward(self, x):
        y = self.layer(x)
        
        if not self.act_after_norm:
            y = self.act(y)
        
        if hasattr(self, 'ln'):
            y = self.ln(y)
        if hasattr(self, 'bn'):
            y = self.bn(y)
        if hasattr(self, 'do'):
            y = self.do(y)
        
        if self.act_after_norm:
            y = self.act(y)
        
        # Add skip connection if possible dimension wise
        if self.skip_connections and (y.shape[-1] == x.shape[-1]):
            return y + x
        else:
            return y


def MLP(layers: List[int], activation: str='relu', layer_norm:bool = False, batch_norm: bool=False,
        dropout: float=0.0, last_act: bool=False, skip_connections=False, act_after_norm=True):
    """
    Return a Multi Layer Perceptron with an arbitrary number of layers.
    
    Args:
        layers     : input structure, such as [128, 64, 64] for a 3-layer network.
        activation : activation function
        layer_norm : layer normalization
        batch_norm : batch normalization
        dropout    : dropout regularization
        skip_connections: skip connections active
        last_act   : apply activation function after the last layer
        act_after_norm  : activation function application order
    
    Returns:
        nn.sequential object
    """
    print(__name__ + f'.MLP: {layers} | activation {activation} | layer_norm {layer_norm} | batch_norm {batch_norm} | dropout {dropout} | skip_connections = {skip_connections} | act_after_norm {act_after_norm} | last_act {last_act}')
    
    # Last layer without activation (or any other operations)
    if not last_act:
        
        return nn.Sequential(*[
            LinearLayer(dim_in=layers[i-1], dim_out=layers[i],
                        activation=activation, layer_norm=layer_norm, batch_norm=batch_norm,
                        dropout=dropout, skip_connections=skip_connections, act_after_norm=act_after_norm)
            for i in range(1,len(layers) - 1)
        ],
            nn.Linear(layers[-2], layers[-1])
        )
    
    # All operations after each layer, including the last layer
    else:
        
        return nn.Sequential(*[
            LinearLayer(dim_in=layers[i-1], dim_out=layers[i],
                        activation=activation, layer_norm=layer_norm, batch_norm=batch_norm,
                        dropout=dropout, skip_connections=skip_connections, act_after_norm=act_after_norm)
            for i in range(1,len(layers))
            ]
        )


def MLP_ALL_ACT(layers: List[int], activation: str='relu', layer_norm: bool=False, batch_norm: bool=False,
                dropout: float=0.0, skip_connections=False, act_after_norm: bool=True):
    """
    Return a Multi Layer Perceptron with an arbitrary number of layers.
    
    All layers with the activation + other operations applied.
    """
    
    return MLP(layers=layers, activation=activation, layer_norm=layer_norm, batch_norm=batch_norm,
               dropout=dropout, last_act=True, skip_connections=skip_connections, act_after_norm=act_after_norm)


class DMLP(nn.Module):
    
    def __init__(self, D, C, out_dim=None, mlp_dim = [128, 64],
                 activation='relu', layer_norm=False, batch_norm=False, dropout=0.0, skip_connections=False,
                 act_after_norm=True, last_tanh=False, last_tanh_scale=10.0, **kwargs):
        """
        Args:
            D       : Input dimension
            C       : Number of classes
            mlp_dim : hidden layer dimensions (array)
            out_dim : Output dimension
        """
        super(DMLP, self).__init__()
        
        self.D = D
        self.C = C
        
        if out_dim is None:
            self.out_dim = C
        else:
            self.out_dim = out_dim
        
        # Add input dimension
        layers = [D]

        # Add hidden dimensions
        for i in range(len(mlp_dim)):
            layers.append(mlp_dim[i])

        # Add output dimension
        layers.append(self.out_dim)
        
        self.mlp = MLP(layers, activation=activation, skip_connections=skip_connections,
                       layer_norm=layer_norm, batch_norm=batch_norm, dropout=dropout, act_after_norm=act_after_norm)
        
        # Add extra final squeezing activation and post-scale aka "soft clipping"
        if last_tanh:
            self.mlp.add_module("tanh",  nn.Tanh())
            self.mlp.add_module("scale", Multiply(last_tanh_scale))
    
    def forward(self,x):
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
