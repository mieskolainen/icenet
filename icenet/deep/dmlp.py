# Deep MLP
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from icenet.deep.deeptools import Multiply


def get_act(act: str):
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


def MLP(layers: List[int], activation: str='relu', batch_norm: bool=False, dropout: float=0.0, last_act: bool=False):
    """
    Return a Multi Layer Perceptron with an arbitrary number of layers.
    
    Args:
        layers     : input structure, such as [128, 64, 64] for a 3-layer network.
        activation : activation function
        batch_norm : batch normalization
        dropout    : dropout regularization
        last_act   : apply activation function after the last layer
    
    Returns:
        nn.sequential object
    """
    print(__name__ + f'.MLP: {layers} | activation {activation} | batch_norm {batch_norm} | dropout {dropout} | last_act {last_act}')
    
    if not last_act: # Without activation after the last layer
        
        if batch_norm:
            return nn.Sequential(*[
                nn.Sequential(
                    nn.Linear(layers[i - 1], layers[i]),
                    get_act(activation),
                    nn.BatchNorm1d(layers[i]),
                    nn.Dropout(dropout, inplace=False)
                )
                for i in range(1,len(layers) - 1)
            ],
                nn.Linear(layers[-2], layers[-1]) # N.B. Last without act!
            )
        else:
            return nn.Sequential(*[
                nn.Sequential(
                    nn.Linear(layers[i - 1], layers[i]),
                    get_act(activation),
                    nn.Dropout(dropout, inplace=False)
                )
                for i in range(1,len(layers) - 1)
            ], 
                nn.Linear(layers[-2], layers[-1]) # N.B. Last without act!
            )
    else:
      
        if batch_norm:
            return nn.Sequential(*[
                nn.Sequential(
                    nn.Linear(layers[i - 1], layers[i]),
                    get_act(activation),
                    nn.BatchNorm1d(layers[i]),
                    nn.Dropout(dropout, inplace=False),
                )
                for i in range(1,len(layers))
            ]
            )
        
        else:
            return nn.Sequential(*[
                nn.Sequential(
                    nn.Linear(layers[i - 1], layers[i]),
                    get_act(activation),
                    nn.Dropout(dropout, inplace=False)
                )
                for i in range(1,len(layers))
            ]
            )  

def MLP_ALL_ACT(layers: List[int], activation: str='relu', batch_norm: bool=False, dropout: float=0.0):
    """
    Return a Multi Layer Perceptron with an arbitrary number of layers.
    
    ALL LAYERS WITH THE ACTIVATION FUNCTION
    """
    
    return MLP(layers=layers, activation=activation, batch_norm=batch_norm, dropout=dropout, last_act=True)


class DMLP(nn.Module):

    def __init__(self, D, C, out_dim=None, mlp_dim = [128, 64], activation='relu', batch_norm=True, dropout=0.0, last_tanh=False, last_tanh_scale=10.0, **kwargs):
        """
        Args:
            D       : Input dimension
            C       : Number of classes
            mlp_dim : hidden layer dimensions (array)
            out_dim : Output dimension
        """
        super(DMLP,self).__init__()

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
        
        self.mlp = MLP(layers, activation=activation, batch_norm=batch_norm, dropout=dropout)
        
        # Add extra final squeezing activation and post-scale aka "soft clipping"
        if last_tanh:
            self.mlp.add_module("tanh",  nn.Tanh())
            self.mlp.add_module("scale", Multiply(last_tanh_scale))
    
    def forward(self,x):
        
        x = self.mlp(x)
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
