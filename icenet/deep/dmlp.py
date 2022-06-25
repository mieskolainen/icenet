# Deep MLP
#
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def MLP(channels, activation='relu', batch_norm=True):
    """
    Return a Multi Layer Perceptron with an arbitrary number of layers.

    WITHOUT LAST ACTIVATION
    
    Args:
        channels   : input structure, such as [128, 64, 64] for a 3-layer network.
        batch_norm : batch normalization
    Returns:
        nn.sequential object
    
    """
    print(__name__ + f'.MLP: Using {activation} activation')

    if batch_norm:
        return nn.Sequential(*[
            nn.Sequential(
                nn.Linear(channels[i - 1], channels[i]),
                nn.ReLU() if activation == 'relu' else nn.Tanh(),
                nn.BatchNorm1d(channels[i])
            )
            for i in range(1,len(channels) - 1)
        ],
            nn.Linear(channels[-2], channels[-1]) # N.B. Last without activation!
        )
    
    else:
        return nn.Sequential(*[
            nn.Sequential(
                nn.Linear(channels[i - 1], channels[i]),
                nn.ReLU() if activation == 'relu' else nn.Tanh()
            )
            for i in range(1,len(channels) - 1)
        ], 
            nn.Linear(channels[-2], channels[-1]) # N.B. Last without activation!
        )


def MLP_ALL_ACT(channels, activation='relu', batch_norm=True):
    """
    Return a Multi Layer Perceptron with an arbitrary number of layers.
    
    ALL LAYERS WITH ACTIVATION
    
    Args:
        channels   : input structure, such as [128, 64, 64] for a 3-layer network.
        batch_norm : batch normalization
    Returns:
        nn.sequential object
    
    """
    print(__name__ + f'.MLP_all_act: Using {activation} activation')

    if batch_norm:
        return nn.Sequential(*[
            nn.Sequential(
                nn.Linear(channels[i - 1], channels[i]),
                nn.ReLU() if activation == 'relu' else nn.Tanh(),
                nn.BatchNorm1d(channels[i])
            )
            for i in range(1,len(channels))
        ]
        )
    
    else:
        return nn.Sequential(*[
            nn.Sequential(
                nn.Linear(channels[i - 1], channels[i]),
                nn.ReLU() if activation == 'relu' else nn.Tanh()
            )
            for i in range(1,len(channels))
        ]
        )



class DMLP(nn.Module):

    def __init__(self, D, C, mlp_dim = [128,64], activation='relu', batch_norm=True):
        """
        Args:
            D       : Input dimension
            mlp_dim : hidden layer dimensions (array)
            C       : Output dimension
        """
        super(DMLP,self).__init__()

        self.D = D
        self.C = C
        
        # Add input dimension
        channels = [D]

        # Add hidden dimensions
        for i in range(len(mlp_dim)):
            channels.append(mlp_dim[i])

        # Add output dimension
        channels.append(C)
        
        self.mlp = MLP(channels, activation=activation, batch_norm=batch_norm)

    def forward(self,x):
        
        x = self.mlp(x)
        return x

    def softpredict(self,x) :
        """ Softmax probability
        """
        if self.training:
            return F.log_softmax(self.forward(x), dim=-1) # Numerically more stable
        else:
            return F.softmax(self.forward(x), dim=-1)

    def binarypredict(self,x) :
        """ Return maximum probability class
        """
        prob = list(self.softpredict(x).detach().numpy())
        return np.argmax(prob, axis=1)
