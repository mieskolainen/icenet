# Deep MLP
#
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from icenet.deep import graph

class DMLP(nn.Module):

    def __init__(self, D, C, mlp_dim = [128,64], batch_norm = True):
        """
        Args:
            D       : Input dimension
            mlp_dim : hidden layer dimensions (array)
            C       : Output dimension
        """
        super(DMLP,self).__init__()

        self.D = D
        self.C = C
        
        # Layer structure
        channels = [D]
        for i in range(len(mlp_dim)):
            channels.append(mlp_dim[i])
        channels.append(C)

        self.mlp = graph.MLP(channels, batch_norm=batch_norm)

    def forward(self,x):
        
        x = self.mlp(x)
        return x

    def softpredict(self,x) :
        """ Softmax probability
        """
        return F.softmax(self.forward(x), dim = 1)

    def binarypredict(self,x) :
        """ Return maximum probability class
        """
        prob = list(self.softpredict(x).detach().numpy())
        return np.argmax(prob, axis=1)
