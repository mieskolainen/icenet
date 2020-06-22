# MaxOUT networks
#
# https://arxiv.org/abs/1302.4389
# 
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MAXOUT(nn.Module):
    """ MAXOUT network

    """
    def __init__(self, D, C, num_units, neurons, dropout):
        """
        Args:
            D: Input dimension
            C: Output dimension
        """
        super(MAXOUT,self).__init__()

        # Input and output dimension
        self.D = D
        self.C = C
        
        # Network modules
        self.fc1_list = nn.ModuleList()
        self.fc2_list = nn.ModuleList()

        self.num_units = num_units
        self.dropout   = nn.Dropout(p = dropout)
        
        for _ in range(self.num_units):
            self.fc1_list.append(nn.Linear(self.D, neurons))
            self.fc2_list.append(nn.Linear(neurons, self.C))
            
    def forward(self,x):
        
        x = self.maxout(x, self.fc1_list)
        x = self.dropout(x)
        x = self.maxout(x, self.fc2_list)

        return x
    
    def maxout(self,x, layer_list):
        """ MAXOUT layer
        """
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output= torch.max(layer(x), max_output)
        return max_output

    def softpredict(self,x) :
        """ Softmax probability
        """
        return F.softmax(self.forward(x), dim = 1)
        
    def binarypredict(self,x) :
        """ Return max probability class.
        """        
        prob = list(self.softpredict(x).detach().numpy())
        return np.argmax(prob, axis=1)
