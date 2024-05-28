# MaxOUT networks
#
# https://arxiv.org/abs/1302.4389
# 
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MAXOUT(nn.Module):
    """ MAXOUT network

    """
    def __init__(self, D, C, num_units, neurons, dropout, out_dim=None, **kwargs):
        """
        Args:
            D: Input dimension
            C: Output dimension
        """
        super(MAXOUT,self).__init__()
        self.D = D
        self.C = C
        
        if out_dim is None:
            self.out_dim = C
        else:
            self.out_dim = out_dim
        
        # Network modules
        self.fc1_list  = nn.ModuleList()
        self.fc2_list  = nn.ModuleList()

        self.num_units = num_units
        self.dropout   = nn.Dropout(p = dropout)
        
        for _ in range(self.num_units):
            self.fc1_list.append(nn.Linear(self.D, neurons))
            nn.init.xavier_normal_(self.fc1_list[-1].weight) # xavier init
            
            self.fc2_list.append(nn.Linear(neurons, self.out_dim))
            nn.init.xavier_normal_(self.fc2_list[-1].weight) # xavier init

    def forward(self, x):
        
        x = self.maxout(x, self.fc1_list)
        x = self.dropout(x)
        x = self.maxout(x, self.fc2_list)
        return x
        
    def maxout(self, x, layer_list):
        """ MAXOUT layer
        """
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(layer(x), max_output)
        return max_output
    
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
