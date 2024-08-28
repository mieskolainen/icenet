# Multinomial Logistic Regression
#
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLGR(nn.Module):
    """ Multinomial Logistic Regression model
    """
    def __init__(self, D, C, out_dim=None):
        """
        Args:
            D : Input dimension
            C : Number of classes
        """
        super(MLGR, self).__init__()

        self.D = D
        self.C = C
        
        if out_dim is None:
            self.out_dim = C
        else:
            self.out_dim = out_dim
        
        # One layer
        self.layer1 = nn.Linear(self.D, self.out_dim)

    def forward(self,x):
        return self.layer1(x)
    
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
