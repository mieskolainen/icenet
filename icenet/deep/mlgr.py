# Multinomial Logistic Regression
#
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MLGR(nn.Module):
    """ Multinomial Logistic Regression model
    """
    def __init__(self, D, C):
        """
        Args:
            D : Input dimension
            C : Output dimension
        """
        super(MLGR,self).__init__()

        self.D = D
        self.C = C
        
        # One layer
        self.layer1 = nn.Linear(self.D, self.C)

    def forward(self,x):
        
        x = self.layer1(x)
        return x

    def softpredict(self,x) :
        """ Softmax probability
        """
        #if self.training:
        #    return F.log_softmax(self.forward(x), dim=-1) # Numerically more stable
        #else:
        return F.softmax(self.forward(x), dim=-1)
    
    def binarypredict(self,x) :
        """ Return maximum probability class
        """
        prob = list(self.softpredict(x).detach().numpy())
        return np.argmax(prob, axis=1)
