# Classic convolutional (LeNet) neural nets
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import numpy as np

import torch
import torch.nn as nn


class LeNet3C(nn.Module):
    """ 3-channel convolution network.
    """
    def __init__(self, D, C):

        super(CONV2, self).__init__()

        self.D = D # Input dimensions (NOT ACTIVE)
        self.C = C # Number of output classes
        
        # Convolution pipeline
        self.block1 = nn.Sequential(

            # 3 input image channels -> 10 output channels/filters
            nn.Conv2d(3, 10, kernel_size=5),
            nn.MaxPool2d(2), # 2x2 window
            nn.ReLU(),

            # 10 output channels/filters -> 20 output channels/filters
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2), # 2x2 window
            nn.Dropout2d(p = 0.5),
        )
        
        # Classifier pipeline
        self.block2 = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(50, C),
        )

    def forward(self, x):
        features = self.block1(x)
        features = features.view(x.shape[0], -1)
        logits   = self.block2(features)
        return logits

    # Returns softmax probability
    def softpredict(self,x) :
        return F.softmax(self.forward(x), dim=1)
