# Convolutional Neural Nets
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, C, dropout_cnn=0.25, dropout_mlp=0.5):
        super(CNN, self).__init__()

        self.C           = C
        self.dropout_cnn = dropout_cnn
        self.dropout_mlp = dropout_mlp

        # Convolution pipeline
        self.block1 = nn.Sequential(

            nn.Conv2d(in_channels=1,  out_channels=32, kernel_size=3, padding=1),
            #nn.MaxPool2d(2), # 2x2 window
            nn.ReLU(),

            nn.Conv2d(in_channels=32,  out_channels=32, kernel_size=3, padding=1),
            #nn.MaxPool2d(2), # 2x2 window
            nn.ReLU(),
            nn.Dropout2d(p = self.dropout_cnn),

            nn.Conv2d(in_channels=32,  out_channels=32, kernel_size=3, padding=1),
            #nn.MaxPool2d(2), # 2x2 window
            nn.ReLU(),
            nn.Dropout2d(p = self.dropout_cnn),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(2), # 2x2 window
            nn.ReLU(),
            nn.Dropout2d(p = self.dropout_cnn)
        )
        self.Z = 5*5*64
        
        # Classifier pipeline
        self.block2 = nn.Sequential(
            nn.Linear(self.Z, 128),
            nn.ReLU(),
            nn.Dropout(p = self.dropout_mlp),
            nn.Linear(128, self.C),
        )

    def forward(self, x):
        # Note: MaxPool(Relu(x)) = Relu(MaxPool(x))
        x = self.block1(x)
        #print(f'\nINPUT: {x.shape}')

        #print(f'\nBEFORE VIEW: {x.shape}')
        x = x.view(-1, self.Z)
        #print(f'\nAFTER VIEW: {x.shape}')

        x = self.block2(x)
        #print(f'\nOUTPUT: {x.shape}')
        return x

    # Returns softmax probability
    def softpredict(self,x) :
        return F.softmax(self.forward(x), dim=1)
