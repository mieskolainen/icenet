# Convolutional Neural Nets
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_MAXO(nn.Module):
    """
    Dual (simultaneous) input network [image tensors x global vectors]
    """
    
    # Note: MaxPool(Relu(x)) = Relu(MaxPool(x))

    def __init__(self, D, C, nchannels=1, nrows=32, ncols=32,
                    dropout_cnn=0.0, mlp_dim=50, num_units=6, dropout_mlp=0.1):
        super(CNN_MAXO, self).__init__()
        
        # -------------------------------------------
        # CNN BLOCK

        self.C           = C
        self.dropout_cnn = dropout_cnn

        # Convolution (feature block) pipeline
        self.block1 = nn.Sequential(

            nn.Conv2d(in_channels=nchannels, out_channels=32, kernel_size=3, padding=1),
            # no batchnorm in the first layer
            nn.ReLU(),
            nn.MaxPool2d(2), # 2x2 window

            nn.Conv2d(in_channels=32,  out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 2x2 window
            nn.Dropout2d(p = self.dropout_cnn),

            nn.Conv2d(in_channels=32,  out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 2x2 window
            nn.Dropout2d(p = self.dropout_cnn),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 2x2 window
            nn.Dropout2d(p = self.dropout_cnn)
        )

        # Determine the intermediate dimension Z with a test input
        x = torch.tensor(np.ones((1, nchannels, nrows, ncols)), dtype=torch.float)
        dim = self.block1(x).shape
        self.Z = dim[1]*dim[2]*dim[3]

        # -------------------------------------------
        # MAXOUT BLOCK

        self.D         = D
        self.mlp_dim   = mlp_dim
        self.num_units = num_units
        self.dropout   = nn.Dropout(p = dropout_mlp)

        # Network modules
        self.fc1_list  = nn.ModuleList()
        self.fc2_list  = nn.ModuleList()

        
        for _ in range(self.num_units):
            self.fc1_list.append(nn.Linear(self.D + self.Z, mlp_dim))
            nn.init.xavier_normal_(self.fc1_list[-1].weight) # xavier init
            
            self.fc2_list.append(nn.Linear(mlp_dim, self.C))
            nn.init.xavier_normal_(self.fc2_list[-1].weight) # xavier init

        # -------------------------------------------

    def maxout(self, x, layer_list):
        """ MAXOUT layer
        """
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(layer(x), max_output)
        return max_output


    def forward(self, data):
        """
        Input data dictionary with members
            'x' : as image tensor
            'u'' : global feature tensor

        or a class with data.x and data.u
        """

        #print(f"forward: {data['x'].shape}")
        #print(f"forward: {data['u'].shape}")
        
        if type(data) is dict:
            X = data['x']
            U = data['u']
        else:
            X = data.x
            U = data.u
        
        # print(f'\nINPUT: {x.shape}')
        x1 = self.block1(X)
        # print(f'\nAFTER BLOCK 1: {x.shape}')
        x1 = x1.view(-1, self.Z)

        # ***
        x = torch.cat((x1, U), 1)
        # ***

        x = self.maxout(x, self.fc1_list)
        x = self.dropout(x)
        x = self.maxout(x, self.fc2_list)

        return x

    # Returns softmax probability
    def softpredict(self, x) :
        #if self.training:
        #    return F.log_softmax(self.forward(x), dim=-1) # Numerically more stable
        #else:
        return F.softmax(self.forward(x), dim=-1)


class CNN(nn.Module):
    # Note: MaxPool(Relu(x)) = Relu(MaxPool(x))

    def __init__(self, C, nchannels=1, nrows=32, ncols=32, dropout_cnn=0.0, dropout_mlp=0.5, mlp_dim=128):
        super(CNN, self).__init__()

        self.C           = C
        self.dropout_cnn = dropout_cnn
        self.dropout_mlp = dropout_mlp
        self.mlp_dim     = mlp_dim

        # Convolution (feature block) pipeline
        self.block1 = nn.Sequential(

            nn.Conv2d(in_channels=nchannels, out_channels=32, kernel_size=3, padding=1),
            # no batchnorm in the first layer
            nn.ReLU(),
            nn.MaxPool2d(2), # 2x2 window

            nn.Conv2d(in_channels=32,  out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 2x2 window
            nn.Dropout2d(p = self.dropout_cnn),

            nn.Conv2d(in_channels=32,  out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 2x2 window
            nn.Dropout2d(p = self.dropout_cnn),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 2x2 window
            nn.Dropout2d(p = self.dropout_cnn)
        )
        
        # Determine the intermediate dimension Z with a test input
        x = torch.tensor(np.ones((1, nchannels, nrows, ncols)), dtype=torch.float)
        dim = self.block1(x).shape
        self.Z = dim[1]*dim[2]*dim[3]
        
        # Classifier pipeline
        self.block2 = nn.Sequential(
            nn.Linear(self.Z, self.mlp_dim),
            nn.ReLU(),
            nn.Dropout(p = self.dropout_mlp),
            nn.Linear(self.mlp_dim, self.C),
        )

    def forward(self, x):
        #print(f'\nINPUT: {x.shape}')

        x = self.block1(x)
        #print(f'\nAFTER BLOCK 1: {x.shape}')

        x = x.view(-1, self.Z)
        #print(f'\nAFTER VIEW: {x.shape}')

        x = self.block2(x)
        #print(f'\nAFTER BLOCK 2: {x.shape}')
        return x

    # Returns softmax probability
    def softpredict(self, x) :
        #if self.training:
        #    return F.log_softmax(self.forward(x), dim=-1) # Numerically more stable
        #else:
        return F.softmax(self.forward(x), dim=-1)
