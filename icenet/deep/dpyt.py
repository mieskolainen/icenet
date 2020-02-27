# Multinomial Logistic Regression net
# MaxOut MAXOUT_MLP net
#
# using PyTorch
# 
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk


import torch
import uproot
import numpy as np
import sklearn

from icenet.tools import aux

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from matplotlib import pyplot as plt


# PyTorch networks extend nn.Module
class LGR(nn.Module):

    def __init__(self, D, C, param):

        super(LGR,self).__init__()

        # Input dimension
        self.D = D

        # Output classes
        self.C = C

        # One layer
        self.L1    = nn.Linear(self.D, self.C)

    # Network forward operator
    def forward(self,x):
        
        x = self.L1(x)
        return x

    # Returns softmax probability
    def softpredict(self,x) :
        return F.softmax(self.forward(x), dim = 1)

    # Return class {0,1} 
    def binarypredict(self,x) :
        
        prob = list(self.softpredict(x).detach().numpy())
        return np.argmax(prob, axis=1)


# PyTorch networks extend nn.Module
class MAXOUT_MLP(nn.Module):

    def __init__(self, D, C, param):

        super(MAXOUT_MLP,self).__init__()

        # Input and output dimension
        self.D = D
        self.C = C
        self.param = param

        # Network modules
        self.fc1_list = nn.ModuleList()
        self.fc2_list = nn.ModuleList()

        self.num_units = param['num_units']
        self.dropout   = nn.AlphaDropout(p = param['dropout'])
        
        for _ in range(self.num_units):
            self.fc1_list.append(nn.Linear(self.D, param['neurons']))
            self.fc2_list.append(nn.Linear(param['neurons'], self.C))
    
    def forward(self,x):

        x = self.maxout(x, self.fc1_list)
        x = self.dropout(x)
        x = self.maxout(x, self.fc2_list)

        return x
    
    # Define maxout layer
    def maxout(self,x, layer_list):

        max_output = layer_list[0](x) # pass x to first unit in layer1
        for _, layer in enumerate(layer_list, start=1):
            max_output= torch.max(layer(x),max_output)
        return max_output

    # Returns softmax probability
    def softpredict(self,x) :
        return F.softmax(self.forward(x), dim = 1)

    # Return class {0,1} 
    def binarypredict(self,x) :
        
        prob = list(self.softpredict(x).detach().numpy())
        return np.argmax(prob, axis=1)


# http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
#
#
def log_sum_exp(x):
    b, _ = torch.max(x, 1)
    # b.size() = [N, ], unsqueeze() required
    y = b + torch.log(torch.exp(x - b.unsqueeze(dim=1).expand_as(x)).sum(1))
    # y.size() = [N, ], no need to squeeze()
    return y


# Per instance weighted cross entropy loss
#
#
def multiclass_cross_entropy(y_hat, y, N_classes, weights) :

    y = F.one_hot(y, N_classes)
    loss = -y*torch.log(y_hat) * weights
    loss = loss.sum() / y.shape[0]

    return loss

# Per instance weighted 'focal entropy loss'
# https://arxiv.org/pdf/1708.02002.pdf
# 
def multiclass_focal_entropy(y_hat, y, N_classes, weights, gamma) :

    y = F.one_hot(y, N_classes)
    loss = -y * torch.pow(1 - y_hat, gamma) * torch.log(y_hat) * weights
    loss = loss.sum() / y.shape[0]
    
    return loss


def train(model, X_trn, Y_trn, X_val, Y_val, trn_weights, param) :

    print(__name__ + '.train: Training samples = {}, Validation samples = {}'.format(X_trn.shape[0], X_val.shape[0]))


    ### Initialize the model
    dim       = X_trn.shape[1]
    N_classes = len(np.unique(Y_trn)) # Count the number of classes
    print(__name__ + ".train: Found {} classes in training sample".format(N_classes))
    
    # Prints the weights and biases
    print(model)
    
    # Class fractions
    YY = Y_trn.numpy()
    frac = []
    for i in range(N_classes):
        frac.append( sum(YY == i) / YY.size )

    print(__name__ + ".train: Class fractions in training sample: ")
    print(frac)
    print('\n')
    
    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = param['learning_rate'], amsgrad = True)
    #optimizer = torch.optim.SGD(model.parameters(), lr = param['learning_rate'])
    

    # List to store losses
    losses   = []
    trn_aucs = []
    val_aucs = []

    print(__name__ + '.train: Training loop ...')

    # Change the shape
    one_hot_weights = np.zeros((len(trn_weights), N_classes))
    for i in range(N_classes):
        one_hot_weights[Y_trn == i, i] = trn_weights[Y_trn == i]
    one_hot_weights = torch.tensor(one_hot_weights, dtype=torch.float32)

    # Epoch loop
    for epoch in tqdm(range(param['epochs']), ncols = 30):

        # X is a torch Variable
        permutation = torch.randperm(X_trn.size()[0])

        # Minibatch loop
        sumloss = 0
        nbatch  = 0
        for i in range(0, X_trn.size()[0], param['batch_size']):

            indices = permutation[i:i + param['batch_size']]
            batch_x, batch_y = X_trn[indices], Y_trn[indices]

            # Predict
            phat = model.softpredict(batch_x)

            # Evaluate loss
            loss = 0
            if param['lossfunc'] == 'cross_entropy':
                loss = multiclass_cross_entropy(phat, batch_y, N_classes, one_hot_weights[indices])

            elif param['lossfunc'] == 'focal_entropy':
                loss = multiclass_focal_entropy(phat, batch_y, N_classes, one_hot_weights[indices], param['gamma'])

            elif param['lossfunc'] == 'inverse_focal':
                loss = multiclass_inverse_focal(phat, batch_y, N_classes, one_hot_weights[indices], param['gamma'])

            else:
                print(__name__ + '.train: Error with unknown lossfunc ')


            # Zero gradients
            optimizer.zero_grad()

            # Compute gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            ### Save metrics
            sumloss += loss.item()
            nbatch  += 1

        avgloss = sumloss/nbatch
        losses.append(avgloss)

        trn_metric = aux.Metric(y_true = Y_trn.detach().numpy(),
            y_soft = model.softpredict(X_trn).detach().numpy()[:,1], valrange = [0,1])

        val_metric = aux.Metric(y_true = Y_val.detach().numpy(),
            y_soft = model.softpredict(X_val).detach().numpy()[:,1], valrange = [0,1])

        trn_aucs.append(trn_metric.auc)
        val_aucs.append(val_metric.auc)
        
        if (epoch % 3 == 0) :
            print('Epoch = {} : train loss = {:.3f} [trn AUC = {:.3f}, val AUC = {:.3f}]'.
                format(epoch, avgloss, trn_aucs[-1], val_aucs[-1]))

    return model, losses, trn_aucs, val_aucs
