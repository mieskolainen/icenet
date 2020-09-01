# Graph Neural Nets
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from   icenet.deep import dopt
from   icenet.tools import aux

from   torch_geometric.nn import max_pool, global_mean_pool, global_max_pool, global_sort_pool
from   torch_geometric.nn import GATConv, SplineConv, GCNConv, SGConv, EdgeConv, DynamicEdgeConv

from   torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d

from   torch_geometric.nn import MessagePassing
from   torch_geometric.nn import GATConv
from   torch_geometric.nn import SplineConv


def train(model, loader, optimizer, device):
    """
    Pytorch geometric based training routine.
    
    Args:
        model     : pytorch geometric model
        loader    : pytorch geometric dataloader
        optimizer : pytorch optimizer
        device    : 'cpu' or 'device'
    
    Returns
        trained model (return implicit via input arguments)
    """
    model.train()
    
    total_loss = 0
    n = 0
    for batch in loader:

        # Change the shape
        w = torch.tensor(aux.weight2onehot(weights=batch.w, Y=batch.y, N_classes=model.C)).to(device)

        # Predict probabilities
        batch = batch.to(device)
        optimizer.zero_grad()
        phat = model.softpredict(batch)

        # Evaluate loss
        loss = dopt.multiclass_cross_entropy(phat=phat, y=batch.y, N_classes=model.C, weights=w)
        loss.backward()

        total_loss += loss.item() * batch.num_graphs
        optimizer.step()
        n += batch.num_graphs

    return total_loss / n


def test(model, loader, optimizer, device):
    """
    Pytorch geometric based testing routine.
    
    Args:
        model     : pytorch geometric model
        loader    : pytorch geometric dataloader
        optimizer : pytorch optimizer
        device    : 'cpu' or 'device'
    
    Returns
        accuracy, auc
    """

    model.eval()

    correct = 0
    aucsum  = 0
    k = 0
    signal_class = 1

    for data in loader:
        data = data.to(device)
        with torch.no_grad():

            phat = model.softpredict(data)
            pred = phat.max(dim=1)[1]

            y_true  = data.y.to('cpu').numpy()
            y_soft  = phat[:, signal_class].to('cpu').numpy()

            metrics = aux.Metric(y_true = y_true, y_soft = y_soft)
            aucsum += metrics.auc

        correct += pred.eq(data.y).sum().item()
        k += 1

    return correct / len(loader.dataset), aucsum / k


# GATConv based graph net
#
# https://arxiv.org/abs/1710.10903
#
class GATNet(torch.nn.Module):
    def __init__(self, D, C, dropout=0.25, task='node'):
        super(GATNet, self).__init__()

        self.D = D
        self.C = C
        self.dropout = dropout
        self.task = task

        self.conv1 = GATConv(self.D, self.D, heads=8, dropout=dropout)
        self.conv2 = GATConv(self.D * 8, self.D, heads=1, concat=False, dropout=dropout)
        
        self.mlp1  = Linear(self.D, self.C)
        self.task  = task

    def forward(self, data):

        x = F.elu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, training=self.training)

        x = F.elu(self.conv2(x,      data.edge_index))
        x = F.dropout(x, training=self.training)

        x = F.relu(self.mlp1(x))

        # ** Mean pooling (to handle graph level classification) **
        if self.task == 'graph' and hasattr(data,'batch'):
            x = global_mean_pool(x, data.batch)

        return x

    # Returns softmax probability
    def softpredict(self,x) :
        if hasattr(x, 'batch'):
            return F.softmax(self.forward(x), dim=1)
        else:
            return F.softmax(self.forward(x), dim=1)[0].unsqueeze(0)

# SplineConv based graph net
#
# https://arxiv.org/abs/1711.08920
#
class SplineNet(torch.nn.Module):
    def __init__(self, D, C, task='node'):
        super(SplineNet, self).__init__()

        self.D     = D
        self.C     = C

        self.conv1 = SplineConv(self.D, self.D, dim=1, degree=1, kernel_size=3)
        self.conv2 = SplineConv(self.D, self.D, dim=1, degree=1, kernel_size=5)
        
        self.mlp1  = Linear(self.D, self.C)
        self.task  = task

    def forward(self, data):

        x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = F.dropout(x, training=self.training)

        x = F.elu(self.conv2(x,      data.edge_index, data.edge_attr))
        x = F.dropout(x, training=self.training)

        x = F.relu(self.mlp1(x))

        # ** Mean pooling (to handle graph level classification) **
        if self.task == 'graph' and hasattr(data,'batch'):
            x = global_max_pool(x, data.batch)

        return x

    # Returns softmax probability
    def softpredict(self,x) :
        if hasattr(x, 'batch'):
            return F.softmax(self.forward(x), dim=1)
        else:
            return F.softmax(self.forward(x), dim=1)[0].unsqueeze(0)

# SGConv based graph net
#
# https://arxiv.org/abs/1902.07153
# 
class SGNet(torch.nn.Module):
    def __init__(self, D, C, K=2, task='node'):
        super(SGNet, self).__init__()

        self.D     = D
        self.C     = C
        self.K     = K
        self.conv1 = SGConv(self.D, self.D, self.K, cached=False)
        self.conv2 = SGConv(self.D, self.D, self.K, cached=False)
        
        self.mlp1  = Linear(self.D, self.C)
        self.task  = task
        
    def forward(self, data):
        x = F.elu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, training=self.training)

        x = F.elu(self.conv2(x,      data.edge_index))
        x = F.dropout(x, training=self.training)

        x = F.relu(self.mlp1(x))

        # ** Mean pooling (to handle graph level classification) **
        if self.task == 'graph' and hasattr(data,'batch'):
            x = global_max_pool(x, data.batch)

        return x

    # Returns softmax probability
    def softpredict(self,x) :
        if hasattr(x, 'batch'):
            return F.softmax(self.forward(x), dim=1)
        else:
            return F.softmax(self.forward(x), dim=1)[0].unsqueeze(0)
