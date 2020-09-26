# Graph Neural Nets
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d

from   torch_geometric.nn import Set2Set, global_mean_pool, global_max_pool, global_sort_pool
from   torch_geometric.nn import NNConv, GINEConv, GATConv, SplineConv, GCNConv, SGConv, SAGEConv, EdgeConv, DynamicEdgeConv
from   torch_geometric.nn import MessagePassing

from   icenet.deep import dopt
from   icenet.tools import aux


def MLP(channels, batch_norm=True):
    """
    Return a Multi Layer Perceptron with an arbitrary number of layers.

    Args:
        channels   : input structure, such as [128, 64, 64] for a 3-layer network.
        batch_norm : batch normalization
    Returns:
        nn.sequential object
    """
    if batch_norm:
        return nn.Sequential(*[
            nn.Sequential(
                nn.Linear(channels[i - 1], channels[i]),
                nn.ReLU(),
                nn.BatchNorm1d(channels[i])
            )
            for i in range(1,len(channels))
        ])
    else:
        return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU())
        for i in range(1,len(channels))
    ])


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
    def __init__(self, D, C, G=0, dropout=0.25, task='node'):
        super(GATNet, self).__init__()

        self.D = D
        self.C = C
        self.G = G

        self.dropout = dropout
        self.task = task

        self.conv1 = GATConv(self.D, self.D, heads=2, dropout=dropout)
        self.conv2 = GATConv(self.D * 2, self.D, heads=1, concat=False, dropout=dropout)
        
        if (self.G > 0):
            self.Z = self.D + self.G
        else:
            self.Z = self.D
        self.mlp1 = Linear(self.Z, self.Z)
        self.mlp2 = Linear(self.Z, self.C)

    def forward(self, data):
        
        if not hasattr(data,'batch'):
            # Create virtual null batch if singlet graph input
            setattr(data, 'batch', torch.tensor(np.zeros(data.x.shape[0]), dtype=torch.long))
        
        x = F.elu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, training=self.training)

        x = F.elu(self.conv2(x,      data.edge_index))
        x = F.dropout(x, training=self.training)

        # ** Global pooling (to handle graph level classification) **
        if self.task == 'graph':
            x = global_max_pool(x, data.batch)

        # Global features concatenated
        if self.G > 0:
            u = data.u.view(-1, self.G)
            x = torch.cat((x, u), 1)

        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))

        return x

    # Returns softmax probability
    def softpredict(self,x):
        return F.softmax(self.forward(x), dim=1)


# DynamicEdgeConv based graph net
#
# https://arxiv.org/abs/1801.07829
#
class DECNet(torch.nn.Module):
    def __init__(self, D, C, G=0, k=4, task='node', aggr='max'):
        super(DECNet, self).__init__()
        
        self.D = D
        self.C = C
        self.G = G

        self.task  = task
        
        # Convolution layers
        self.conv1 = DynamicEdgeConv(MLP([2 * self.D, 32, 32]), k=k, aggr=aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 32, 64]), k=k, aggr=aggr)
        
        # "Fusion" layer taking in conv1 and conv2 outputs
        self.lin1  = MLP([32 + 64, 96])
        
        if (self.G > 0):
            self.Z = 96 + self.G
        else:
            self.Z = 96

        # Final layers concatenating everything
        self.mlp1  = MLP([self.Z, self.Z, self.C])

    def forward(self, data):

        if not hasattr(data,'batch'):
            # Create virtual null batch if singlet graph input
            setattr(data, 'batch', torch.tensor(np.zeros(data.x.shape[0]), dtype=torch.long))

        x1 = self.conv1(data.x, data.batch)
        x2 = self.conv2(x1,     data.batch)

        x = self.lin1(torch.cat([x1, x2], dim=1))

        # ** Global pooling (to handle graph level classification) **
        if self.task == 'graph':
            x = global_max_pool(x, data.batch)

        # Global features concatenated
        if self.G > 0:
            u = data.u.view(-1, self.G)
            x = torch.cat((x, u), 1)

        # Final layers
        x = self.mlp1(x)

        return x

    # Returns softmax probability
    def softpredict(self,x) :
        return F.softmax(self.forward(x), dim=1)


# NNConv based graph net
#
# https://arxiv.org/abs/1704.01212
#
class NNNet(torch.nn.Module):
    def __init__(self, D, C, G=0, E=1, Q=96, task='node', aggr='add', pooltype='s2s'):
        super(NNNet, self).__init__()

        self.D = D  # node feature dimension
        self.E = E  # edge feature dimension
        self.G = G  # global feature dimension
        self.C = C  # number output classes
        
        self.Q = Q  # latent dimension

        self.task     = task
        self.pooltype = pooltype

        # Convolution layers
        # nn with size [-1, num_edge_features] x [-1, in_channels * out_channels]
        self.conv1 = NNConv(in_channels=D, out_channels=D, nn=MLP([E, D*D]), aggr=aggr)
        self.conv2 = NNConv(in_channels=D, out_channels=D, nn=MLP([E, D*D]), aggr=aggr)
        
        # "Fusion" layer taking in conv layer outputs
        self.lin1  = MLP([D+D, Q])

        # Set2Set pooling operation produces always output with 2 x input dimension
        # => use linear layer to project down
        if pooltype == 's2s':
            self.S2Spool = Set2Set(in_channels=Q, processing_steps=3, num_layers=1)
            self.S2Slin  = Linear(2*Q, Q)

        if (self.G > 0):
            self.Z = Q + self.G
        else:
            self.Z = Q
        
        # Final layers concatenating everything
        self.mlp1  = MLP([self.Z, self.Z, self.C])


    def forward(self, data):

        if not hasattr(data,'batch'):
            # Create virtual null batch if singlet graph input
            setattr(data, 'batch', torch.tensor(np.zeros(data.x.shape[0]), dtype=torch.long))

        x1 = self.conv1(data.x, data.edge_index, data.edge_attr)
        x2 = self.conv2(x1,     data.edge_index, data.edge_attr)
        x  = self.lin1(torch.cat([x1, x2], dim=1))

        # ** Global pooling **
        if self.task == 'graph':
            if self.pooltype == 's2s':
                x = self.S2Spool(x, data.batch)
                x = self.S2Slin(x)
            elif self.pooltype == 'max':
                x = global_max_pool(x, data.batch)

        # Global features concatenated
        if self.G > 0:
            u = data.u.view(-1, self.G)
            x = torch.cat([x, u], 1)

        # Final layers
        x = self.mlp1(x)

        return x

    # Returns softmax probability
    def softpredict(self,x) :
        return F.softmax(self.forward(x), dim=1)


# SplineConv based graph net
#
# https://arxiv.org/abs/1711.08920
#
class SplineNet(torch.nn.Module):
    def __init__(self, D, C, E, G=0, task='node'):
        super(SplineNet, self).__init__()

        self.D     = D
        self.C     = C
        self.E     = E
        self.G     = G
        self.task  = task

        self.conv1 = SplineConv(self.D, self.D, dim=E, degree=1, kernel_size=3)
        self.conv2 = SplineConv(self.D, self.D, dim=E, degree=1, kernel_size=5)
        
        if (self.G > 0):
            self.Z = self.D + self.G
        else:
            self.Z = self.D
        self.mlp1 = Linear(self.Z, self.Z)
        self.mlp2 = Linear(self.Z, self.C)

    def forward(self, data):

        if not hasattr(data,'batch'):
            # Create virtual null batch if singlet graph input
            setattr(data, 'batch', torch.tensor(np.zeros(data.x.shape[0]), dtype=torch.long))
        
        # SplineConv supports only 1-dimensional edge attributes
        x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = F.dropout(x, training=self.training)

        x = F.elu(self.conv2(x,      data.edge_index, data.edge_attr))
        x = F.dropout(x, training=self.training)
        
        # ** Global pooling **
        if self.task == 'graph':
            x = global_max_pool(x, data.batch)

        # Global features concatenated
        if self.G > 0:
            u = data.u.view(-1, self.G)
            x = torch.cat((x, u), 1)

        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))

        return x

    # Returns softmax probability
    def softpredict(self,x) :
        return F.softmax(self.forward(x), dim=1)


# SAGEConv based graph net
#
# https://arxiv.org/abs/1706.02216
# 
class SAGENet(torch.nn.Module):
    def __init__(self, D, C, G=0, task='node'):
        super(SAGENet, self).__init__()

        self.D     = D
        self.C     = C
        self.G     = G

        self.conv1 = SAGEConv(self.D, self.D)
        self.conv2 = SAGEConv(self.D, self.D)
        
        if (self.G > 0):
            self.Z = self.D + self.G
        else:
            self.Z = self.D
        self.mlp1 = Linear(self.Z, self.Z)
        self.mlp2 = Linear(self.Z, self.C)

        self.task  = task
        
    def forward(self, data):

        if not hasattr(data,'batch'):
            # Create virtual null batch if singlet graph input
            setattr(data, 'batch', torch.tensor(np.zeros(data.x.shape[0]), dtype=torch.long))

        x = F.elu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, training=self.training)

        x = F.elu(self.conv2(x,      data.edge_index))
        x = F.dropout(x, training=self.training)

        # ** Global pooling **
        if self.task == 'graph':
            x = global_max_pool(x, data.batch)

        # Global features concatenated
        if self.G > 0:
            u = data.u.view(-1, self.G)
            x = torch.cat((x, u), 1)

        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))

        return x
    
    # Returns softmax probability
    def softpredict(self,x) :
        return F.softmax(self.forward(x), dim=1)


# SGConv based graph net
#
# https://arxiv.org/abs/1902.07153
# 
class SGNet(torch.nn.Module):
    def __init__(self, D, C, G=0, K=2, task='node'):
        super(SGNet, self).__init__()

        self.D     = D
        self.C     = C
        self.G     = G
        self.K     = K

        self.conv1 = SGConv(self.D, self.D, self.K, cached=False)
        self.conv2 = SGConv(self.D, self.D, self.K, cached=False)
        
        if (self.G > 0):
            self.Z = self.D + self.G
        else:
            self.Z = self.D
        self.mlp1 = Linear(self.Z, self.Z)
        self.mlp2 = Linear(self.Z, self.C)

        self.task  = task
        
    def forward(self, data):

        if not hasattr(data,'batch'):
            # Create virtual null batch if singlet graph input
            setattr(data, 'batch', torch.tensor(np.zeros(data.x.shape[0]), dtype=torch.long))

        x = F.elu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, training=self.training)

        x = F.elu(self.conv2(x,      data.edge_index))
        x = F.dropout(x, training=self.training)

        # ** Global pooling **
        if self.task == 'graph':
            x = global_max_pool(x, data.batch)

        # Global features concatenated
        if self.G > 0:
            u = data.u.view(-1, self.G)
            x = torch.cat((x, u), 1)

        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))

        return x
    
    # Returns softmax probability
    def softpredict(self,x):
        return F.softmax(self.forward(x), dim=1)


# GINEConv based graph net
#
# https://arxiv.org/abs/1810.00826
# https://arxiv.org/abs/1905.12265
#
class GINENet(torch.nn.Module):
    def __init__(self, D, C, G=0, task='node'):
        super(GINENet, self).__init__()

        self.D = D
        self.C = C
        self.G = G

        self.task  = task

        # Convolution layers
        self.conv1 = GINEConv(MLP([self.D, self.D]))
        self.conv2 = GINEConv(MLP([self.D, 64]))
        
        # "Fusion" layer taking in conv1 and conv2 outputs
        self.lin1  = MLP([self.D + 64, 96])

        if (self.G > 0):
            self.Z = 96 + self.G
        else:
            self.Z = 96

        # Final layers concatenating everything
        self.mlp1  = MLP([self.Z, self.Z, self.C])

    def forward(self, data):

        if not hasattr(data,'batch'):
            # Create virtual null batch if singlet graph input
            setattr(data, 'batch', torch.tensor(np.zeros(data.x.shape[0]), dtype=torch.long))

        # ----------------------------------------------------------------
        # GINEConv requires node features and edge features with the same dimension.
        # Increase dimensionality below
        D_n = data.x[0].size(-1)
        D_e = data.edge_attr.size(-1)
        if D_n > D_e:
            lin = Linear(1, D_n)
            x   = data.x
            edge_attr = lin(data.edge_attr)
        elif D_e > D_n:
            lin = Linear(1, D_e)
            x   = lin(data.x)
            edge_attr = data.edge_attr
        # ----------------------------------------------------------------

        x1 = self.conv1(x,  data.edge_index, edge_attr)
        x2 = self.conv2(x1, data.edge_index, edge_attr)

        x  = self.lin1(torch.cat([x1, x2], dim=1))

        # ** Global pooling **
        if self.task == 'graph':
            x = global_max_pool(x, data.batch)

        # Global features concatenated
        if self.G > 0:
            u = data.u.view(-1, self.G)
            x = torch.cat((x, u), 1)

        # Final layers
        x = self.mlp1(x)
        return x

    # Returns softmax probability
    def softpredict(self,x):
        return F.softmax(self.forward(x), dim=1)
