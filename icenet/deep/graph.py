# Graph Neural Nets
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d

from   typing import Union, Tuple, Callable
from   torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size

from   torch_geometric.nn import Set2Set, global_add_pool, global_mean_pool, global_max_pool, global_sort_pool
from   torch_geometric.nn import NNConv, GINEConv, GATConv, SplineConv, GCNConv, SGConv, SAGEConv, EdgeConv, DynamicEdgeConv
from   torch_geometric.nn import MessagePassing

from torch_scatter import scatter_add, scatter_max, scatter_mean


from icenet.deep.pgraph import *
from icenet.deep import dopt
from icenet.tools import aux
from icenet.tools import aux_torch

from icenet.deep import losstools


from typing import Callable, Union, Optional
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing


class SuperEdgeConv(MessagePassing):
    r"""
    Custom convolution operator.
    """

    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super(SuperEdgeConv, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        #self.reset_parameters()

    #def reset_parameters(self):
        #reset(self.nn)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
            edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:

        return self.nn(torch.cat([x_i, x_j - x_i, edge_attr], dim=-1))

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


def MLP(channels, activation='relu', batch_norm=True):
    """
    Return a Multi Layer Perceptron with an arbitrary number of layers.
    
    Args:
        channels   : input structure, such as [128, 64, 64] for a 3-layer network.
        batch_norm : batch normalization
    Returns:
        nn.sequential object
    
    """
    if activation == 'relu':
        print(f'MLP: Using relu activation')
    else:
        print(f'MLP: Using tanh activation')

    if batch_norm:
        return nn.Sequential(*[
            nn.Sequential(
                nn.Linear(channels[i - 1], channels[i]),
                nn.ReLU() if activation == 'relu' else nn.Tanh(),
                nn.BatchNorm1d(channels[i])
            )
            for i in range(1,len(channels))
        ])
    else:
        return nn.Sequential(*[
            nn.Sequential(
                nn.Linear(channels[i - 1], channels[i]),
                nn.ReLU() if activation == 'relu' else nn.Tanh()
            )
            for i in range(1,len(channels))
        ])


def train(model, loader, optimizer, device, param):
    """
    Pytorch geometric based training routine.
    
    Args:
        model     : pytorch geometric model
        loader    : pytorch geometric dataloader
        optimizer : pytorch optimizer
        device    : 'cpu' or 'device'
        param:    : optimization parameters
    
    Returns
        trained model (return implicit via input arguments)
    """
    model.train()
    
    total_loss = 0
    n = 0
    for batch in loader:

        batch = batch.to(device)
        optimizer.zero_grad()

        # Compute loss
        batch_weights = aux_torch.weight2onehot(weights=batch.w, Y=batch.y, N_classes=model.C)
        loss = losstools.loss_wrapper(model=model, x=batch, y=batch.y, N_classes=model.C, weights=batch_weights, param=param)

        # Propagate gradients
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
            weights = data.w.to('cpu').numpy()
            
            metrics = aux.Metric(y_true=y_true, y_soft=y_soft, weights=weights)
            aucsum += metrics.auc

        correct += pred.eq(data.y).sum().item()
        k += 1

    return correct / len(loader.dataset), aucsum / k

# PANConv based graph net
# https://arxiv.org/abs/2006.16811
#
class PANNet(torch.nn.Module):
    def __init__(self, D, C, G=0, E=None, cdim=64, dropout=0.5, conv_aggr=None, global_pool='max', filter_size=5, task='node'):

        super(PANNet, self).__init__()

        self.D = D
        self.C = C
        self.G = G
        self.cdim = cdim

        self.dropout     = dropout
        self.task        = task
        self.global_pool = global_pool
        self.task = task

        # --------------------------------------------

        self.conv1 = PANConv(D, cdim, filter_size)
        self.pool1 = PANXUMPooling(cdim)
        # self.drop1 = PANDropout()

        self.conv2 = PANConv(cdim, cdim, filter_size)
        self.pool2 = PANXUMPooling(cdim)
        # self.drop2 = PANDropout()

        self.conv3 = PANConv(cdim, cdim, filter_size)
        self.pool3 = PANXUMPooling(cdim)
        
        # Set2Set pooling operation produces always output with 2 x input dimension
        # => use linear layer to project down
        if self.global_pool == 's2s':
            self.S2Spool = Set2Set(in_channels=self.cdim, processing_steps=3, num_layers=1)
            self.S2Slin  = Linear(2*self.cdim, self.cdim)
        
        # ------------------------------
        if (self.G > 0):
            self.Z = self.cdim + self.G
        else:
            self.Z = self.cdim

        self.mlp1  = MLP([self.Z, self.Z, self.C])


    def forward(self, data, conv_only=False):
        
        if not hasattr(data,'batch') or data.batch is None:
            # Create virtual null batch if singlet graph input
            setattr(data, 'batch', torch.tensor(np.zeros(data.x.shape[0]), dtype=torch.long))


        x, edge_index, batch = data.x, data.edge_index, data.batch
        perm_list = list()
        edge_mask_list = None

        x = self.conv1(x, edge_index)
        M = self.conv1.m
        x, edge_index, _, batch, perm, score_perm = self.pool1(x, edge_index, batch=batch, M=M)
        perm_list.append(perm)

        # AFTERDROP, edge_mask_list = self.drop1(edge_index, p=0.5)
        x = self.conv2(x, edge_index, edge_mask_list=edge_mask_list)
        M = self.conv2.m
        x, edge_index, _, batch, perm, score_perm = self.pool2(x, edge_index, batch=batch, M=M)
        perm_list.append(perm)

        # AFTERDROP, edge_mask_list = self.drop2(edge_index, p=0.5)
        x = self.conv3(x, edge_index, edge_mask_list=edge_mask_list)
        M = self.conv3.m
        x, edge_index, _, batch, perm, score_perm = self.pool3(x, edge_index, batch=batch, M=M)
        perm_list.append(perm)

        # ** Global pooling (to handle graph level classification) **
        if self.task == 'graph':
            if self.global_pool == 's2s':
                x = self.S2Spool(x, batch)
                x = self.S2Slin(x)
            elif self.global_pool == 'max':
                x = global_max_pool(x, batch)
            elif self.global_pool == 'add':
                x = global_add_pool(x, batch)
            elif self.global_pool == 'mean':
                x = global_mean_pool(x, batch)
            else:
                raise Exception(__name__ + f': Unknown global_pool <{self.global_pool}>')

        if conv_only: # Return convolution part
            return x
        
        # Global features concatenated
        if self.G > 0:
            u = data.u.view(-1, self.G)
            x = torch.cat((x, u), 1)

        # Final layers
        x = self.mlp1(x)
        return x
    
    # Returns softmax probability
    def softpredict(self,x):
        if self.training:
            return F.log_softmax(self.forward(x), dim=-1) # Numerically more stable
        else:
            return F.softmax(self.forward(x), dim=-1)


# GATConv based graph net
#
# https://arxiv.org/abs/1710.10903
#
class GATNet(torch.nn.Module):
    def __init__(self, D, C, G=0, E=None, cdim=96, conv_aggr=None, global_pool='max', dropout=0.0, task='node'):
        super(GATNet, self).__init__()

        self.D = D
        self.C = C
        self.G = G
        self.cdim = cdim

        self.dropout     = dropout
        self.task        = task
        self.global_pool = global_pool

        self.conv1 = GATConv(self.D, self.D, heads=2, dropout=dropout)
        self.conv2 = GATConv(self.D * 2, self.D, heads=1, concat=False, dropout=dropout)

        # "Fusion" layer taking in conv1 and conv2 outputs
        self.lin1  = MLP([self.D*2 + self.D, self.cdim])
        
        # Set2Set pooling operation produces always output with 2 x input dimension
        # => use linear layer to project down
        if self.global_pool == 's2s':
            self.S2Spool = Set2Set(in_channels=self.cdim, processing_steps=3, num_layers=1)
            self.S2Slin  = Linear(2*self.cdim, self.cdim)

        if (self.G > 0):
            self.Z = self.cdim + self.G
        else:
            self.Z = self.cdim

        # Final layers concatenating everything
        self.mlp1  = MLP([self.Z, self.Z, self.C])

    def forward(self, data, conv_only=False):
        
        if not hasattr(data,'batch') or data.batch is None:
            # Create virtual null batch if singlet graph input
            setattr(data, 'batch', torch.tensor(np.zeros(data.x.shape[0]), dtype=torch.long))

        x1 = self.conv1(data.x, data.edge_index)
        x2 = self.conv2(x1,     data.edge_index)

        x = self.lin1(torch.cat([x1, x2], dim=1))

        # ** Global pooling (to handle graph level classification) **
        if self.task == 'graph':
            if self.global_pool == 's2s':
                x = self.S2Spool(x, data.batch)
                x = self.S2Slin(x)
            elif self.global_pool == 'max':
                x = global_max_pool(x, data.batch)
            elif self.global_pool == 'add':
                x = global_add_pool(x, data.batch)
            elif self.global_pool == 'mean':
                x = global_mean_pool(x, data.batch)
            else:
                raise Exception(__name__ + f': Unknown global_pool <{self.global_pool}>')

        if conv_only: # Return convolution part
            return x

        # Global features concatenated
        if self.G > 0:
            u = data.u.view(-1, self.G)
            x = torch.cat((x, u), 1)

        # Final layers
        x = self.mlp1(x)

        return x

    # Returns softmax probability
    def softpredict(self,x):
        if self.training:
            return F.log_softmax(self.forward(x), dim=-1) # Numerically more stable
        else:
            return F.softmax(self.forward(x), dim=-1)



# SuperEdgeConv based graph net
#
# https://arxiv.org/abs/xyz
#
class SUPNet(torch.nn.Module):
    def __init__(self, D, C, G=0, k=1, E=None, cdim=96, task='node', conv_aggr='max', global_pool='max'):
        super(SUPNet, self).__init__()
        
        self.D = D
        self.C = C
        self.G = G
        self.cdim = cdim

        self.task        = task
        self.global_pool = global_pool

        # Convolution layers
        self.conv1 = SuperEdgeConv(MLP([2 * self.D + E, 32, 32]), aggr=conv_aggr)
        self.conv2 = SuperEdgeConv(MLP([2 * 32 + E, 64]), aggr=conv_aggr)
        
        # "Fusion" layer taking in conv1 and conv2 outputs
        self.lin1  = MLP([32 + 64, self.cdim])
        
        # Set2Set pooling operation produces always output with 2 x input dimension
        # => use linear layer to project down
        if self.global_pool == 's2s':
            self.S2Spool = Set2Set(in_channels=self.cdim, processing_steps=3, num_layers=1)
            self.S2Slin  = Linear(2*self.cdim, self.cdim)

        if (self.G > 0):
            self.Z = self.cdim + self.G
        else:
            self.Z = self.cdim

        # Final layers concatenating everything
        self.mlp1  = MLP([self.Z, self.Z, self.C])

    def forward(self, data, conv_only=False):

        if not hasattr(data,'batch') or data.batch is None:
            # Create virtual null batch if singlet graph input
            setattr(data, 'batch', torch.tensor(np.zeros(data.x.shape[0]), dtype=torch.long))
        
        x1 = self.conv1(data.x, data.edge_index, data.edge_attr)
        x2 = self.conv2(x1,     data.edge_index, data.edge_attr)
        
        x  = self.lin1(torch.cat([x1, x2], dim=1))

        # ** Global pooling (to handle graph level classification) **
        if self.task == 'graph':
            if self.global_pool == 's2s':
                x = self.S2Spool(x, data.batch)
                x = self.S2Slin(x)
            elif self.global_pool == 'max':
                x = global_max_pool(x, data.batch)
            elif self.global_pool == 'add':
                x = global_add_pool(x, data.batch)
            elif self.global_pool == 'mean':
                x = global_mean_pool(x, data.batch)
            else:
                raise Exception(__name__ + f': Unknown global_pool <{self.global_pool}>')
        
        if conv_only: # Return convolution part
            return x

        # Global features concatenated
        if self.G > 0:
            u = data.u.view(-1, self.G)
            x = torch.cat((x, u), 1)

        # Final layers
        x = self.mlp1(x)

        return x

    # Returns softmax probability
    def softpredict(self,x) :
        if self.training:
            return F.log_softmax(self.forward(x), dim=-1) # Numerically more stable
        else:
            return F.softmax(self.forward(x), dim=-1)


# Pure EdgeConv based graph net
# 
# https://arxiv.org/abs/1801.07829
#
class ECNet(torch.nn.Module):
    def __init__(self, D, C, G=0, k=1, E=None, cdim=96, task='node', conv_aggr='max', global_pool='max'):
        super(ECNet, self).__init__()
        
        self.D = D
        self.C = C
        self.G = G
        self.cdim = cdim

        self.task  = task
        self.global_pool = global_pool
        
        # Convolution layers
        self.conv1 = EdgeConv(MLP([2 * self.D, 32, 32]), aggr=conv_aggr)
        self.conv2 = EdgeConv(MLP([2 * 32, 64]), aggr=conv_aggr)
        
        # "Fusion" layer taking in conv1 and conv2 outputs
        self.lin1  = MLP([32 + 64, self.cdim])
        
        # Set2Set pooling operation produces always output with 2 x input dimension
        # => use linear layer to project down
        if self.global_pool == 's2s':
            self.S2Spool = Set2Set(in_channels=self.cdim, processing_steps=3, num_layers=1)
            self.S2Slin  = Linear(2*self.cdim, self.cdim)

        if (self.G > 0):
            self.Z = self.cdim + self.G
        else:
            self.Z = self.cdim

        # Final layers concatenating everything
        self.mlp1  = MLP([self.Z, self.Z, self.C])

    def forward(self, data, conv_only=False):
        
        if not hasattr(data,'batch') or data.batch is None:
            # Create virtual null batch if singlet graph input
            setattr(data, 'batch', torch.tensor(np.zeros(data.x.shape[0]), dtype=torch.long))
        
        x1 = self.conv1(data.x, data.edge_index)
        x2 = self.conv2(x1,     data.edge_index)
        
        x  = self.lin1(torch.cat([x1, x2], dim=1))

        # ** Global pooling (to handle graph level classification) **
        if self.task == 'graph':
            if self.global_pool == 's2s':
                x = self.S2Spool(x, data.batch)
                x = self.S2Slin(x)
            elif self.global_pool == 'max':
                x = global_max_pool(x, data.batch)
            elif self.global_pool == 'add':
                x = global_add_pool(x, data.batch)
            elif self.global_pool == 'mean':
                x = global_mean_pool(x, data.batch)
            else:
                raise Exception(__name__ + f': Unknown global_pool <{self.global_pool}>')

        if conv_only: # Return convolution part
            return x

        # Global features concatenated
        if self.G > 0:
            u = data.u.view(-1, self.G)
            x = torch.cat((x, u), 1)

        # Final layers
        x = self.mlp1(x)

        return x

    # Returns softmax probability
    def softpredict(self,x) :
        if self.training:
            return F.log_softmax(self.forward(x), dim=-1) # Numerically more stable
        else:
            return F.softmax(self.forward(x), dim=-1)


# DynamicEdgeConv based graph net
#
# https://arxiv.org/abs/1801.07829
#
class DECNet(torch.nn.Module):
    def __init__(self, D, C, G=0, k=4, E=None, cdim=96, task='node', conv_aggr='max', global_pool='max'):
        super(DECNet, self).__init__()
        
        self.D = D
        self.C = C
        self.G = G
        self.cdim = cdim

        self.task  = task
        self.global_pool = global_pool
        
        # Convolution layers
        self.conv1 = DynamicEdgeConv(MLP([2 * self.D, 32, 32]), k=k, aggr=conv_aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 32, 64]), k=k, aggr=conv_aggr)
        
        # "Fusion" layer taking in conv1 and conv2 outputs
        self.lin1  = MLP([32 + 64, self.cdim])
        
        # Set2Set pooling operation produces always output with 2 x input dimension
        # => use linear layer to project down
        if self.global_pool == 's2s':
            self.S2Spool = Set2Set(in_channels=self.cdim, processing_steps=3, num_layers=1)
            self.S2Slin  = Linear(2*self.cdim, self.cdim)

        if (self.G > 0):
            self.Z = self.cdim + self.G
        else:
            self.Z = self.cdim

        # Final layers concatenating everything
        self.mlp1  = MLP([self.Z, self.Z, self.C])

    def forward(self, data, conv_only=False):

        if not hasattr(data,'batch') or data.batch is None:
            # Create virtual null batch if singlet graph input
            setattr(data, 'batch', torch.tensor(np.zeros(data.x.shape[0]), dtype=torch.long))

        x1 = self.conv1(data.x, data.batch)
        x2 = self.conv2(x1,     data.batch)

        x = self.lin1(torch.cat([x1, x2], dim=1))

        # ** Global pooling (to handle graph level classification) **
        if self.task == 'graph':
            if self.global_pool == 's2s':
                x = self.S2Spool(x, data.batch)
                x = self.S2Slin(x)
            elif self.global_pool == 'max':
                x = global_max_pool(x, data.batch)
            elif self.global_pool == 'add':
                x = global_add_pool(x, data.batch)
            elif self.global_pool == 'mean':
                x = global_mean_pool(x, data.batch)
            else:
                raise Exception(__name__ + f': Unknown global_pool <{self.global_pool}>')

        if conv_only: # Return convolution part
            return x

        # Global features concatenated
        if self.G > 0:
            u = data.u.view(-1, self.G)
            x = torch.cat((x, u), 1)

        # Final layers
        x = self.mlp1(x)

        return x

    # Returns softmax probability
    def softpredict(self,x) :
        if self.training:
            return F.log_softmax(self.forward(x), dim=-1) # Numerically more stable
        else:
            return F.softmax(self.forward(x), dim=-1)


# NNConv based graph net
#
# https://arxiv.org/abs/1704.01212
#
class NNNet(torch.nn.Module):
    def __init__(self, D, C, G=0, E=1, cdim=96, task='node', conv_aggr='add', global_pool='s2s'):
        super(NNNet, self).__init__()

        self.D = D  # node feature dimension
        self.E = E  # edge feature dimension
        self.G = G  # global feature dimension
        self.C = C  # number output classes
        
        self.cdim = cdim  # latent dimension

        self.task        = task
        self.global_pool = global_pool

        # Convolution layers
        # nn with size [-1, num_edge_features] x [-1, in_channels * out_channels]
        self.conv1 = NNConv(in_channels=D, out_channels=D, nn=MLP([E, D*D]), aggr=conv_aggr)
        self.conv2 = NNConv(in_channels=D, out_channels=D, nn=MLP([E, D*D]), aggr=conv_aggr)
        
        # "Fusion" layer taking in conv layer outputs
        self.lin1  = MLP([D+D, self.cdim])

        # Set2Set pooling operation produces always output with 2 x input dimension
        # => use linear layer to project down
        if self.global_pool == 's2s':
            self.S2Spool = Set2Set(in_channels=self.cdim, processing_steps=3, num_layers=1)
            self.S2Slin  = Linear(2*self.cdim, self.cdim)

        if (self.G > 0):
            self.Z = self.cdim + self.G
        else:
            self.Z = self.cdim
        
        # Final layers concatenating everything
        self.mlp1  = MLP([self.Z, self.Z, self.C])


    def forward(self, data, conv_only=False):

        if not hasattr(data,'batch') or data.batch is None:
            # Create virtual null batch if singlet graph input
            setattr(data, 'batch', torch.tensor(np.zeros(data.x.shape[0]), dtype=torch.long))

        x1 = self.conv1(data.x, data.edge_index, data.edge_attr)
        x2 = self.conv2(x1,     data.edge_index, data.edge_attr)
        x  = self.lin1(torch.cat([x1, x2], dim=1))

        # ** Global pooling **
        if self.task == 'graph':
            if self.global_pool == 's2s':
                x = self.S2Spool(x, data.batch)
                x = self.S2Slin(x)
            elif self.global_pool == 'max':
                x = global_max_pool(x, data.batch)
            elif self.global_pool == 'add':
                x = global_add_pool(x, data.batch)
            elif self.global_pool == 'mean':
                x = global_mean_pool(x, data.batch)
            else:
                raise Exception(__name__ + f': Unknown global_pool <{self.global_pool}>')

        if conv_only: # Return convolution part
            return x

        # Global features concatenated
        if self.G > 0:
            u = data.u.view(-1, self.G)
            x = torch.cat([x, u], 1)

        # Final layers
        x = self.mlp1(x)

        return x
    
    # Returns softmax probability
    def softpredict(self,x) :
        if self.training:
            return F.log_softmax(self.forward(x), dim=-1) # Numerically more stable
        else:
            return F.softmax(self.forward(x), dim=-1)


# SplineConv based graph net
#
# https://arxiv.org/abs/1711.08920
#
class SplineNet(torch.nn.Module):
    def __init__(self, D, C, G=0,  E=None, conv_aggr=None, global_pool='max', task='node'):
        super(SplineNet, self).__init__()

        self.D     = D
        self.C     = C
        self.E     = E
        self.G     = G
        self.task  = task
        self.global_pool = global_pool

        self.conv1 = SplineConv(self.D, self.D, dim=E, degree=1, kernel_size=3)
        self.conv2 = SplineConv(self.D, self.D, dim=E, degree=1, kernel_size=5)
        
        # Set2Set pooling operation produces always output with 2 x input dimension
        # => use linear layer to project down
        if self.global_pool == 's2s':
            self.S2Spool = Set2Set(in_channels=self.D, processing_steps=3, num_layers=1)
            self.S2Slin  = Linear(2*self.D, self.D)

        if (self.G > 0):
            self.Z = self.D + self.G
        else:
            self.Z = self.D
        self.mlp1 = Linear(self.Z, self.Z)
        self.mlp2 = Linear(self.Z, self.C)

    def forward(self, data, conv_only=False):

        if not hasattr(data,'batch') or data.batch is None:
            # Create virtual null batch if singlet graph input
            setattr(data, 'batch', torch.tensor(np.zeros(data.x.shape[0]), dtype=torch.long))
        
        # SplineConv supports only 1-dimensional edge attributes
        x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        x = F.dropout(x, training=self.training)

        x = F.elu(self.conv2(x,      data.edge_index, data.edge_attr))
        x = F.dropout(x, training=self.training)
        
        # ** Global pooling **
        if self.task == 'graph':
            if self.global_pool == 's2s':
                x = self.S2Spool(x, data.batch)
                x = self.S2Slin(x)
            elif self.global_pool == 'max':
                x = global_max_pool(x, data.batch)
            elif self.global_pool == 'add':
                x = global_add_pool(x, data.batch)
            elif self.global_pool == 'mean':
                x = global_mean_pool(x, data.batch)
            else:
                raise Exception(__name__ + f': Unknown global_pool <{self.global_pool}>')

        if conv_only: # Return convolution part
            return x

        # Global features concatenated
        if self.G > 0:
            u = data.u.view(-1, self.G)
            x = torch.cat((x, u), 1)

        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))

        return x

    # Returns softmax probability
    def softpredict(self,x) :
        if self.training:
            return F.log_softmax(self.forward(x), dim=-1) # Numerically more stable
        else:
            return F.softmax(self.forward(x), dim=-1)


# SAGEConv based graph net
#
# https://arxiv.org/abs/1706.02216
# 
class SAGENet(torch.nn.Module):
    def __init__(self, D, C, G=0, E=None, conv_aggr=None, global_pool='max', task='node'):
        super(SAGENet, self).__init__()

        self.D     = D
        self.C     = C
        self.G     = G

        self.conv1 = SAGEConv(self.D, self.D)
        self.conv2 = SAGEConv(self.D, self.D)
        
        # Set2Set pooling operation produces always output with 2 x input dimension
        # => use linear layer to project down
        if self.global_pool == 's2s':
            self.S2Spool = Set2Set(in_channels=self.D, processing_steps=3, num_layers=1)
            self.S2Slin  = Linear(2*self.D, self.D)

        if (self.G > 0):
            self.Z = self.D + self.G
        else:
            self.Z = self.D
        self.mlp1 = Linear(self.Z, self.Z)
        self.mlp2 = Linear(self.Z, self.C)

        self.task  = task
        self.global_pool = global_pool
        
    def forward(self, data, conv_only=False):

        if not hasattr(data,'batch') or data.batch is None:
            # Create virtual null batch if singlet graph input
            setattr(data, 'batch', torch.tensor(np.zeros(data.x.shape[0]), dtype=torch.long))

        x = F.elu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, training=self.training)

        x = F.elu(self.conv2(x,      data.edge_index))
        x = F.dropout(x, training=self.training)

        # ** Global pooling **
        if self.task == 'graph':
            if self.global_pool == 's2s':
                x = self.S2Spool(x, data.batch)
                x = self.S2Slin(x)
            elif self.global_pool == 'max':
                x = global_max_pool(x, data.batch)
            elif self.global_pool == 'add':
                x = global_add_pool(x, data.batch)
            elif self.global_pool == 'mean':
                x = global_mean_pool(x, data.batch)
            else:
                raise Exception(__name__ + f': Unknown global_pool <{self.global_pool}>')

        if conv_only: # Return convolution part
            return x

        # Global features concatenated
        if self.G > 0:
            u = data.u.view(-1, self.G)
            x = torch.cat((x, u), 1)

        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))

        return x
    
    # Returns softmax probability
    def softpredict(self,x) :
        if self.training:
            return F.log_softmax(self.forward(x), dim=-1) # Numerically more stable
        else:
            return F.softmax(self.forward(x), dim=-1)


# SGConv based graph net
#
# https://arxiv.org/abs/1902.07153
# 
class SGNet(torch.nn.Module):
    def __init__(self, D, C, G=0, K=2, E=None, conv_aggr=None, global_pool='max', task='node'):
        super(SGNet, self).__init__()

        self.D     = D
        self.C     = C
        self.G     = G
        self.K     = K

        self.conv1 = SGConv(self.D, self.D, self.K, cached=False)
        self.conv2 = SGConv(self.D, self.D, self.K, cached=False)
        
        # Set2Set pooling operation produces always output with 2 x input dimension
        # => use linear layer to project down
        if self.global_pool == 's2s':
            self.S2Spool = Set2Set(in_channels=self.D, processing_steps=3, num_layers=1)
            self.S2Slin  = Linear(2*self.D, self.D)

        if (self.G > 0):
            self.Z = self.D + self.G
        else:
            self.Z = self.D
        self.mlp1 = Linear(self.Z, self.Z)
        self.mlp2 = Linear(self.Z, self.C)

        self.task  = task
        self.global_pool = global_pool

    def forward(self, data, conv_only=False):

        if not hasattr(data,'batch') or data.batch is None:
            # Create virtual null batch if singlet graph input
            setattr(data, 'batch', torch.tensor(np.zeros(data.x.shape[0]), dtype=torch.long))

        x = F.elu(self.conv1(data.x, data.edge_index))
        x = F.dropout(x, training=self.training)

        x = F.elu(self.conv2(x,      data.edge_index))
        x = F.dropout(x, training=self.training)

        # ** Global pooling **
        if self.task == 'graph':
            if self.global_pool == 's2s':
                x = self.S2Spool(x, data.batch)
                x = self.S2Slin(x)
            elif self.global_pool == 'max':
                x = global_max_pool(x, data.batch)
            elif self.global_pool == 'add':
                x = global_add_pool(x, data.batch)
            elif self.global_pool == 'mean':
                x = global_mean_pool(x, data.batch)
            else:
                raise Exception(__name__ + f': Unknown global_pool <{self.global_pool}>')

        if conv_only: # Return convolution part
            return x

        # Global features concatenated
        if self.G > 0:
            u = data.u.view(-1, self.G)
            x = torch.cat((x, u), 1)

        x = F.relu(self.mlp1(x))
        x = F.relu(self.mlp2(x))

        return x
    
    # Returns softmax probability
    def softpredict(self,x):
        if self.training:
            return F.log_softmax(self.forward(x), dim=-1) # Numerically more stable
        else:
            return F.softmax(self.forward(x), dim=-1)


# GINEConv based graph net
#
# https://arxiv.org/abs/1810.00826
# https://arxiv.org/abs/1905.12265
#
class GINENet(torch.nn.Module):
    def __init__(self, D, C, G=0, E=None, cdim=96, conv_aggr=None, global_pool='max', task='node'):
        super(GINENet, self).__init__()

        self.D = D
        self.C = C
        self.G = G
        self.cdim = cdim

        self.task  = task
        self.global_pool = global_pool

        # Convolution layers
        self.conv1 = GINEConv(MLP([self.D, self.D]))
        self.conv2 = GINEConv(MLP([self.D, 64]))
        
        # "Fusion" layer taking in conv1 and conv2 outputs
        self.lin1  = MLP([self.D + 64, self.cdim])

        # Set2Set pooling operation produces always output with 2 x input dimension
        # => use linear layer to project down
        if self.global_pool == 's2s':
            self.S2Spool = Set2Set(in_channels=self.cdim, processing_steps=3, num_layers=1)
            self.S2Slin  = Linear(2*self.cdim, self.cdim)

        if (self.G > 0):
            self.Z = self.cdim + self.G
        else:
            self.Z = self.cdim

        # Final layers concatenating everything
        self.mlp1  = MLP([self.Z, self.Z, self.C])


    def forward(self, data, conv_only=False):

        if not hasattr(data,'batch') or data.batch is None:
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
            if self.global_pool == 's2s':
                x = self.S2Spool(x, data.batch)
                x = self.S2Slin(x)
            elif self.global_pool == 'max':
                x = global_max_pool(x, data.batch)
            elif self.global_pool == 'add':
                x = global_add_pool(x, data.batch)
            elif self.global_pool == 'mean':
                x = global_mean_pool(x, data.batch)
            else:
                raise Exception(__name__ + f': Unknown global_pool <{self.global_pool}>')
        
        if conv_only: # Return convolution part
            return x

        # Global features concatenated
        if self.G > 0:
            u = data.u.view(-1, self.G)
            x = torch.cat((x, u), 1)

        # Final layers
        x = self.mlp1(x)
        return x

    # Returns softmax probability
    def softpredict(self,x):
        if self.training:
            return F.log_softmax(self.forward(x), dim=-1) # Numerically more stable
        else:
            return F.softmax(self.forward(x), dim=-1)
