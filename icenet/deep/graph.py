# Graph Neural Nets
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.nn import Sequential, Linear, ReLU, Dropout, BatchNorm1d
import torch_geometric

from   typing import Union, Tuple, Callable
from   torch_geometric.typing import OptTensor, OptPairTensor, Adj, Size

from   torch_geometric.nn import Set2Set, global_add_pool, global_mean_pool, global_max_pool, global_sort_pool
from   torch_geometric.nn import NNConv, GINEConv, GATConv, SplineConv, GCNConv, SGConv, SAGEConv, EdgeConv, DynamicEdgeConv
from   torch_geometric.nn import MessagePassing

from torch_scatter import scatter_add, scatter_max, scatter_mean

from icenet.deep.da import GradientReversal

from icenet.deep.pgraph import *
from icenet.deep.dmlp import MLP, MLP_ALL_ACT

from typing import Callable, Union, Optional
from torch_geometric.typing import OptTensor, PairTensor, PairOptTensor, Adj

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing


class SuperEdgeConv(MessagePassing):
    r"""
    Custom GNN
    """
    def __init__(self, nn: Callable, nn_final: Callable, eps: float=0.0,
        train_eps: bool=True, aggr: str='mean', mp_attn_dim: int=0, **kwargs):
        
        if aggr == 'attention':
            aggr = torch_geometric.nn.aggr.MultiAggregation(aggrs=['mean', 'std', 'max', 'min'], mode='attn',
                mode_kwargs={'in_channels':mp_attn_dim, 'out_channels':mp_attn_dim, 'num_heads':4})

        super().__init__(aggr=aggr, **kwargs)
        self.nn       = nn
        self.nn_final = nn_final
        
        """
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        """

        self.reset_parameters()

    def reset_parameters(self):
        torch_geometric.nn.inits.reset(self.nn)
        torch_geometric.nn.inits.reset(self.nn_final)
        #self.eps.data.fill_(self.initial_eps)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
            edge_attr: OptTensor = None, edge_weight: OptTensor = None, size: Size = None) -> Tensor:
        """"""
        #print(f'forward: x: {x.shape}')

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_weight=edge_weight, size=None)

        return self.nn_final(torch.concat([x[0], out], dim=-1))

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: OptTensor, edge_weight: OptTensor) -> Tensor:

        #print(f'message: x_i: {x_i.shape}')
        #print(f'message: x_j: {x_j.shape}')

        if edge_attr is not None and len(edge_attr.shape) == 1: # if 1-dim edge_attributes
            edge_attr = edge_attr[:,None]

        if edge_attr is not None:
            out = self.nn(torch.cat([x_i, x_j - x_i, x_j * x_i, edge_attr], dim=-1))
        else:
            out = self.nn(torch.cat([x_i, x_j - x_i, x_j * x_i], dim=-1))

        return out if edge_weight is None else out * edge_weight.view(-1, 1)

    # Memory efficient fusion of message and aggregate via sparse matrix
    #def message_and_aggregate(self, adj_t: SparseTensor,
    #                          x: OptPairTensor) -> Tensor:
    #    adj_t = adj_t.set_value(None, layout=None)
    #    return matmul(adj_t, x[0], reduce=self.aggr)

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

"""
class SuperEdgeConv(MessagePassing):
    #Custom convolution operator.
    
    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super(SuperEdgeConv, self).__init__(aggr=aggr, **kwargs)
        self.nn = nn
        #self.reset_parameters()

    #def reset_parameters(self):
        #reset(self.nn)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj,
            edge_attr: OptTensor = None, size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:

        return self.nn(torch.cat([x_i, x_j - x_i, edge_attr], dim=-1))

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)
"""

class GNNGeneric(torch.nn.Module):
    """
    Technical Remarks:
        
        Remember always to use MLP_ALL_ACT in the intermediate blocks, i.e.
        MLPs with an activation function also after the last layer.
        (otherwise very bad performance may happen for certain message passing / convolution operators).
        
    """
    
    def SuperEdgeConv_(self, data):
        # Message passing
        x1 = self.conv1(data.x, data.edge_index, data.edge_attr)
        x2 = self.conv2(x1,     data.edge_index, data.edge_attr)
        x3 = self.conv3(x2,     data.edge_index, data.edge_attr)
        
        # Apply "residual-fusion"
        return self.lin1(torch.cat([x1, x2, x3], dim=1))

    def EdgeConv_(self, data):
        # Message passing
        x1 = self.conv1(data.x, data.edge_index)
        x2 = self.conv2(x1,     data.edge_index)
        x3 = self.conv3(x2,     data.edge_index)
        
        # Apply "residual-fusion"
        return self.lin1(torch.cat([x1, x2, x3], dim=1))

    def DynamicEdgeConv_(self, data):
        # Message passing
        x1 = self.conv1(data.x, data.batch)
        x2 = self.conv2(x1,     data.batch)
        x3 = self.conv3(x2,     data.batch)
        
        # Apply "residual-fusion"
        return self.lin1(torch.cat([x1, x2, x3], dim=1))

    def NNConv_(self, data):
        # Message passing
        x1 = self.conv1(data.x, data.edge_index, data.edge_attr)
        x2 = self.conv2(x1,     data.edge_index, data.edge_attr)
        x3 = self.conv3(x2,     data.edge_index, data.edge_attr)
        
        # Apply "residual-fusion"
        return self.lin1(torch.cat([x1, x2, x3], dim=1))

    def GATConv_(self, data):
        # Message passing
        x1 = self.conv1(data.x, data.edge_index)
        x2 = self.conv2(x1,     data.edge_index)
        x3 = self.conv3(x2,     data.edge_index)
        
        # Apply "residual-fusion"
        return self.lin1(torch.cat([x1, x2, x3], dim=1))

    def SplineConv_(self, data):
        # Message passing
        x1 = self.conv1(data.x, data.edge_index, data.edge_attr)
        x1 = F.elu(x1)
        
        x2 = self.conv2(x1,     data.edge_index, data.edge_attr)
        x2 = F.elu(x2)
        
        x3 = self.conv3(x2,     data.edge_index, data.edge_attr)
        x3 = F.elu(x3)

        # Apply "residual-fusion"
        return self.lin1(torch.cat([x1, x2, x3], dim=1))

    def SAGEConv_(self, data):
        # Message passing
        x1 = self.conv1(data.x, data.edge_index)
        x2 = self.conv2(x1,     data.edge_index)
        x3 = self.conv3(x2,     data.edge_index)
        
        # Apply "residual-fusion"
        return self.lin1(torch.cat([x1, x2, x3], dim=1))

    def SGConv_(self, data):
        # Message passing
        x1 = self.conv1(data.x, data.edge_index)
        x2 = self.conv2(x1,     data.edge_index)
        x3 = self.conv3(x2,     data.edge_index)
        
        # Apply "residual-fusion"
        return self.lin1(torch.cat([x1, x2, x3], dim=1))

    def GINEConv_(self, data):
        # Message passing
        x, edge_attr = self.GINE_helper(data)
        x1 = self.conv1(x,  data.edge_index, edge_attr)
        x2 = self.conv2(x1, data.edge_index, edge_attr)
        x3 = self.conv3(x2, data.edge_index, edge_attr)        

        # Apply "residual-fusion"
        return self.lin1(torch.cat([x1, x2, x3], dim=1))

    def PANConv_(self, data):
        # Message passing
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

        return x

    def __init__(self, d_dim, c_dim, u_dim=0, e_dim=None, z_dim=96,
        conv_type        = 'EdgeConv',
        task             = 'node',
        global_pool      = 'mean',
        
        conv_MLP_act     = 'relu',
        conv_MLP_bn      = True,
        conv_MLP_dropout = 0.0,
        conv_aggr        = 'max',
        conv_knn         = 8,
        
        fusion_MLP_act     = 'relu',
        fusion_MLP_bn      = False,
        fusion_MLP_dropout = 0.0,

        final_MLP_act      = 'relu',
        final_MLP_bn       = False,
        final_MLP_dropout  = 0.0,

        DA_active        = False,
        DA_alpha         = 1.0,
        DA_MLP           = [128, 64],

        DA_MLP_act       = 'relu',
        DA_MLP_bn        = False,
        DA_MLP_dropout   = 0.0
        ):

        super(GNNGeneric, self).__init__()
        
        self.d_dim = d_dim  # node feature dimension
        self.u_dim = u_dim  # graph global feature dimension
        self.e_dim = e_dim  # edge feature dimension

        self.c_dim = c_dim  # number of output classes
        self.C     = c_dim  # (used elsewhere in icenet)

        self.z_dim = z_dim  # latent dimension

        self.task           = task           # 'node', 'edge', 'graph'
        self.global_pool    = global_pool    # 's2s', 'max', 'mean', 'add'


        self.DA_active      = DA_active      # Domain adaptation (True, False)
        self.DA_alpha       = DA_alpha 
        self.DA_MLP         = DA_MLP
        self.DA_MLP_act     = DA_MLP_act
        self.DA_MLP_bn      = DA_MLP_bn
        self.DA_MLP_dropout = DA_MLP_dropout

        # SuperEdgeConv,   https://arxiv.org/abs/xyz
        if   conv_type == 'SuperEdgeConv':
            self.conv1 = SuperEdgeConv(aggr=conv_aggr, mp_attn_dim=self.z_dim,
                nn=MLP_ALL_ACT([3 * self.d_dim + self.e_dim, self.z_dim], activation=conv_MLP_act, batch_norm=conv_MLP_bn, dropout=conv_MLP_dropout),
                nn_final=MLP_ALL_ACT([self.d_dim + self.z_dim, self.z_dim], activation=conv_MLP_act, batch_norm=conv_MLP_bn, dropout=conv_MLP_dropout))
            self.conv2 = SuperEdgeConv(aggr=conv_aggr, mp_attn_dim=self.z_dim,
                nn=MLP_ALL_ACT([3 * self.z_dim + self.e_dim, self.z_dim], activation=conv_MLP_act, batch_norm=conv_MLP_bn, dropout=conv_MLP_dropout),
                nn_final=MLP_ALL_ACT([2*self.z_dim, self.z_dim], activation=conv_MLP_act, batch_norm=conv_MLP_bn, dropout=conv_MLP_dropout))
            self.conv3 = SuperEdgeConv(aggr=conv_aggr, mp_attn_dim=self.z_dim,
                nn=MLP_ALL_ACT([3 * self.z_dim + self.e_dim, self.z_dim], activation=conv_MLP_act, batch_norm=conv_MLP_bn, dropout=conv_MLP_dropout),
                nn_final=MLP_ALL_ACT([2*self.z_dim, self.z_dim], activation=conv_MLP_act, batch_norm=conv_MLP_bn, dropout=conv_MLP_dropout))
            
            self.lin1  = MLP_ALL_ACT([3*self.z_dim, self.z_dim],
                activation=fusion_MLP_act, batch_norm=fusion_MLP_bn, dropout=fusion_MLP_dropout)

            self.conv  = self.SuperEdgeConv_

        # EdgeConv,        https://arxiv.org/abs/1801.07829
        elif conv_type == 'EdgeConv':
            self.conv1 = EdgeConv(MLP_ALL_ACT([2 * self.d_dim, self.z_dim],
                activation=conv_MLP_act, batch_norm=conv_MLP_bn, dropout=conv_MLP_dropout), aggr=conv_aggr)
            self.conv2 = EdgeConv(MLP_ALL_ACT([2 * self.z_dim, self.z_dim],
                activation=conv_MLP_act, batch_norm=conv_MLP_bn, dropout=conv_MLP_dropout), aggr=conv_aggr)
            self.conv3 = EdgeConv(MLP_ALL_ACT([2 * self.z_dim, self.z_dim],
                activation=conv_MLP_act, batch_norm=conv_MLP_bn, dropout=conv_MLP_dropout), aggr=conv_aggr)
            
            self.lin1  = MLP_ALL_ACT([3*self.z_dim, self.z_dim],
                activation=fusion_MLP_act, batch_norm=fusion_MLP_bn, dropout=fusion_MLP_dropout)

            self.conv  = self.EdgeConv_

        # DynamicEdgeConv, https://arxiv.org/abs/1801.07829
        elif conv_type == 'DynamicEdgeConv':
            self.conv1 = DynamicEdgeConv(MLP_ALL_ACT([2 * self.d_dim, self.z_dim],
                activation=conv_MLP_act, batch_norm=conv_MLP_bn, dropout=conv_MLP_dropout), k=conv_knn, aggr=conv_aggr)
            self.conv2 = DynamicEdgeConv(MLP_ALL_ACT([2 * self.z_dim, self.z_dim],
                activation=conv_MLP_act, batch_norm=conv_MLP_bn, dropout=conv_MLP_dropout), k=conv_knn, aggr=conv_aggr)
            self.conv3 = DynamicEdgeConv(MLP_ALL_ACT([2 * self.z_dim, self.z_dim],
                activation=conv_MLP_act, batch_norm=conv_MLP_bn, dropout=conv_MLP_dropout), k=conv_knn, aggr=conv_aggr)
            
            self.lin1  = MLP_ALL_ACT([3*self.z_dim, self.z_dim],
                activation=fusion_MLP_act, batch_norm=fusion_MLP_bn, dropout=fusion_MLP_dropout)

            self.conv  = self.DynamicEdgeConv_

        # NNConv,          https://arxiv.org/abs/1704.01212
        # nn with size [-1, num_edge_features] x [-1, in_channels * out_channels]
        elif conv_type == 'NNConv':
            self.conv1 = NNConv(in_channels=self.d_dim, out_channels=self.d_dim, nn=MLP_ALL_ACT([self.e_dim, self.d_dim**2],
                activation=conv_MLP_act, batch_norm=conv_MLP_bn, dropout=conv_MLP_dropout), aggr=conv_aggr)
            self.conv2 = NNConv(in_channels=self.d_dim, out_channels=self.d_dim, nn=MLP_ALL_ACT([self.e_dim, self.d_dim**2],
                activation=conv_MLP_act, batch_norm=conv_MLP_bn, dropout=conv_MLP_dropout), aggr=conv_aggr)
            self.conv3 = NNConv(in_channels=self.d_dim, out_channels=self.d_dim, nn=MLP_ALL_ACT([self.e_dim, self.d_dim**2],
                activation=conv_MLP_act, batch_norm=conv_MLP_bn, dropout=conv_MLP_dropout), aggr=conv_aggr)
            
            self.lin1  = MLP_ALL_ACT([self.d_dim + self.d_dim + self.d_dim, self.z_dim],
                activation=fusion_MLP_act, batch_norm=fusion_MLP_bn, dropout=fusion_MLP_dropout)

            self.conv  = self.NNConv_

        # GATConv,         https://arxiv.org/abs/1710.10903
        elif conv_type == 'GATConv':

            heads = 2

            self.conv1 = GATConv(self.d_dim,         self.z_dim, heads=heads, dropout=conv_MLP_dropout)
            self.conv2 = GATConv(self.z_dim * heads, self.z_dim, heads=heads, dropout=conv_MLP_dropout)
            self.conv3 = GATConv(self.z_dim * heads, self.z_dim, heads=heads, dropout=conv_MLP_dropout)
            
            self.lin1  = MLP_ALL_ACT([3 * self.z_dim * heads, self.z_dim],
                activation=fusion_MLP_act, batch_norm=fusion_MLP_bn, dropout=fusion_MLP_dropout)

            self.conv  = self.GATConv_

        # SplineConv,      https://arxiv.org/abs/1711.08920
        elif conv_type == 'SplineConv':

            self.conv1 = SplineConv(self.d_dim, self.d_dim, dim=self.e_dim, degree=1, kernel_size=3)
            self.conv2 = SplineConv(self.d_dim, self.d_dim, dim=self.e_dim, degree=1, kernel_size=5)
            self.conv3 = SplineConv(self.d_dim, self.d_dim, dim=self.e_dim, degree=1, kernel_size=5)
            
            self.lin1  = MLP_ALL_ACT([3*self.d_dim, self.z_dim],
                activation=fusion_MLP_act, batch_norm=fusion_MLP_bn, dropout=fusion_MLP_dropout)

            self.conv  = self.SplineConv_

        # SAGEConv,        https://arxiv.org/abs/1706.02216
        elif conv_type == 'SAGEConv':
            self.conv1 = SAGEConv(self.d_dim, self.d_dim)
            self.conv2 = SAGEConv(self.d_dim, self.d_dim)
            self.conv3 = SAGEConv(self.d_dim, self.d_dim)
            
            self.lin1  = MLP_ALL_ACT([3*self.d_dim, self.z_dim],
                activation=fusion_MLP_act, batch_norm=fusion_MLP_bn, dropout=fusion_MLP_dropout)
            
            self.conv  = self.SAGEConv_

        # SGConv,          https://arxiv.org/abs/1902.07153
        elif conv_type == 'SGConv':
            K = 1 # Change this to input parameters

            self.conv1 = SGConv(self.d_dim, self.d_dim, K, cached=False)
            self.conv2 = SGConv(self.d_dim, self.d_dim, K, cached=False)
            self.conv3 = SGConv(self.d_dim, self.d_dim, K, cached=False)
            
            self.lin1  = MLP_ALL_ACT([3*self.d_dim, self.z_dim],
                activation=fusion_MLP_act, batch_norm=fusion_MLP_bn, dropout=fusion_MLP_dropout)
            
            self.conv  = self.SGConv_
            
        # GINEConv,        https://arxiv.org/abs/1810.00826, https://arxiv.org/abs/1905.12265
        elif conv_type == 'GINEConv':
            self.conv1 = GINEConv(MLP([self.d_dim, self.z_dim, self.z_dim])),
            self.conv2 = GINEConv(MLP([self.z_dim, self.z_dim, self.z_dim]))
            self.conv3 = GINEConv(MLP([self.z_dim, self.z_dim, self.z_dim]))
            
            self.lin1  = MLP_ALL_ACT([3*self.z_dim, self.z_dim],
                activation=fusion_MLP_act, batch_norm=fusion_MLP_bn, dropout=fusion_MLP_dropout)
           
            self.conv  = self.GINEConv_

        # PANConv,         https://arxiv.org/abs/2006.16811
        elif conv_type == 'PANConv':
            filter_size = 5

            self.conv1 = PANConv(self.d_dim, self.z_dim, filter_size)
            self.pool1 = PANXUMPooling(self.z_dim)
            # self.drop1 = PANDropout()

            self.conv2 = PANConv(self.z_dim, self.z_dim, filter_size)
            self.pool2 = PANXUMPooling(self.z_dim)
            # self.drop2 = PANDropout()

            self.conv3 = PANConv(self.z_dim, self.z_dim, filter_size)
            self.pool3 = PANXUMPooling(self.z_dim)
            
            self.conv  = self.PANConv_
            
        else:
            raise Exception(__name__ + f'.GraphNetGeneric: Unknown conv_type = {conv_type}')

        # ----------------------------------------------------
        ## Pooling for graph level inference
        ## Set2Set pooling operation produces always output with 2 x input dimension
        # => use linear layer to project down

        if self.task == 'graph' and self.global_pool == 's2s':
            self.S2Spool = Set2Set(in_channels=self.z_dim, processing_steps=3, num_layers=1)
            self.S2Slin  = Linear(2*self.z_dim, self.z_dim)
        # ----------------------------------------------------

        ## Add global feature dimension
        if (self.u_dim > 0):
            self.z_dim = self.z_dim + self.u_dim
        

        # ----------------------------------------------------
        ## Final MLP

        # Node level or graph level inference
        if self.task == 'node' or self.task == 'graph':
            self.mlp_final = MLP([self.z_dim, self.z_dim // 2, self.z_dim // 2, self.c_dim],
                activation=final_MLP_act, batch_norm=final_MLP_bn, dropout=final_MLP_dropout)

        # 2-point (node) probability computation function (edge level inference)
        elif self.task == 'edge_asymmetric':
            self.mlp_final = MLP([2 * self.z_dim, self.z_dim // 2, self.z_dim//2, self.c_dim],
                        activation=final_MLP_act, batch_norm=final_MLP_bn, dropout=final_MLP_dropout)
        
        elif self.task == 'edge_symmetric':
            self.mlp_final =  MLP([self.z_dim, self.z_dim // 2, self.z_dim//2, self.c_dim],
                        activation=final_MLP_act, batch_norm=final_MLP_bn, dropout=final_MLP_dropout)
        else:
            raise Exception(__name__ + f'.GraphNetGeneric: Unknown task = {task} parameter')
        # ----------------------------------------------------

        ### Domain adaptation via gradient reversal
        if self.DA_active:
            self.DA_grad_reverse  = GradientReversal(self.DA_alpha)
            self.DA_MLP_net       = MLP([self.c_dim] + self.MLP_dim + [2],
                activation=self.DA_MLP_act, batch_norm=self.DA_MLP_bn, dropout=self.DA_MLP_dropout)

        # ----------------------------------------------------


    def forward_2pt(self, z, edge_index):
        """
        MLP decoder of two-point correlations (edges)

        Because this function is not (necessarily) permutation symmetric between edge_index[0] and [1],
        we can learn (in principle) a directed or undirected edge (adjacency) behavior.
        """
        
        # Not permutation symmetric under i <-> j exchange
        if   self.task == 'edge_asymmetric':
            X = torch.cat((z[edge_index[0], ...], z[edge_index[1], ...]), dim=-1)
        
        # Permutation symmetric under i <-> j exchange
        elif self.task == 'edge_symmetric':
            X = z[edge_index[0], ...] * z[edge_index[1], ...]

        return self.mlp_final(X)


    def GINE_helper(data):
        """
        GINEConv requires node features and edge features with the same dimension.
        Increase dimensionality here.
        """
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

        return x, edge_attr
        
    def forward_with_DA(self, data):
        """
        Forward call with Domain Adaptation
        """
        x     = self.forward(data, conv_only=True)
        x_out = self.inference(x=x, data=data) # GNN-convolution to final inference net
        
        x     = self.DA_grad_reverse(x)
        x_DA  = self.DA_MLP_net(x=x)           # Domain adaptation (source, target) discriminator net

        return x_out, x_DA

    def forward(self, data, conv_only=False):

        if not hasattr(data,'batch') or data.batch is None:
            # Create virtual null batch if singlet graph input
            setattr(data, 'batch', torch.tensor(np.zeros(data.x.shape[0]), dtype=torch.long))
        
        ## Apply GNN message passing layers
        x = self.conv(data)

        ## Global node feature pooling (to handle graph level classification)
        if self.task == 'graph':
            if   self.global_pool == 's2s':
                x = self.S2Spool(x, data.batch)
                x = self.S2Slin(x)
            elif self.global_pool == 'max':
                x = global_max_pool(x, data.batch)
            elif self.global_pool == 'add':
                x = global_add_pool(x, data.batch)
            elif self.global_pool == 'mean':
                x = global_mean_pool(x, data.batch)
            else:
                raise Exception(__name__ + f'.forward: Unknown global_pool <{self.global_pool}>')
        
        # ===========================================
        if conv_only: # Return convolution part
            return x
        # ===========================================

        # Final inference net
        x = self.inference(x=x, data=data)

        return x

    def inference(self, x, data):
        """
        Final inference network forward call
        """

        ## Global features concatenated
        if self.u_dim > 0:
            u = data.u.view(-1, self.u_dim)
            x = torch.cat((x, u), 1)

        ## Final MLP map
        
        # Edge level inference
        if 'edge' in self.task:
            x = self.forward_2pt(x, data.edge_index)

        # Node or graph level inference
        else:
            x = self.mlp_final(x)

        return x

    # Returns softmax probability
    def softpredict(self,x):
        #if self.training:
        #    return F.log_softmax(self.forward(x), dim=-1) # Numerically more stable
        #else:
        return F.softmax(self.forward(x), dim=-1)
