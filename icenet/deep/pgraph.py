# https://arxiv.org/abs/2006.16811
# 
# Path Integral Based Convolution and Pooling for Graph Neural Networks
#
# From: https://github.com/YuGuangWang/PAN

import torch
import torch.nn.functional as F

from torch.nn import Parameter
from torch_scatter import scatter_add, scatter_max, scatter_mean

from torch_geometric.utils import softmax, degree
from torch_geometric.nn import MessagePassing
from torch_geometric.data import DataLoader, Data
from torch_geometric.datasets import TUDataset
from torch_geometric.utils.num_nodes import maybe_num_nodes
#from torch_geometric.nn.pool import TopKPooling, SAGPooling

from torch.utils.data import random_split

from torch_sparse import spspmm
from torch_sparse import coalesce
from torch_sparse import eye



class PANConv(MessagePassing):
    def __init__(self, in_channels, out_channels, filter_size=4, panconv_filter_weight=None):
        super(PANConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.m = None
        self.filter_size = filter_size
        if panconv_filter_weight is None:
            self.panconv_filter_weight = torch.nn.Parameter(0.5 * torch.ones(filter_size), requires_grad=True)

    def forward(self, x, edge_index, num_nodes=None, edge_mask_list=None):
        # x has shape [N, in_channels]
        if edge_mask_list is None:
            AFTERDROP = False
        else:
            AFTERDROP = True

        # edge_index has shape [2, E]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        # Step 1: Path integral
        edge_index, edge_weight = self.panentropy_sparse(edge_index, num_nodes, AFTERDROP, edge_mask_list)
        #edge_index, edge_weight = self.panentropy_sparse(edge_index, num_nodes)
        #edge_index, edge_weight = self.panentropy(edge_index, num_nodes)
        

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)
        x_size0 = x.size(0)

        # Step 3: Compute normalization
        row, col = edge_index
        deg = degree(row, x_size0, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        norm = norm.mul(edge_weight)

        # save M
        m_list = norm.mul(edge_weight).view(-1, 1).squeeze()
        m_adj = torch.zeros(x_size0, x_size0, device=edge_index.device)
        m_adj[row, col] = m_list
        self.m = m_adj

        # Step 4-6: Start propagating messages.
        return self.propagate(edge_index, size=(x_size0, x_size0), x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out

    def panentropy(self, edge_index, num_nodes):

        # sparse to dense
        # adj = to_dense_adj(edge_index)
        adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
        adj[edge_index[0, :], edge_index[1, :]] = 1

        # iteratively add weighted matrix power
        adjtmp = torch.eye(num_nodes, device=edge_index.device)
        pan_adj = self.panconv_filter_weight[0] * torch.eye(num_nodes, device=edge_index.device)

        for i in range(self.filter_size - 1):
            adjtmp = torch.mm(adjtmp, adj)
            pan_adj = pan_adj + self.panconv_filter_weight[i+1] * adjtmp

        # dense to sparse
        edge_index_new = torch.nonzero(pan_adj).t()
        edge_weight_new = pan_adj[edge_index_new[0], edge_index_new[1]]

        return edge_index_new, edge_weight_new

    def panentropy_sparse(self, edge_index, num_nodes, AFTERDROP, edge_mask_list):

        edge_value = torch.ones(edge_index.size(1), device=edge_index.device)
        edge_index, edge_value = coalesce(edge_index, edge_value, num_nodes, num_nodes)

        # iteratively add weighted matrix power
        pan_index, pan_value = eye(num_nodes, device=edge_index.device)
        indextmp = pan_index.clone().to(edge_index.device)
        valuetmp = pan_value.clone().to(edge_index.device)

        pan_value = self.panconv_filter_weight[0] * pan_value

        for i in range(self.filter_size - 1):
            if AFTERDROP:
                indextmp, valuetmp = spspmm(indextmp, valuetmp, edge_index, edge_value * edge_mask_list[i], num_nodes, num_nodes, num_nodes)
            else:
                indextmp, valuetmp = spspmm(indextmp, valuetmp, edge_index, edge_value, num_nodes, num_nodes, num_nodes)
            valuetmp = valuetmp * self.panconv_filter_weight[i+1]
            indextmp, valuetmp = coalesce(indextmp, valuetmp, num_nodes, num_nodes)
            pan_index = torch.cat((pan_index, indextmp), 1)
            pan_value = torch.cat((pan_value, valuetmp))

        return coalesce(pan_index, pan_value, num_nodes, num_nodes, op='add')



# equation 14
class PANUMPooling(torch.nn.Module):
    r""" Specific Graph pooling layer based on unnormalized M from PAN, which can only work after PANConv.
    """
    def __init__(self, in_channels, ratio=0.5, min_score=None, multiplier=1, nonlinearity=torch.tanh):
        super(PANUMPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

    def forward(self, x, edge_index, edge_weight=None, M=None, UM=None, batch=None, num_nodes=None):

        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        # compute score
        diag_UM = torch.diag(M)
        score = diag_UM.squeeze()

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = self.topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]

        edge_index, edge_weight = self.filter_adj(edge_index, edge_weight, perm, num_nodes=score.size(0))

        return x, edge_index, edge_weight, batch, perm, score[perm]

    def topk(self, x, ratio, batch, min_score=None, tol=1e-7):

        if min_score is not None:
            # Make sure that we do not drop all nodes in a graph.
            scores_max = scatter_max(x, batch)[0][batch] - tol
            scores_min = scores_max.clamp(max=min_score)

            perm = torch.nonzero(x > scores_min).view(-1)
        else:
            num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
            batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

            cum_num_nodes = torch.cat(
                [num_nodes.new_zeros(1),
                 num_nodes.cumsum(dim=0)[:-1]], dim=0)

            index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
            index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

            dense_x = x.new_full((batch_size * max_num_nodes, ), -2)
            dense_x[index] = x
            dense_x = dense_x.view(batch_size, max_num_nodes)

            _, perm = dense_x.sort(dim=-1, descending=True)

            perm = perm + cum_num_nodes.view(-1, 1)
            perm = perm.view(-1)

            k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
            mask = [
                torch.arange(k[i], dtype=torch.long, device=x.device) +
                i * max_num_nodes for i in range(batch_size)
            ]
            mask = torch.cat(mask, dim=0)

            perm = perm[mask]

        return perm

    def filter_adj(self, edge_index, edge_weight, perm, num_nodes=None):

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        mask = perm.new_full((num_nodes, ), -1)
        i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
        mask[perm] = i

        row, col = edge_index
        row, col = mask[row], mask[col]
        mask = (row >= 0) & (col >= 0)
        row, col = row[mask], col[mask]

        if edge_weight is not None:
            edge_weight = edge_weight[mask]

        return torch.stack([row, col], dim=0), edge_weight


# equation 15
class PANXUMPooling(torch.nn.Module):
    r""" General Graph pooling layer based on PAN, which can work with all layers.
    """
    def __init__(self, in_channels, ratio=0.5, pan_pool_weight=None, min_score=None, multiplier=1,
                 nonlinearity=torch.tanh, filter_size=3, panpool_filter_weight=None):
        super(PANXUMPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity

        self.transform = Parameter(torch.ones(in_channels), requires_grad=True)

        if pan_pool_weight is None:
            #self.weight = torch.tensor([0.7, 0.3], device=self.transform.device)
            self.pan_pool_weight = torch.nn.Parameter(0.5 * torch.ones(2), requires_grad=True)
        else:
            self.pan_pool_weight = pan_pool_weight

    def forward(self, x, edge_index, edge_weight=None, M=None, UM=None, batch=None, num_nodes=None):

        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        # diag of unnormalized M
        diag_UM = torch.diag(M).squeeze()

        # linear transform
        xtransform = torch.matmul(x, self.transform)

        # aggregate score
        score = self.pan_pool_weight[0] * xtransform + self.pan_pool_weight[1] * diag_UM

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = self.topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_weight = self.filter_adj(edge_index, edge_weight, perm, num_nodes=score.size(0))

        return x, edge_index, edge_weight, batch, perm, score[perm]

    def topk(self, x, ratio, batch, min_score=None, tol=1e-7):

        if min_score is not None:
            # Make sure that we do not drop all nodes in a graph.
            scores_max = scatter_max(x, batch)[0][batch] - tol
            scores_min = scores_max.clamp(max=min_score)

            perm = torch.nonzero(x > scores_min).view(-1)
        else:
            num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
            batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

            cum_num_nodes = torch.cat(
                [num_nodes.new_zeros(1),
                 num_nodes.cumsum(dim=0)[:-1]], dim=0)

            index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
            index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

            dense_x = x.new_full((batch_size * max_num_nodes, ), -2)
            dense_x[index] = x
            dense_x = dense_x.view(batch_size, max_num_nodes)

            _, perm = dense_x.sort(dim=-1, descending=True)

            perm = perm + cum_num_nodes.view(-1, 1)
            perm = perm.view(-1)

            k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
            mask = [
                torch.arange(k[i], dtype=torch.long, device=x.device) +
                i * max_num_nodes for i in range(batch_size)
            ]
            mask = torch.cat(mask, dim=0)

            perm = perm[mask]

        return perm

    def filter_adj(self, edge_index, edge_weight, perm, num_nodes=None):

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        mask = perm.new_full((num_nodes, ), -1)
        i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
        mask[perm] = i

        row, col = edge_index
        row, col = mask[row], mask[col]
        mask = (row >= 0) & (col >= 0)
        row, col = row[mask], col[mask]

        if edge_weight is not None:
            edge_weight = edge_weight[mask]

        return torch.stack([row, col], dim=0), edge_weight

    def panentropy_sparse(self, edge_index, num_nodes):

        edge_value = torch.ones(edge_index.size(1), device=edge_index.device)
        edge_index, edge_value = coalesce(edge_index, edge_value, num_nodes, num_nodes)

        # iteratively add weighted matrix power
        pan_index, pan_value = eye(num_nodes, device=edge_index.device)
        indextmp = pan_index.clone().to(edge_index.device)
        valuetmp = pan_value.clone().to(edge_index.device)

        pan_value = self.panpool_filter_weight[0] * pan_value

        for i in range(self.filter_size - 1):
            #indextmp, valuetmp = coalesce(indextmp, valuetmp, num_nodes, num_nodes)
            indextmp, valuetmp = spspmm(indextmp, valuetmp, edge_index, edge_value, num_nodes, num_nodes, num_nodes)
            valuetmp = valuetmp * self.panpool_filter_weight[i+1]
            indextmp, valuetmp = coalesce(indextmp, valuetmp, num_nodes, num_nodes)
            pan_index = torch.cat((pan_index, indextmp), 1)
            pan_value = torch.cat((pan_value, valuetmp))

        return coalesce(pan_index, pan_value, num_nodes, num_nodes, op='add')


# equation 16
class PANXHMPooling(torch.nn.Module):
    r""" General Graph pooling layer based on PAN, which can work with all layers.
    """
    def __init__(self, in_channels, ratio=0.5, pan_pool_weight=None, min_score=None, multiplier=1,
                 nonlinearity=torch.tanh, filter_size=3, panpool_filter_weight=None):
        super(PANXHMPooling, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity
        self.transform = Parameter(torch.ones(in_channels), requires_grad=True)


    def forward(self, x, edge_index, edge_weight=None, M=None, UM=None, batch=None, num_nodes=None):

        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        # diag of unnormalized M
        diag_M = torch.diag(M).squeeze()

        # linear transform
        xtransform = torch.matmul(x, self.transform)

        # aggregate score
        score = xtransform * diag_M

        if self.min_score is None:
            score = self.nonlinearity(score)
        else:
            score = softmax(score, batch)

        perm = self.topk(score, self.ratio, batch, self.min_score)
        x = x[perm] * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x

        batch = batch[perm]
        edge_index, edge_weight = self.filter_adj(edge_index, edge_weight, perm, num_nodes=score.size(0))

        return x, edge_index, edge_weight, batch, perm, score[perm]

    def topk(self, x, ratio, batch, min_score=None, tol=1e-7):

        if min_score is not None:
            # Make sure that we do not drop all nodes in a graph.
            scores_max = scatter_max(x, batch)[0][batch] - tol
            scores_min = scores_max.clamp(max=min_score)

            perm = torch.nonzero(x > scores_min).view(-1)
        else:
            num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
            batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

            cum_num_nodes = torch.cat(
                [num_nodes.new_zeros(1),
                 num_nodes.cumsum(dim=0)[:-1]], dim=0)

            index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
            index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

            dense_x = x.new_full((batch_size * max_num_nodes, ), -2)
            dense_x[index] = x
            dense_x = dense_x.view(batch_size, max_num_nodes)

            _, perm = dense_x.sort(dim=-1, descending=True)

            perm = perm + cum_num_nodes.view(-1, 1)
            perm = perm.view(-1)

            k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
            mask = [
                torch.arange(k[i], dtype=torch.long, device=x.device) +
                i * max_num_nodes for i in range(batch_size)
            ]
            mask = torch.cat(mask, dim=0)

            perm = perm[mask]

        return perm

    def filter_adj(self, edge_index, edge_weight, perm, num_nodes=None):

        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        mask = perm.new_full((num_nodes, ), -1)
        i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
        mask[perm] = i

        row, col = edge_index
        row, col = mask[row], mask[col]
        mask = (row >= 0) & (col >= 0)
        row, col = row[mask], col[mask]

        if edge_weight is not None:
            edge_weight = edge_weight[mask]

        return torch.stack([row, col], dim=0), edge_weight

    def panentropy_sparse(self, edge_index, num_nodes):

        edge_value = torch.ones(edge_index.size(1), device=edge_index.device)
        edge_index, edge_value = coalesce(edge_index, edge_value, num_nodes, num_nodes)

        # iteratively add weighted matrix power
        pan_index, pan_value = eye(num_nodes, device=edge_index.device)
        indextmp = pan_index.clone().to(edge_index.device)
        valuetmp = pan_value.clone().to(edge_index.device)

        pan_value = self.panpool_filter_weight[0] * pan_value

        for i in range(self.filter_size - 1):
            #indextmp, valuetmp = coalesce(indextmp, valuetmp, num_nodes, num_nodes)
            indextmp, valuetmp = spspmm(indextmp, valuetmp, edge_index, edge_value, num_nodes, num_nodes, num_nodes)
            valuetmp = valuetmp * self.panpool_filter_weight[i+1]
            indextmp, valuetmp = coalesce(indextmp, valuetmp, num_nodes, num_nodes)
            pan_index = torch.cat((pan_index, indextmp), 1)
            pan_value = torch.cat((pan_value, valuetmp))

        return coalesce(pan_index, pan_value, num_nodes, num_nodes, op='add')



### define dropout

class PANDropout(torch.nn.Module):
    def __init__(self, filter_size=4):
        super(PANDropout, self).__init__()

        self.filter_size =filter_size

    def forward(self, edge_index, p=0.5):
        # p - probability of an element to be zeroed

        # sava all network
        #edge_mask_list = []
        edge_mask_list = torch.empty(0)
        edge_mask_list.to(edge_index.device)

        num = edge_index.size(1)
        bern = torch.distributions.bernoulli.Bernoulli(torch.tensor([p]))

        for i in range(self.filter_size - 1):
            edge_mask = bern.sample([num]).squeeze()
            #edge_mask_list.append(edge_mask)
            edge_mask_list = torch.cat([edge_mask_list, edge_mask])

        return True, edge_mask_list

