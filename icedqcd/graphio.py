# Graph data readers and parsers for DQCD
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

import numpy as np
import awkward as ak

import numba
from   tqdm import tqdm
import copy
import pickle
import os
import uuid

from termcolor import colored, cprint

import torch
from   torch_geometric.data import Data

import multiprocessing
from   torch.utils.data import dataloader
from   torch.multiprocessing import reductions
from   multiprocessing.reduction import ForkingPickler

import icenet.algo.analytic as analytic
from   icenet.tools import aux
from   icenet.tools.icevec import vec4


def parse_graph_data(X, ids, features, graph_param, Y=None, weights=None, maxevents=None, EPS=1e-12, null_value=float(-999.0)):
    """
    Jagged array data into pytorch-geometric style Data format array.
    
    Args:
        X          :  Jagged Awkward array of variables
        ids        :  Variable names as an array of strings
        features   :  Array of active global feature strings
        graph_param:  Graph construction parameters dict
        Y          :  Target class  array (if any, typically MC only)
        weights    :  (Re-)weighting array (if any, typically MC only)
    
    Returns:
        Array of pytorch-geometric Data objects
    """

    M_MUON     = 0.105658 # Muon mass (GeV)

    global_on  = graph_param['global_on']
    coord      = graph_param['coord']
    directed   = graph_param['directed']
    self_loops = graph_param['self_loops']

    # --------------------------------------------------------------------

    num_node_features   = 4
    num_edge_features   = 4
    num_global_features = len(features)
    
    num_events = np.min([len(X), maxevents]) if maxevents is not None else len(X)
    dataset    = []
    
    print(__name__ + f'.parse_graph_data: Converting {num_events} events into graphs ...')
    zerovec = vec4()
    num_empty = 0

    # Loop over events
    for ev in tqdm(range(num_events)):

        num_nodes = 1 + len(X[ev].Muon.eta) # +1 for virtual node (empty data)
        num_edges = analytic.count_simple_edges(num_nodes=num_nodes, directed=directed, self_loops=self_loops)

        # Construct 4-vector for each object
        p4vec = []
        N_c = len(X[ev].Muon.eta)
        
        if N_c > 0:
            for k in range(N_c):
                v = vec4()
                v.setPtEtaPhiM(X[ev].Muon.pt[k], X[ev].Muon.eta[k], X[ev].Muon.phi[k], M_MUON)
                p4vec.append(v)

        # Empty information
        else:
            num_empty += 1
            # However, never skip empty events here!!, do pre-filtering before this function if needed
        
        # ====================================================================
        # CONSTRUCT TENSORS

        # Construct output class, note [] is important to have for right dimensions
        if Y is not None:
            y = torch.tensor([Y[ev]], dtype=torch.long)
        else:
            y = torch.tensor([0],    dtype=torch.long)

        # Training weights, note [] is important to have for right dimensions
        if weights is not None:
            w = torch.tensor([weights[ev]], dtype=torch.float)
        else:
            w = torch.tensor([1.0],  dtype=torch.float)

        ## Construct node features
        x = get_node_features(p4vec=p4vec, num_nodes=num_nodes, num_node_features=num_node_features, coord=coord)
        x = torch.tensor(x, dtype=torch.float)

        ## Construct edge features
        edge_attr  = analytic.get_Lorentz_edge_features(p4vec=p4vec, num_nodes=num_nodes, num_edges=num_edges, num_edge_features=num_edge_features)
        edge_attr  = torch.tensor(edge_attr, dtype=torch.float)
        
        ## Construct edge connectivity
        edge_index = analytic.get_simple_edge_index(num_nodes=num_nodes, num_edges=num_edges, self_loops=self_loops, directed=directed)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        ## Construct global feature vector
        if global_on == False: # Null the global features
            u = torch.tensor([], dtype=torch.float)
        else:
            u_mat = null_value * np.zeros(len(features))
            for j in range(len(features)):
                x = ak.to_numpy(X[ev][features[j]])
                if x is not []: u_mat[j] = x
            u = torch.tensor(u_mat, dtype=torch.float)
        
        dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, w=w, u=u))
    
    print(__name__ + f'.parse_graph_data: Empty events: {num_empty} / {num_events} = {num_empty/num_events:0.5f} (using only global data u)')        
    
    return dataset


def get_node_features(p4vec, num_nodes, num_node_features, coord):

    # Node feature matrix
    x = np.zeros((num_nodes, num_node_features))

    for i in range(num_nodes):

        # i = 0 case is the virtual node
        if i > 0:
            if   coord == 'ptetaphim':
                x[i,0] = p4vec[i-1].pt
                x[i,1] = p4vec[i-1].eta
                x[i,2] = p4vec[i-1].phi
                x[i,3] = p4vec[i-1].m
            elif coord == 'pxpypze':
                x[i,0] = p4vec[i-1].px
                x[i,1] = p4vec[i-1].py
                x[i,2] = p4vec[i-1].pz
                x[i,3] = p4vec[i-1].e
            else:
                raise Exception(__name__ + f'.get_node_features: Unknown coordinate representation')

    return x
