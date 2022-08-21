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


def parse_graph_data(X, ids, features, node_features, graph_param, Y=None, weights=None, maxevents=None, EPS=1e-12, null_value=float(-999.0)):
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
    M_PION     = 0.139    # Charged pion mass (GeV)

    global_on  = graph_param['global_on']
    coord      = graph_param['coord']
    directed   = graph_param['directed']
    self_loops = graph_param['self_loops']

    # --------------------------------------------------------------------
    num_events = np.min([len(X), maxevents]) if maxevents is not None else len(X)
    dataset    = []
    
    print(__name__ + f'.parse_graph_data: Converting {num_events} events into graphs ...')

    num_global_features = len(features)
    
    num_node_features        = {}
    node_features_hetero_ind = {}

    k = 0
    for key in node_features.keys():
        num_node_features[key]       = len(node_features[key])
        node_features_hetero_ind[key] = np.arange(k, k + num_node_features[key])
        k += num_node_features[key]

    print(__name__ + f'.parse_graph_data: hetero_ind: {node_features_hetero_ind}')
    num_edge_features = 4
    
    # -----------------------------------------------
    # Heterogeneous features per node
    jvname = {}
    for key in num_node_features:
        jvname[key] = []
        for j in range(num_node_features[key]):

            # Awkward groups jagged variables, e.g. 'sv_x' to sv.x
            jvname[key].append(node_features[key][j].split('_', 1)) # argument 1 takes the first '_' occurance
    # -----------------------------------------------

    # Loop over events
    num_empty = 0

    for ev in tqdm(range(num_events)):

        # Count the number of heterogeneous nodes by picking the first feature
        nums = {}
        for key in jvname.keys():
            nums[key] = len(X[ev][jvname[key][0][0]][jvname[key][0][1]])
        
        #print(__name__ + f'.parse_graph_data: event: {ev} | num_nodes: {nums}')

        num_nodes = 1 + np.sum([nums[key] for key in nums.keys()]) # +1 for virtual node (empty data)
        num_edges = analytic.count_simple_edges(num_nodes=num_nodes, directed=directed, self_loops=self_loops)

        # ------------------------------------------------------------------
        # Construct the node features
        p4vec  = []
        fvec   = []
        node_hetero_type = []

        total = 0
        for key in jvname.keys():

            if nums[key] > 0:

                # Loop over nodes of this hetero-type
                for k in range(nums[key]):
                    
                    # Construct 4-momentum
                    v = vec4() # initialized to zero by default

                    if   key == 'muon':
                        v.setPtEtaPhiM(X[ev][jvname['muon'][0][0]].pt[k],
                                       X[ev][jvname['muon'][0][0]].eta[k],
                                       X[ev][jvname['muon'][0][0]].phi[k], M_MUON)
                    elif key == 'jet':
                        v.setPtEtaPhiM(X[ev][jvname['jet'][0][0]].pt[k],
                                       X[ev][jvname['jet'][0][0]].eta[k],
                                       X[ev][jvname['jet'][0][0]].phi[k], 0)
                    elif key == 'cpf':
                        v.setXYZM(X[ev].cpf.px[k], X[ev].cpf.py[k], X[ev].cpf.pz[k], 0)
                    elif key == 'npf':
                        v.setXYZM(X[ev].npf.pz[k], X[ev].npf.py[k], X[ev].npf.pz[k], 0)
                    else:
                        True # Use null 4-vector for 'sv'

                    p4vec.append(v)

                    # Construct abstract feature vector variable by variable
                    v = np.zeros(num_node_features[key])
                    for j in range(len(v)):
                        v[j] = ak.to_numpy(X[ev][jvname[key][j][0]][jvname[key][j][1]][k])

                    fvec.append(v)
                    node_hetero_type.append(key)

        # Empty information
        if num_nodes - 1 == 0:
            num_empty += 1
            # However, never skip empty events here!!, do pre-filtering before this function if needed
        
        # ====================================================================
        # CONSTRUCT TENSORS

        # Construct output class, note [] is important to have for right dimensions
        if Y is not None:
            y = torch.tensor([Y[ev]], dtype=torch.long)
        else:
            y = torch.tensor([0],     dtype=torch.long)

        # Training weights, note [] is important to have for right dimensions
        if weights is not None:
            w = torch.tensor([weights[ev]], dtype=torch.float)
        else:
            w = torch.tensor([1.0],  dtype=torch.float)

        ## Construct node features
        x = get_node_features(num_nodes=num_nodes, num_node_features=num_node_features,
            p4vec=p4vec, fvec=fvec, node_features_hetero_ind=node_features_hetero_ind,
            node_hetero_type=node_hetero_type, coord=coord)

        x[~np.isfinite(x)] = null_value # Input protection
        x = torch.tensor(x, dtype=torch.float)

        ## Construct edge features
        edge_attr  = analytic.get_Lorentz_edge_features(p4vec=p4vec, num_nodes=num_nodes, num_edges=num_edges, num_edge_features=num_edge_features)
        
        edge_attr[~np.isfinite(edge_attr)] = null_value # Input protection
        edge_attr  = torch.tensor(edge_attr, dtype=torch.float)
        
        ## Construct edge connectivity
        edge_index = analytic.get_simple_edge_index(num_nodes=num_nodes, num_edges=num_edges, self_loops=self_loops, directed=directed)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        ## Construct global feature vector
        if global_on == False: # Null the global features
            u = torch.tensor([], dtype=torch.float)
        else:
            u_mat = np.zeros(len(features), dtype=float)
            for j in range(len(features)):
                xx = ak.to_numpy(X[ev][features[j]])
                if xx is not []: u_mat[j] = xx
            
            u_mat[~np.isfinite(u_mat)] = null_value # Input protection
            u = torch.tensor(u_mat, dtype=torch.float)

        dataset.append(Data(num_nodes=x.shape[0], x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, w=w, u=u))
    
    print(__name__ + f'.parse_graph_data: Empty events: {num_empty} / {num_events} = {num_empty/num_events:0.5f} (using only global data u)')        
    
    return dataset


def get_node_features(p4vec, fvec, num_nodes, num_node_features, node_features_hetero_ind, node_hetero_type, coord):

    # Node feature matrix
    x = np.zeros((num_nodes, np.sum([num_node_features[key] for key in num_node_features.keys()] )), dtype=float)

    for i in range(num_nodes):

        index = i-1

        # i = 0 case is the virtual node
        if i > 0:

            """
            # 4-momentum
            if   coord == 'ptetaphim':
                x[i,0] = p4vec[index].pt
                x[i,1] = p4vec[index].eta
                x[i,2] = p4vec[index].phi
                x[i,3] = p4vec[index].m
            elif coord == 'pxpypze':
                x[i,0] = p4vec[index].px
                x[i,1] = p4vec[index].py
                x[i,2] = p4vec[index].pz
                x[i,3] = p4vec[index].e
            else:
                raise Exception(__name__ + f'.get_node_features: Unknown coordinate representation')
            """

            # Other features
            key = node_hetero_type[index]
            x[i, node_features_hetero_ind[key]] = fvec[index]

    # Cast
    x = x.astype(float)

    return x
