# Graph data readers and parsers for HGCAL
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

import numpy as np
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


def parse_graph_data_trackster(data, weights=None, maxevents=int(1e9)):
    """
    TRACKSTER LEVEL
    
    Parse graph data to torch geometric format
    
    Args:
        data: awkward arra
    """

    event = 0
    print(data['x'][0])
    #print(data['edge_index'])
    #print(data['edge_labels'])

    nevents = len(data['x'])
    print(f'nevents = {nevents}')

    nevents = np.min([nevents, maxevents])

    graph_dataset = []

    ## For all events
    for ev in tqdm(range(nevents)):

        # --------------------------------------------
        ## ** Construct node features **
        nodes = data['x'][ev]

        x = np.zeros((len(nodes), 8))
        
        x[:,0] = nodes.barycenter_x.to_numpy()
        x[:,1] = nodes.barycenter_y.to_numpy()
        x[:,2] = nodes.barycenter_z.to_numpy()
        
        x[:,3] = nodes.raw_energy.to_numpy()
        x[:,4] = nodes.raw_em_energy.to_numpy()

        x[:,5] = nodes.EV1.to_numpy()
        x[:,6] = nodes.EV2.to_numpy()
        x[:,7] = nodes.EV3.to_numpy()

        x = torch.tensor(x, dtype=torch.float)
        # --------------------------------------------


        # --------------------------------------------
        ## ** Construct edge indices ** (TBD: how about self-connections and (un)-direction?)

        edge_index = data['edge_index'][ev]
        
        # size: 2 x num_edges
        edge_index = torch.tensor(np.array(edge_index, dtype=int), dtype=torch.long)
        # --------------------------------------------

        # --------------------------------------------
        ## ** Construct edge labels ** (training truth)

        y = data['edge_labels'][ev]

        # size: num_edges
        y = torch.tensor(np.array(y, dtype=int), dtype=torch.long)
        # --------------------------------------------
        
        # --------------------------------------------
        # Per edge training weights
        w = torch.ones_like(y,  dtype=torch.float)
        # --------------------------------------------
        
        # Global features
        u = torch.tensor([], dtype=torch.float)
        
        
        graph_dataset.append(Data(x=x, edge_index=edge_index, edge_attr=None, y=y, w=w, u=u))

    return graph_dataset


def parse_graph_data(X, ids, features, graph_param, Y=None, weights=None, maxevents=None, EPS=1e-12):
    """
    EVENT LEVEL (PROCESSING CANDIDATES)
    
    Jagged array data into pytorch-geometric style Data format array.
    
    Args:
        X          :  Jagged array of variables
        ids        :  Variable names as an array of strings
        features   :  Array of active global feature strings
        graph_param:  Graph construction parameters dict
        Y          :  Target class  array (if any, typically MC only)
        weights    :  (Re-)weighting array (if any, typically MC only)
    
    Returns:
        Array of pytorch-geometric Data objects
    """

    global_on = graph_param['global_on']
    coord     = graph_param['coord']
    
    num_node_features   = 4
    num_edge_features   = 4
    num_global_features = 0
    num_classes         = 2
    
    num_events = np.min([X.shape[0], maxevents]) if maxevents is not None else X.shape[0]
    dataset  = []
    
    print(__name__ + f'.parse_graph_data: Converting {num_events} events into graphs ...')
    zerovec = vec4()

    # Collect feature indices
    feature_ind = np.zeros(len(features), dtype=np.int32)
    for i in range(len(features)):
        feature_ind[i] = ids.index(features[i])

    # Collect track indices
    #ind__trk_pt        = ids.index('trk_pt')
    #ind__trk_eta       = ids.index('trk_eta')
    #ind__trk_phi       = ids.index('trk_phi')
    
    ind__candidate_energy = ids.index('candidate_energy')
    ind__candidate_px     = ids.index('candidate_px')
    ind__candidate_py     = ids.index('candidate_py')
    ind__candidate_pz     = ids.index('candidate_pz')

    num_empty_HGCAL = 0

    # Loop over events
    for i in tqdm(range(num_events)):

        num_nodes = 1 + len(X[i, ind__candidate_energy]) # + 1 virtual node
        num_edges = num_nodes**2                         # include self-connections
        
        # Construct 4-vector for the track, with pion mass
        #p4track = vec4()
        #p4track.setPtEtaPhiM(X[i, ind__trk_pt], X[i, ind__trk_eta], X[i, ind__trk_phi], 0.13957)

        # Construct 4-vector for each HGCAL candidate [@@ JAGGED @@]
        p4vec = []
        N_c = len(X[i, ind__candidate_energy])
        
        if N_c > 0:
            for k in range(N_c): 
                
                energy = X[i, ind__candidate_energy][k]
                px     = X[i, ind__candidate_px][k]
                py     = X[i, ind__candidate_py][k]
                pz     = X[i, ind__candidate_pz][k]
                
                v = vec4()
                v.setPxPyPzE(px, py, pz, energy)

                p4vec.append( v )
        
        # Empty HGCAL cluster information
        else:
            num_empty_HGCAL += 1
            # However, never skip empty HGCAL cluster events here!!, do pre-filtering before this function if needed
        
        # ====================================================================
        # CONSTRUCT TENSORS

        # Construct output class, note [] is important to have for right dimensions
        if Y is not None:
            y = torch.tensor([Y[i]], dtype=torch.long)
        else:
            y = torch.tensor([0],    dtype=torch.long)

        # Training weights, note [] is important to have for right dimensions
        if weights is not None:
            w = torch.tensor([weights[i]], dtype=torch.float)
        else:
            w = torch.tensor([1.0],  dtype=torch.float)

        ## Construct global feature vector
        #u = torch.tensor(X[i, feature_ind].tolist(), dtype=torch.float)
        
        ## Construct node features
        x = get_node_features(p4vec=p4vec, p4track=None, X=X[i], ids=ids, num_nodes=num_nodes, num_node_features=num_node_features, coord=coord)
        x = torch.tensor(x, dtype=torch.float)

        ## Construct edge features
        edge_attr  = analytic.get_Lorentz_edge_features(p4vec=p4vec, num_nodes=num_nodes, num_edges=num_edges, num_edge_features=num_edge_features)
        edge_attr  = torch.tensor(edge_attr, dtype=torch.float)
        
        ## Construct edge connectivity
        edge_index = get_edge_index(num_nodes=num_nodes, num_edges=num_edges)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Add this event
        if global_on == False: # Null the global features
            u = torch.tensor(np.zeros(num_global_features), dtype=torch.float)
        
        dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, w=w, u=u))
    
    print(__name__ + f'.parse_graph_data: Empty HGCAL cluster events: {num_empty_HGCAL} / {num_events} = {num_empty_HGCAL/num_events:0.5f} (using only global data u)')        
    
    return dataset



def get_node_features(p4vec, p4track, X, ids, num_nodes, num_node_features, coord):

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
                raise Exception(__name__ + f'parse_graph_data: Unknown coordinate representation')
            
            """
            # other features
            x[i,4] = p4track.deltaR(p4vec[i-1])

            try:
                x[i,5] = X[ids.index('candidate_nhit')][i-1]
            except:
                continue
                # Not able to read it (empty cluster data)
            """
            
    return x


@numba.njit
def get_edge_index(num_nodes, num_edges):

    # Graph connectivity: (~ sparse adjacency matrix)
    edge_index = np.zeros((2, num_edges))

    n = 0
    for i in range(num_nodes):
        for j in range(num_nodes):

            # Full connectivity
            edge_index[0,n] = i
            edge_index[1,n] = j
            n += 1
    
    return edge_index

