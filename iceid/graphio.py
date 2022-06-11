# Graph data readers and parsers for electron ID
#
# Mikael Mieskolainen, 2021
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


# Torch conversion
def graph2torch(X):

    # Turn into torch geometric Data object
    Y = np.zeros(len(X), dtype=object)
    for i in range(len(X)):

        d = X[i]
        Y[i] = Data(x=torch.tensor(d['x'], dtype=torch.float),
                    edge_index=torch.tensor(d['edge_index'], dtype=torch.long),
                    edge_attr =torch.tensor(d['edge_attr'],  dtype=torch.float),
                    y=torch.tensor(d['y'], dtype=torch.long),
                    w=torch.tensor(d['w'], dtype=torch.float),
                    u=torch.tensor(d['u'], dtype=torch.float))
    return Y


def parse_tensor_data(X, ids, image_vars, args):
    """
    Args:
        X     :  Jagged array of variables
        ids  :  Variable names as an array of strings
        args  :  Arguments
    
    Returns:
        Tensor of pytorch-geometric Data objects
    """

    newind  = np.where(np.isin(ids, image_vars))
    newind  = np.array(newind).flatten()
    newvars = []
    for i in newind :
        newvars.append(ids[i])

    # Pick image data
    X_image = X[:, newind]

    # Use single channel tensors
    if   args['image_param']['channels'] == 1:
        xyz = [['image_clu_eta', 'image_clu_phi', 'image_clu_e']]

    # Use multichannel tensors
    elif args['image_param']['channels'] == 2:
        xyz = [['image_clu_eta', 'image_clu_phi', 'image_clu_e'], 
               ['image_pf_eta',  'image_pf_phi',  'image_pf_p']]
    else:
        raise Except(__name__ + f'.splitfactor: Unknown [image_param][channels] parameter')

    eta_binedges = args['image_param']['eta_bins']
    phi_binedges = args['image_param']['phi_bins']    

    # Pick tensor data out
    cprint(__name__ + f'.splitfactor: jagged2tensor processing ...', 'yellow')
    tensor = aux.jagged2tensor(X=X_image, ids=newvars, xyz=xyz, x_binedges=eta_binedges, y_binedges=phi_binedges)

    return tensor


def parse_graph_data_np(X, ids, features, Y=None, W=None, global_on=True, coord='ptetaphim'):

    """
    Args:
        X         :  Jagged array of variables
        ids      :  Variable names as an array of strings
        features  :  Array of active scalar feature strings
        Y         :  Target class  array (if any, typically MC only)
        W         :  (Re-)weighting array (if any, typically MC only)
        global_on :  Global features on / off
        coord     :  Coordinates used for nodes ('ptetaphim', 'pxpypze')
        
    Returns:
        Array of pytorch-geometric Data objects
    """

    # -------------------------------------------

    num_node_features = 6
    num_edge_features = 4
    num_classes       = 2

    N_events = X.shape[0]
    dataset  = []
    zerovec  = vec4()


    # Collect feature indices
    feature_ind = np.zeros(len(features), dtype=np.int32)
    for i in range(len(features)):
        feature_ind[i] = ids.index(features[i])


    # Collect indices
    ind__trk_pt        = ids.index('trk_pt')
    ind__trk_eta       = ids.index('trk_eta')
    ind__trk_phi       = ids.index('trk_phi')

    ind__image_clu_e   = ids.index('image_clu_e')
    ind__image_clu_eta = ids.index('image_clu_eta')
    ind__image_clu_phi = ids.index('image_clu_phi')


    # Loop over events
    for e in tqdm(range(N_events)):

        num_nodes = 1 + len(X[e, ind__image_clu_eta]) # +1 virtual node
        num_edges = num_nodes**2 # include self-connections
        
        # Construct 4-vector for the track, with pion mass
        p4track = vec4.setPtEtaPhiM(X[e, ind__trk_pt], X[e, ind__trk_eta], X[e, ind__trk_phi], 0.13957)

        # Construct 4-vector for each ECAL cluster [@@ JAGGED @@]
        p4vec = []
        if len(X[e, ind__image_clu_e]) > 0:
            pt    = X[e, ind__image_clu_e] / np.cosh(X[e, ind__image_clu_eta]) # Massless approx.
            p4vec = vec4.setPtEtaPhiM(pt, X[e, ind__image_clu_eta], X[e, ind__image_clu_phi], 0) # Massless


        # ====================================================================
        # CONSTRUCT TENSORS

        # Construct output class, note [] is important to have for right dimensions
        y = [Y[e]] if Y is not None else [0]

        # Training weights, note [] is important to have for right dimensions
        w = [W[e]] if W is not None else [1.0]

        ###y = torch.tensor(y, dtype=torch.long)
        ###w = torch.tensor(w, dtype=torch.float)

        ## Construct global feature vector
        u = X[e, feature_ind].tolist()
        ###u = torch.tensor(u, dtype=torch.float)
        
        ## Construct node features
        x = get_node_features(p4vec=p4vec, p4track=p4track, X=X[e], ids=ids, num_nodes=num_nodes, num_node_features=num_node_features, coord=coord)
        ###x = torch.tensor(x, dtype=torch.float)

        ## Construct edge features
        edge_attr  = get_edge_features(p4vec=p4vec, num_nodes=num_nodes, num_edges=num_edges, num_edge_features=num_edge_features)
        ###edge_attr  = torch.tensor(edge_attr, dtype=torch.float)

        ## Construct edge connectivity
        edge_index = get_edge_index(num_nodes=num_nodes, num_edges=num_edges)
        ###edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Add this event
        if global_on == False: # Null the global features
            u = np.zeros(len(u))
            ###u = torch.tensor(np.zeros(len(u)), dtype=torch.float)

        # Pure dictionary
        dataset.append({'x':x, 'edge_index':edge_index, 'edge_attr':edge_attr, 'y':y, 'w':w, 'u':u})
        #dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, w=w, u=u))

    return dataset


def parse_graph_data(X, ids, features, Y=None, weights=None, global_on=True, coord='ptetaphim', EPS=1e-12):
    """
    Jagged array data into pytorch-geometric style Data format array.
    
    Args:
        X         :  Jagged array of variables
        ids       :  Variable names as an array of strings
        features  :  Array of active scalar feature strings
        Y         :  Target class  array (if any, typically MC only)
        weights   :  (Re-)weighting array (if any, typically MC only)
        global_on :  Global features on / off
        coord     :  Coordinates used for nodes ('ptetaphim', 'pxpypze')
        
    Returns:
        Array of pytorch-geometric Data objects
    """
    
    num_node_features = 6
    num_edge_features = 4
    num_classes       = 2

    N_events = X.shape[0]
    dataset  = []

    print(__name__ + f'.parse_graph_data: Converting {N_events} events into graphs ...')
    zerovec = vec4()

    # Collect feature indices
    feature_ind = np.zeros(len(features), dtype=np.int32)
    for i in range(len(features)):
        feature_ind[i] = ids.index(features[i])


    # Collect indices
    ind__trk_pt        = ids.index('trk_pt')
    ind__trk_eta       = ids.index('trk_eta')
    ind__trk_phi       = ids.index('trk_phi')

    ind__image_clu_e   = ids.index('image_clu_e')
    ind__image_clu_eta = ids.index('image_clu_eta')
    ind__image_clu_phi = ids.index('image_clu_phi')


    num_empty_ECAL = 0

    # Loop over events
    for i in tqdm(range(N_events)):

        num_nodes = 1 + len(X[i, ind__image_clu_eta]) # + 1 virtual node
        num_edges = num_nodes**2                      # include self-connections
        
        # Construct 4-vector for the track, with pion mass
        p4track = vec4()
        p4track.setPtEtaPhiM(X[i, ind__trk_pt], X[i, ind__trk_eta], X[i, ind__trk_phi], 0.13957)

        # Construct 4-vector for each ECAL cluster [@@ JAGGED @@]
        p4vec = []
        N_c = len(X[i, ind__image_clu_e])
        
        if N_c > 0:
            for k in range(N_c): 
                
                pt    = X[i, ind__image_clu_e][k] / np.cosh(X[i, ind__image_clu_eta][k]) # Massless approx.
                eta   = X[i, ind__image_clu_eta][k]
                phi   = X[i, ind__image_clu_phi][k]

                v = vec4()
                v.setPtEtaPhiM(pt, eta, phi, 0)

                p4vec.append( v )
        
        # Empty ECAL cluster information
        else:
            num_empty_ECAL += 1
            # However, never skip empty ECAL cluster events here!!, do pre-filtering before this function if needed
        
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
        u = torch.tensor(X[i, feature_ind].tolist(), dtype=torch.float)
        
        ## Construct node features
        x = get_node_features(p4vec=p4vec, p4track=p4track, X=X[i], ids=ids, num_nodes=num_nodes, num_node_features=num_node_features, coord=coord)
        x = torch.tensor(x, dtype=torch.float)

        ## Construct edge features
        edge_attr  = get_edge_features(p4vec=p4vec, num_nodes=num_nodes, num_edges=num_edges, num_edge_features=num_edge_features)
        edge_attr  = torch.tensor(edge_attr, dtype=torch.float)

        ## Construct edge connectivity
        edge_index = get_edge_index(num_nodes=num_nodes, num_edges=num_edges)
        edge_index = torch.tensor(edge_index, dtype=torch.long)


        # Add this event
        if global_on == False: # Null the global features
            u = torch.tensor(np.zeros(len(u)), dtype=torch.float)

        dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, w=w, u=u))
    
    print(__name__ + f'.parse_graph_data: Empty ECAL cluster events: {num_empty_ECAL} / {N_events} = {num_empty_ECAL/N_events:0.5f} (using only global data u)')        
    
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
            
            # other features
            x[i,4] = p4track.deltaR(p4vec[i-1])

            try:
                x[i,5] = X[ids.index('image_clu_nhit')][i-1]
            except:
                continue
                # Not able to read it (empty cluster data)
            
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


def get_edge_features(p4vec, num_nodes, num_edges, num_edge_features, EPS=1E-12):

    # Edge features: [num_edges, num_edge_features]
    edge_attr = np.zeros((num_edges, num_edge_features), dtype=float)
    indexlist = np.zeros((num_nodes, num_nodes), dtype=int)
    
    n = 0
    for i in range(num_nodes):
        for j in range(num_nodes):

            # Compute only non-zero
            if (i > 0 and j > 0) and (j > i):

                p4_i   = p4vec[i-1]
                p4_j   = p4vec[j-1]

                # kt-metric (anti)
                dR2_ij = p4_i.deltaR(p4_j)**2
                kt2_i  = p4_i.pt2 + EPS 
                kt2_j  = p4_j.pt2 + EPS
                edge_attr[n,0] = analytic.ktmetric(kt2_i=kt2_i, kt2_j=kt2_j, dR2_ij=dR2_ij, p=-1, R=1.0)
                
                # Lorentz scalars
                edge_attr[n,1] = (p4_i + p4_j).m2  # Mandelstam s-like
                edge_attr[n,2] = (p4_i - p4_j).m2  # Mandelstam t-like
                edge_attr[n,3] = p4_i.dot4(p4_j)     # 4-dot

            indexlist[i,j] = n
            n += 1

    ### Copy to the lower triangle for speed (we have symmetric adjacency)
    n = 0
    for i in range(num_nodes):
        for j in range(num_nodes):

            # Copy only non-zero
            if (i > 0 and j > 0) and (j < i):
                edge_attr[n,:] = edge_attr[indexlist[j,i],:] # note [j,i] !
            n += 1

    return edge_attr


'''
def find_k_nearest(edge_attr, num_nodes, k=5):
    """
    Find fixed k-nearest neighbours, return corresponding edge connectivities
    """

    # Loop over each node
    for i in range(num_nodes):

        # Loop over each other node, take distances
        for j in range(num_nodes):

    # Graph connectivity: (~ adjacency matrix)
    edge_index = torch.tensor(np.zeros((2, num_edges)), dtype=torch.long)
'''

