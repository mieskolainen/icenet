# Graph data readers and parsers for electron ID
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import numpy as np
import numba
from tqdm import tqdm
import copy
import pickle
import os
import uuid

import torch
from torch_geometric.data import Data
import multiprocessing

import uproot_methods
import icenet.algo.analytic as analytic

from icenet.tools import aux


from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler

default_collate_func = dataloader.default_collate



data = []

def parse_graph_data(X, VARS, features, Y=None, W=None, global_on=True, coord='ptetaphim', CPU_count=None):
    """
    Jagged array data into pytorch-geometric style Data format array.
    
    Args:
        X         :  Jagged array of variables
        VARS      :  Variable names as an array of strings
        features  :  Array of active scalar feature strings
        Y         :  Target class  array (if any, typically MC only)
        W         :  (Re-)weighting array (if any, typically MC only)
        global_on :  Global features on / off
        coord     :  Coordinates used for nodes ('ptetaphim', 'pxpypze')
        
    Returns:
        Array of pytorch-geometric Data objects
    """
    
    os.makedirs("./tmp", exist_ok = True)
    
    if CPU_count is None:
        CPU_count = int(np.ceil(multiprocessing.cpu_count()/2))

    # Compute indices
    N_events  = X.shape[0]
    print(__name__ + f'.parse_graph_data: Converting {N_events} events into graphs with {CPU_count} CPU processes ...')

    if N_events <= 256:
        CPU_count = 1

    # Get indices per thread
    block_ind = aux.split_start_end(range(N_events), CPU_count)
    print(block_ind)

    global data
    data = {
        'X'         : X,
        'VARS'      : VARS,
        'features'  : features,
        'Y'         : Y,
        'W'         : W,
        'global_on' : global_on,
        'coord'     : coord,
        'UUID'      : uuid.uuid1()
    }

    # Parallel processing (crashes with torch outputs, so use numpy only!)
    pool   = multiprocessing.Pool(CPU_count)
    output = pool.map(innerwrap, block_ind)

    # Fuse results from the processes
    result = []
    for i in range(CPU_count):

        with open(f"./tmp/graph-dump_{block_ind[i][0]}_{block_ind[i][1]}_{data['UUID']}.pkl", 'rb') as handle:
            dataset = pickle.load(handle)

        # Turn into torch geometric Data object
        dd = []
        for e in range(len(dataset)):

            d = dataset[e]
            dd.append( Data(x=torch.tensor(d['x'], dtype=torch.float),
                            edge_index=torch.tensor(d['edge_index'], dtype=torch.long),
                            edge_attr =torch.tensor(d['edge_attr'],  dtype=torch.float),
                            y=torch.tensor(d['y'], dtype=torch.long),
                            w=torch.tensor(d['w'], dtype=torch.float),
                            u=torch.tensor(d['u'], dtype=torch.float)))

        result = result + dd

    # Remove tmp files
    for i in range(CPU_count):
        os.system(f"rm ./tmp/graph-dump_{block_ind[i][0]}_{block_ind[i][1]}_{data['UUID']}.pkl")

    return result


def innerwrap(block_ind, EPS=1e-12):

    global data

    X         = data['X']
    VARS      = data['VARS']
    features  = data['features']
    Y         = data['Y']
    W         = data['W']
    global_on = data['global_on']
    coord     = data['coord']

    # -------------------------------------------

    num_node_features = 6
    num_edge_features = 4
    num_classes       = 2

    N_events = X.shape[0]
    dataset  = []
    zerovec  = uproot_methods.TLorentzVector(0,0,0,0)

    # Collect feature indices
    feature_ind = np.zeros(len(features), dtype=np.int32)
    for i in range(len(features)):
        feature_ind[i] = VARS.index(features[i])


    # Collect indices
    ind__trk_pt        = VARS.index('trk_pt')
    ind__trk_eta       = VARS.index('trk_eta')
    ind__trk_phi       = VARS.index('trk_phi')

    ind__image_clu_e   = VARS.index('image_clu_e')
    ind__image_clu_eta = VARS.index('image_clu_eta')
    ind__image_clu_phi = VARS.index('image_clu_phi')


    # Loop over events
    for e in tqdm(range(block_ind[0], block_ind[1]+1)): # Note +1

        num_nodes = 1 + len(X[e, ind__image_clu_eta]) # +1 virtual node
        num_edges = num_nodes**2 # include self-connections
        
        # Construct 4-vector for the track, with pion mass
        p4track = \
            uproot_methods.TLorentzVector.from_ptetaphim(
                X[e, ind__trk_pt], X[e, ind__trk_eta], X[e, ind__trk_phi], 0.13957)

        # Construct 4-vector for each ECAL cluster [@@ JAGGED @@]
        p4vec = []
        if len(X[e, ind__image_clu_e]) > 0:
            pt    = X[e, ind__image_clu_e] / np.cosh(X[e, ind__image_clu_eta]) # Massless approx.
            p4vec = uproot_methods.TLorentzVectorArray.from_ptetaphim(
                pt, X[e, ind__image_clu_eta], X[e, ind__image_clu_phi], 0) # Massless


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
        x = get_node_features(p4vec=p4vec, p4track=p4track, X=X[e], VARS=VARS, num_nodes=num_nodes, num_node_features=num_node_features, coord=coord)
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

        dataset.append({'x':x, 'edge_index':edge_index, 'edge_attr':edge_attr, 'y':y, 'w':w, 'u':u})
        #dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, w=w, u=u))

    # Save to disk
    with open(f"./tmp/graph-dump_{block_ind[0]}_{block_ind[1]}_{data['UUID']}.pkl", 'wb') as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return True



def get_node_features(p4vec, p4track, X, VARS, num_nodes, num_node_features, coord):

    # Node feature matrix
    x = np.zeros((num_nodes, num_node_features))

    for i in range(num_nodes):

        # i = 0 case is the virtual node
        if i > 0:
            if   coord == 'ptetaphim':
                x[i,0] = p4vec[i-1].pt
                x[i,1] = p4vec[i-1].eta
                x[i,2] = p4vec[i-1].phi
                x[i,3] = p4vec[i-1].mass
            elif coord == 'pxpypze':
                x[i,0] = p4vec[i-1].x
                x[i,1] = p4vec[i-1].y
                x[i,2] = p4vec[i-1].z
                x[i,3] = p4vec[i-1].t
            else:
                raise Exception(__name__ + f'parse_graph_data: Unknown coordinate representation')
            
            # other features
            x[i,4] = X[VARS.index('image_clu_nhit')][i-1]
            x[i,5] = p4track.delta_r(p4vec[i-1])

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
                dR2_ij = ((p4_i.eta - p4_j.eta)**2 + (p4_i.phi - p4_j.phi)**2)
                kt2_i  = p4_i.pt2 + EPS 
                kt2_j  = p4_j.pt2 + EPS
                edge_attr[n,0] = analytic.ktmetric(kt2_i=kt2_i, kt2_j=kt2_j, dR2_ij=dR2_ij, p=-1, R=1.0)
                
                # Lorentz scalars
                edge_attr[n,1] = (p4_i + p4_j).p2  # Mandelstam s-like
                edge_attr[n,2] = (p4_i - p4_j).p2  # Mandelstam t-like
                edge_attr[n,3] = p4_i.dot(p4_j)    # 4-dot

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

