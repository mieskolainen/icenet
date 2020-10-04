# Graph data readers and parsers for electron ID
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk


import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data

import uproot_methods

import icenet.algo.analytic as analytic



def parse_graph_data(X, VARS, features, Y=None, W=None, global_on=True, coord='ptetaphim', EPS=1e-12):
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

    num_node_features = 6
    num_edge_features = 4
    num_classes       = 2

    N_events = X.shape[0]
    dataset  = []

    print(__name__ + f'.parse_graph_data: Converting {N_events} events into graphs ...')
    zerovec = uproot_methods.TLorentzVector(0,0,0,0)

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
    for e in tqdm(range(N_events)):

        num_nodes = 1 + len(X[e, ind__image_clu_eta]) # + 1 virtual node
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
        # INITIALIZE TENSORS

        # Node feature matrix
        x = torch.tensor(np.zeros((num_nodes, num_node_features)), dtype=torch.float)

        # Graph connectivity: (~ sparse adjacency matrix)
        edge_index = torch.tensor(np.zeros((2, num_edges)), dtype=torch.long)

        # Edge features: [num_edges, num_edge_features]
        edge_attr  = torch.tensor(np.zeros((num_edges, num_edge_features)), dtype=torch.float)

        # Construct output class, note [] is important to have for right dimensions
        if Y is not None:
            y = torch.tensor([Y[e]], dtype=torch.long)
        else:
            y = torch.tensor([0], dtype=torch.long)

        # Training weights, note [] is important to have for right dimensions
        if W is not None:
            w = torch.tensor([W[e]], dtype=torch.float)
        else:
            w = torch.tensor([1.0], dtype=torch.float)

        # Construct global feature vector
        u = torch.tensor(X[e, feature_ind].tolist(), dtype=torch.float)
        

        # ====================================================================
        # CONSTRUCT TENSORS

        # ----------------------------------------------------------------
        # Construct node features
        for i in range(num_nodes):

            if i > 0:
                if   coord == 'ptetaphim':
                    x[i,0] = torch.tensor(p4vec[i-1].pt)
                    x[i,1] = torch.tensor(p4vec[i-1].eta)
                    x[i,2] = torch.tensor(p4vec[i-1].phi)
                    x[i,3] = torch.tensor(p4vec[i-1].mass)
                elif coord == 'pxpypze':
                    x[i,0] = torch.tensor(p4vec[i-1].x)
                    x[i,1] = torch.tensor(p4vec[i-1].y)
                    x[i,2] = torch.tensor(p4vec[i-1].z)
                    x[i,3] = torch.tensor(p4vec[i-1].t)
                else:
                    raise Exception(__name__ + f'parse_graph_data: Unknown coordinate representation')
                
                # other features
                x[i,4] = torch.tensor(X[e, VARS.index('image_clu_nhit')][i-1])
                x[i,5] = p4track.delta_r(p4vec[i-1])

        # ----------------------------------------------------------------
        ### Construct edge features
        n = 0
        for i in range(num_nodes):
            for j in range(num_nodes):

                if i > 0 and j > 0: # i,j = 0 case is the virtual node

                    p4_i   = p4vec[i-1]
                    p4_j   = p4vec[j-1]

                    # kt-metric (anti)
                    dR2_ij = ((p4_i.eta - p4_j.eta)**2 + (p4_i.phi - p4_j.phi)**2)
                    kt2_i  = p4_i.pt**2 + EPS 
                    kt2_j  = p4_j.pt**2 + EPS
                    edge_attr[n,0] = analytic.ktmetric(kt2_i=kt2_i, kt2_j=kt2_j, dR2_ij=dR2_ij, p=-1, R=1.0)
                    
                    # Lorentz scalars
                    edge_attr[n,1] = (p4_i + p4_j).p2  # Mandelstam s-like
                    edge_attr[n,2] = (p4_i - p4_j).p2  # Mandelstam t-like
                    edge_attr[n,3] = p4_i.dot(p4_j)    # 4-dot

                n += 1

        # ----------------------------------------------------------------
        ### Construct edge connectivity
        n = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                
                #if i > 0 and j > 0: # Skip virtual node

                # Full connectivity, including self-loops
                edge_index[0,n] = i
                edge_index[1,n] = j
            n += 1

        # Add this event
        if global_on == False: # Null the global features
            u = torch.tensor(np.zeros(len(u)), dtype=torch.float)

        dataset.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, w=w, u=u))

    return dataset


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

