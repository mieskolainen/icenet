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


def parse_graph_data(X, VARS, features, Y=None, W=None, EPS=1e-12):
    """
    Jagged array data into pytorch-geometric style Data format array.

    Args:
        X        :  Jagged array of variables
        VARS     :  Variable names as an array of strings
        features :  Array of active scalar feature strings
        Y        :  Target class  array (if any, typically MC only)
        W        :  (Re-)weighting array (if any, typically MC only)
        
    Returns:
        Array of pytorch-geometric Data objects
    """
    
    N_events = X.shape[0]
    dataset  = []

    print(__name__ + f'.parse_graph_data: Converting {N_events} events into graphs ...')

    zerovec = uproot_methods.TLorentzVector.from_ptetaphim(0,0,0,0)

    for e in tqdm(range(N_events)):

        num_nodes = 1 + len(X[e, VARS.index('image_clu_eta')]) # + 1 virtual node
        num_edges = num_nodes**2 # include self-connections

        num_node_features = 7
        num_edge_features = 3
        num_classes       = 2
        
        # Construct 4-vector for the track, with zero-mass
        p4track = uproot_methods.TLorentzVector.from_ptetaphim(
            X[e, VARS.index('trk_pt')],
            X[e, VARS.index('trk_eta')],
            X[e, VARS.index('trk_phi')],
            0)

        # Construct 4-vector for each cluster [@@ JAGGED @@]
        p4vec = []
        if len(X[e, VARS.index('image_clu_eta')]) > 0:
            p4vec = uproot_methods.TLorentzVectorArray.from_ptetaphim(
                X[e, VARS.index('image_clu_e')] / np.cosh(X[e, VARS.index('image_clu_eta')]),
                X[e, VARS.index('image_clu_eta')],
                X[e, VARS.index('image_clu_phi')], 0) # Set photon mass

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
        u = torch.tensor(np.zeros(len(features)) + 1, dtype=torch.float)
        for i in range(len(features)):
            u[i] = torch.tensor(X[e, VARS.index(features[i])], dtype=torch.float)

        # New global features
        # + 1
        if len(p4vec) > 0:
            u[-1] = p4vec.sum().p2 # Total invariant mass**2 of clusters

        # ====================================================================
        # CONSTRUCT TENSORS

        # ----------------------------------------------------------------
        # Construct node features
        for i in range(num_nodes):

            if i > 0:
                # Features spesific to each node
                x[i,0] = torch.tensor(X[e, VARS.index('image_clu_eta')][i-1])
                x[i,1] = torch.tensor(X[e, VARS.index('image_clu_phi')][i-1])
                x[i,2] = torch.tensor(X[e, VARS.index('image_clu_e')][i-1])
                x[i,3] = torch.tensor(X[e, VARS.index('image_clu_nhit')][i-1])

                # Relative coordinates
                diff   = p4track - p4vec[i-1]
                x[i,4] = diff.x
                x[i,5] = diff.y
                x[i,6] = diff.z

        # ----------------------------------------------------------------
        ### Construct edge features
        n = 0
        for i in range(num_nodes):
            for j in range(num_nodes):

                if i > 0 and j > 0: # i,j = 0 case is the virtual node
                    p4_i = p4vec[i-1]
                    p4_j = p4vec[j-1]
                else:
                    p4_i = zerovec
                    p4_j = zerovec

                # kt-metric (anti)
                dR2_ij = ((p4_i.eta - p4_j.eta)**2 + (p4_i.phi - p4_j.phi)**2)
                kt2_i  = p4_i.pt**2 + EPS 
                kt2_j  = p4_j.pt**2 + EPS
                edge_attr[n,0] = analytic.ktmetric(kt2_i=kt2_i, kt2_j=kt2_j, dR2_ij=dR2_ij, p=-1, R=1.0)
                
                # Lorentz scalars
                edge_attr[n,1] = (p4_i + p4_j).p2  # Mandelstam s-like
                edge_attr[n,2] = (p4_i - p4_j).p2  # Mandelstam t-like

                n += 1

        # ----------------------------------------------------------------
        ### Normalize edge features to be in interval [0,1] (some algorithms require it)
        ### Improve this part!!
        '''
        for p in range(num_edge_features):
            maxvalue = torch.max(edge_attr[:,p]) + EPS
            edge_attr[:,p] /= maxvalue
        '''

        # ----------------------------------------------------------------
        ### Construct edge connectivity
        n = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                
                # Full connectivity, including self-loops
                edge_index[0,n] = i
                edge_index[1,n] = j
                n += 1

        # Add this event
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

