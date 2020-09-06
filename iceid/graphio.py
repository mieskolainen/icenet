# Graph data readers and parsers for electron ID
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk


import numpy as np
from tqdm import tqdm

import torch
from torch_geometric.data import Data

import icenet.algo.analytic as analytic


def parse_graph_data(X, VARS, features, Y=None, W=None):
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

    for e in tqdm(range(N_events)):

        num_nodes = 1 + len(X[:, VARS.index('image_clu_eta')][e]) # + 1 virtual node
        num_edges = num_nodes**2 # include self-connections

        num_node_features = 3 + len(features)
        num_edge_features = 1
        num_classes       = 2

        # ====================================================================
        # INITIALIZE TENSORS

        # Node feature matrix
        x = torch.tensor(np.zeros((num_nodes, num_node_features)), dtype=torch.float)

        # Graph connectivity: (~ adjacency matrix)
        edge_index = torch.tensor(np.zeros((2, num_edges)), dtype=torch.long)

        # Edge features: [num_edges, num_edge_features]
        edge_attr  = torch.tensor(np.zeros((num_edges, num_edge_features)), dtype=torch.float)

        # Construct output class, note [] is important to have for right dimensions
        if Y is not None:
            y = torch.tensor([Y[e]], dtype=torch.long)
        else:
            y = torch.tensor([0],    dtype=torch.long)

        # Training weights, note [] is important to have for right dimensions
        if W is not None:
            w = torch.tensor([W[e]], dtype=float)
        else:
            w = torch.tensor([1.0],  dtype=float)

        # Construct global feature vector
        u = torch.tensor([0], dtype=float)

        # ====================================================================
        # CONSTRUCT TENSORS

        # Construct node features
        for i in range(num_nodes):

            # Hand-crafted features replicated to all nodes (vertices)
            # (RAM wasteful, think about alternative strategies)
            k = 0
            for name in features:
                x[i,k] = torch.tensor(X[:, VARS.index(name)][e])
                k += 1

            if i > 0: # image features spesific to each node

                x[i,k]   = torch.tensor(X[:, VARS.index('image_clu_eta')][e][i-1])
                x[i,k+1] = torch.tensor(X[:, VARS.index('image_clu_phi')][e][i-1])
                x[i,k+2] = torch.tensor(X[:, VARS.index('image_clu_e')][e][i-1])

        ### Construct edge features
        n = 0
        for i in range(num_nodes):
            for j in range(num_nodes):

                if i > 0 and j > 0: # i,j = 0 case is the virtual node

                    # L2-distance squared in (eta,phi) plane
                    dR2_ij = (X[:, VARS.index('image_clu_eta')][e][i-1] - X[:, VARS.index('image_clu_eta')][e][j-1])**2 + \
                             (X[:, VARS.index('image_clu_phi')][e][i-1] - X[:, VARS.index('image_clu_phi')][e][j-1])**2

                    # E ~= pt * cosh(eta)
                    kt2_i = (X[:, VARS.index('image_clu_e')][e][i-1] / np.cosh(X[:, VARS.index('image_clu_eta')][e][i-1]))**2
                    kt2_j = (X[:, VARS.index('image_clu_e')][e][j-1] / np.cosh(X[:, VARS.index('image_clu_eta')][e][j-1]))**2
                else:
                    dR2_ij = 0
                    kt2_i  = 1e-12
                    kt2_j  = 1e-12

                # Add kt-metric like edge features, with p = -1,0,1 (for now only with p=-1)
                for p in range(1):
                    edge_attr[n,p] = analytic.ktmetric(kt2_i=kt2_i, kt2_j=kt2_j, dR2_ij=dR2_ij, p=-1+p, R=1.0)

                n += 1

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
