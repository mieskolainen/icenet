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



def gram_matrix(X, type='dot'):
    """
    Gram matrix for 4-vectors.

    Args:
        X : Array of 4-vectors (N)
        type : Type of Lorentz scalars computed ('dot', 's', 't')

    Returns:
        G : Gram matrix (NxN)
    """

    N = len(X)
    G = np.zeros((N,N))
    for i in range(len(X)):
        for j in range(len(X)):
            if   type == 'dot':
                G[i,j] = X[i].dot(X[j])   ## 4-dot product
            elif type == 's':
                G[i,j] = (X[i] + X[j]).p2 ## s-type
            elif type == 't':
                G[i,j] = (X[i] - X[j]).p2 ## t-type
            else:
                raise Exception('gram_matrix: Unknown type!')

    return G


def parse_graph_data(X, VARS, features, Y=None, W=None, EPS=1e-12, global_on=True, coord='ptetaphim'):
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
    
    N_events = X.shape[0]
    dataset  = []

    print(__name__ + f'.parse_graph_data: Converting {N_events} events into graphs ...')
    zerovec = uproot_methods.TLorentzVector.from_ptetaphim(0,0,0,0)

    for e in tqdm(range(N_events)):

        num_nodes = 1 + len(X[e, VARS.index('image_clu_eta')]) # + 1 virtual node
        num_edges = num_nodes**2 # include self-connections

        num_node_features = 6
        num_edge_features = 3
        num_classes       = 2
        
        # Construct 4-vector for the track, with pion mass
        p4track = uproot_methods.TLorentzVector.from_ptetaphim(
            X[e, VARS.index('trk_pt')],
            X[e, VARS.index('trk_eta')],
            X[e, VARS.index('trk_phi')],
            0.13957)

        # Construct 4-vector for each ECAL cluster [@@ JAGGED @@]
        p4vec = []
        if len(X[e, VARS.index('image_clu_eta')]) > 0:
            p4vec = uproot_methods.TLorentzVectorArray.from_ptetaphim(
                X[e, VARS.index('image_clu_e')] / np.cosh(X[e, VARS.index('image_clu_eta')]),
                X[e, VARS.index('image_clu_eta')],
                X[e, VARS.index('image_clu_phi')], 0) # Massless

        # Construct Gram matrix
        if len(p4vec) > 0 and global_on:

            G1 = gram_matrix(p4vec, type='dot'); G1 /= (np.linalg.norm(G1) + 1e-9)
            G2 = gram_matrix(p4vec, type='s');   G2 /= (np.linalg.norm(G2) + 1e-9)
            G3 = gram_matrix(p4vec, type='t');   G3 /= (np.linalg.norm(G3) + 1e-9)

            # Determinant
            d1    = np.linalg.det(G1)
            d2    = np.linalg.det(G2)
            d3    = np.linalg.det(G3)
        else:
            d1    = 0
            d2    = 0
            d3    = 0  

        #print(f"N: {len(G1)} | det1: {d1:0.3E} | det2: {d2:0.3E} | det3: {d3:0.3E} | is_e: {X[e,VARS.index('is_e')]}")

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
        u = torch.tensor(np.zeros(len(features)) + 1 + 3, dtype=torch.float)
        for i in range(len(features)):
            u[i] = torch.tensor(X[e, VARS.index(features[i])], dtype=torch.float)

        # New global features
        # + 1
        if len(p4vec) > 0:
            u[-1] = p4vec.sum().p2 # Total invariant mass**2 of clusters
            u[-2] = d1
            u[-3] = d2
            u[-4] = d3
        
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

