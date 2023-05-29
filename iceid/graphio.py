# Graph and tensor data readers and parsers for electron ID
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

import numpy as np
from   tqdm import tqdm

from termcolor import colored, cprint

import torch
from   torch_geometric.data import Data

import icenet.algo.analytic as analytic
from   icenet.tools import aux
from   icenet.tools.icevec import vec4


# # Torch conversion
# def graph2torch(X):

#     # Turn into torch geometric Data object
#     Y = np.zeros(len(X), dtype=object)
#     for i in range(len(X)):
        
#         d = X[i]
#         Y[i] = Data(x=torch.tensor(d['x'], dtype=torch.float),
#                     edge_index=torch.tensor(d['edge_index'], dtype=torch.long),
#                     edge_attr =torch.tensor(d['edge_attr'],  dtype=torch.float),
#                     y=torch.tensor(d['y'], dtype=torch.long),
#                     w=torch.tensor(d['w'], dtype=torch.float),
#                     u=torch.tensor(d['u'], dtype=torch.float))
#     return Y


def parse_tensor_data(X, ids, image_vars, args):
    """
    Args:
        X     :  Jagged array of variables
        ids   :  Variable names as an array of strings
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


def parse_graph_data(X, ids, features, graph_param, Y=None, weights=None, entry_start=None, entry_stop=None, EPS=1e-12, null_value=-999.0):
    """
    Jagged array data into pytorch-geometric style Data format array.
    
    Args:
        X          :  Jagged array of variables
        ids        :  Variable names as an array of strings
        features   :  List of active global feature strings
        graph_param:  Graph construction parameters dict
        Y          :  Target class  array (if any, typically MC only)
        weights    :  (Re-)weighting array (if any, typically MC only)
    
    Returns:
        List of pytorch-geometric Data objects
    """
    M_PION     = 0.13957 # charged pion mass (GeV)
    
    global_on  = graph_param['global_on']
    coord      = graph_param['coord']
    directed   = graph_param['directed']
    self_loops = graph_param['self_loops']

    # ----------------------------------------------
    
    num_node_features = 6
    num_edge_features = 4
    
    entry_start, entry_stop, num_events = aux.slice_range(start=entry_start, stop=entry_stop, N=len(X))
    dataset = []
    
    print(__name__ + f'.parse_graph_data: Converting {num_events} events into graphs ...')
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
    for ev in tqdm(range(entry_start, entry_stop)):

        num_nodes = 1 + len(X[ev, ind__image_clu_eta]) # +1 for the virtual node (empty data)
        num_edges = analytic.count_simple_edges(num_nodes=num_nodes, directed=directed, self_loops=self_loops)

        # Construct 4-vector for the track, with pion mass
        p4track = vec4()
        p4track.setPtEtaPhiM(X[ev, ind__trk_pt], X[ev, ind__trk_eta], X[ev, ind__trk_phi], M_PION)

        # Construct 4-vector for each ECAL cluster
        p4vec = []
        N_c = len(X[ev, ind__image_clu_e])
        
        if N_c > 0:
            for k in range(N_c): 
                
                pt    = X[ev, ind__image_clu_e][k] / np.cosh(X[ev, ind__image_clu_eta][k]) # Massless approx.
                eta   = X[ev, ind__image_clu_eta][k]
                phi   = X[ev, ind__image_clu_phi][k]

                v = vec4()
                v.setPtEtaPhiM(pt, eta, phi, 0)
                
                p4vec.append(v)

        # Empty ECAL cluster information
        else:
            num_empty_ECAL += 1
            # However, never skip empty ECAL cluster events here!!, do pre-filtering before this function if needed
        
        p4vec.append(vec4()) # Add empty 4-vector (for all events, due to empty events)

        # ====================================================================
        # CONSTRUCT TENSORS

        # Construct output class, note [] is important to have for right dimensions
        if Y is None:
            y = torch.tensor([0],     dtype=torch.long)            
        else:
            y = torch.tensor([Y[ev]], dtype=torch.long)
        
        # Training weights, note [] is important to have for right dimensions
        if weights is None:
            w = torch.tensor([1.0],  dtype=torch.float)    
        else:
            w = torch.tensor([weights[ev]], dtype=torch.float)

        ## Construct node features
        x = get_node_features(p4vec=p4vec, p4track=p4track, X=X[ev], ids=ids, num_nodes=num_nodes, num_node_features=num_node_features, coord=coord)
        
        x[~np.isfinite(x)] = null_value # Input protection
        x = torch.tensor(x, dtype=torch.float)

        ## Construct edge features
        edge_attr  = analytic.get_Lorentz_edge_features(p4vec=p4vec, num_nodes=num_nodes, \
            num_edges=num_edges, num_edge_features=num_edge_features, self_loops=self_loops, directed=directed)
        
        edge_attr[~np.isfinite(edge_attr)] = null_value # Input protection
        edge_attr  = torch.tensor(edge_attr, dtype=torch.float)

        ## Construct edge connectivity
        edge_index = analytic.get_simple_edge_index(num_nodes=num_nodes, num_edges=num_edges, self_loops=self_loops, directed=directed)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        ## Construct global feature vector
        if global_on == False: # Null the global features
            u = torch.tensor([], dtype=torch.float)
        else:
            u = (X[ev, feature_ind]).astype(float)
            u[~np.isfinite(u)] = null_value # Input protection
            u = torch.tensor(u, dtype=torch.float)
        
        dataset.append(Data(num_nodes=x.shape[0], x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, w=w, u=u))
    
    print(__name__ + f'.parse_graph_data: Empty ECAL events: {num_empty_ECAL} / {num_events} = {num_empty_ECAL/num_events:0.5f} (using only global data u)')        
    
    return dataset


def get_node_features(p4vec, p4track, X, ids, num_nodes, num_node_features, coord):
    
    # Node feature matrix
    x = np.zeros((num_nodes, num_node_features), dtype=float)

    for i in range(num_nodes - 1): # Last one is the empty event case
        
        if   coord == 'ptetaphim':
            x[i,0] = p4vec[i].pt
            x[i,1] = p4vec[i].eta
            x[i,2] = p4vec[i].phi
            x[i,3] = p4vec[i].m
        elif coord == 'pxpypze':
            x[i,0] = p4vec[i].px
            x[i,1] = p4vec[i].py
            x[i,2] = p4vec[i].pz
            x[i,3] = p4vec[i].e
        else:
            raise Exception(__name__ + f'.get_node_features: Unknown coordinate representation')
        
        # other features
        x[i,4] = p4track.deltaR(p4vec[i])

        try:
            x[i,5] = X[ids.index('image_clu_nhit')][i]
        except:
            continue
            # Not able to read it (empty cluster data)
    
    # Cast
    x = x.astype(float)

    return x
