# Graph data readers and parsers for DQCD
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import awkward as ak
import ray
from tqdm import tqdm

from termcolor import colored, cprint

import torch
from   torch_geometric.data import Data

import icenet.algo.analytic as analytic
from   icenet.tools import aux
from   icenet.tools.icevec import vec4


@ray.remote
def parse_graph_data_ray(X, ids, features, node_features, graph_param,
    Y=None, weights=None, entry_start=None, entry_stop=None, null_value=float(-999.0), EPS=1e-12):
    
    return parse_graph_data(X, ids, features, node_features, graph_param,
        Y, weights, entry_start, entry_stop, null_value, EPS)

def parse_graph_data(X, ids, features, node_features, graph_param,
    Y=None, weights=None, entry_start=None, entry_stop=None, null_value=float(-999.0), EPS=1e-12):
    """
    Jagged array data into pytorch-geometric style Data format array.
    
    Args:
        X          :  Jagged Awkward array of variables
        ids        :  Variable names as an array of strings
        features   :  List of active global feature strings
        graph_param:  Graph construction parameters dict
        Y          :  Target class  array (if any, typically MC only)
        weights    :  (Re-)weighting array (if any, typically MC only)
    
    Returns:
        List of pytorch-geometric Data objects
    """

    M_MUON     = float(0.105658) # Muon mass (GeV)
    M_PION     = float(0.139)    # Charged pion mass (GeV)

    global_on  = graph_param['global_on']
    coord      = graph_param['coord']
    directed   = graph_param['directed']
    self_loops = graph_param['self_loops']

    # --------------------------------------------------------------------
    entry_start, entry_stop, num_events = aux.slice_range(start=entry_start, stop=entry_stop, N=len(X))
    dataset = []
    
    #print(__name__ + f'.parse_graph_data: Converting {num_events} events into graphs ...')

    num_global_features = len(features)
    
    num_node_features        = {}
    node_features_hetero_ind = {}

    k = 0
    for key in node_features.keys():
        num_node_features[key]        = len(node_features[key])
        node_features_hetero_ind[key] = np.arange(k, k + num_node_features[key])
        k += num_node_features[key]

    #print(__name__ + f'.parse_graph_data: hetero_ind: {node_features_hetero_ind}')
    num_edge_features = 4
    
    # -----------------------------------------------
    # Heterogeneous features per node
    jvname = {}
    for key in num_node_features:
        jvname[key] = []
        for j in range(num_node_features[key]):

            # Awkward groups jagged variables, e.g. 'sv_x' to sv.x
            # argument 1 takes the first '_' occurance
            jvname[key].append(node_features[key][j].split('_', 1))
    # -----------------------------------------------

    # Loop over events
    for ev in tqdm(range(entry_start, entry_stop), miniters=int(np.ceil(num_events/10))):
        
        # Count the number of heterogeneous nodes by picking the first feature
        nums = {}
        for key in jvname.keys():
            nums[key] = len(X[ev][jvname[key][0][0]][jvname[key][0][1]])
        
        #print(__name__ + f'.parse_graph_data: event: {ev} | num_nodes: {nums}')

        num_nodes = np.sum([nums[key] for key in nums.keys()])
        num_edges = analytic.count_simple_edges(num_nodes=num_nodes, directed=directed, self_loops=self_loops)

        # ------------------------------------------------------------------
        # Construct the node features
        p4vec  = []

        # Node feature matrix
        x = np.zeros((num_nodes, np.sum([num_node_features[key] for key in num_node_features.keys()] )), dtype=float)
        tot_nodes = 0

        for key in jvname.keys():

            if nums[key] > 0:

                # ------------------------------------------------
                # Collect all particle features in columnar way, all particles simultaneously
                for j in range(num_node_features[key]):
                    x[tot_nodes:tot_nodes+nums[key], node_features_hetero_ind[key][j]] = \
                        X[ev][jvname[key][j][0]][jvname[key][j][1]]
                
                tot_nodes += nums[key] # Running index
                # ------------------------------------------------

                """
                if key == 'muon':

                    a = jvname['muon'][0][0]
                    p4 = vector.Array({
                                    'pt' : X[ev][a].pt,
                                    'eta': X[ev][a].eta,
                                    'phi': X[ev][a].phi,
                                    'm':   (0.0*X[ev][a].phi + 1.0)*M_MUON})

                if key == 'jet' or key == 'sv':

                    a = jvname[key][0][0]
                    p4 = vector.Array({
                                    'pt' : X[ev][a].pt,
                                    'eta': X[ev][a].eta,
                                    'phi': X[ev][a].phi,
                                    'm':   X[ev][a].mass})
                
                if p4 is None:
                    p4vec = p4
                else:
                    p4vec = ak.concatenate((p4vec,p4))
                """

                # Loop over nodes of this hetero-type
                for k in range(nums[key]):
                    
                    # Construct 4-momentum
                    v = vec4() # initialized to zero by default

                    if key == 'muon':
                        a = jvname['muon'][0][0]
                        v.setPtEtaPhiM(X[ev][a].pt[k],
                                       X[ev][a].eta[k],
                                       X[ev][a].phi[k],
                                       M_MUON)

                    elif key == 'jet' or key == 'sv':
                        a = jvname[key][0][0]
                        v.setPtEtaPhiM(X[ev][a].pt[k],
                                       X[ev][a].eta[k],
                                       X[ev][a].phi[k],
                                       X[ev][a].mass[k])
                    
                    elif key == 'cpf':
                        v.setXYZM(X[ev].cpf.px[k], X[ev].cpf.py[k], X[ev].cpf.pz[k], M_PION)

                    elif key == 'npf':
                        v.setXYZM(X[ev].npf.pz[k], X[ev].npf.py[k], X[ev].npf.pz[k], 0)
                    
                    else:
                        True # Use null 4-vector

                    p4vec.append(v)

        # Empty information
        #if num_nodes - 1 == 0:
        #    num_empty += 1
        #    # However, never skip empty events here!!, do pre-filtering before this function if needed
        
        # ====================================================================
        # CONSTRUCT TENSORS
        # # https://pytorch.org/docs/stable/tensors.html

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
            u_mat = np.zeros(len(features), dtype=float)
            for j in range(len(features)):
                xx = ak.to_numpy([X[ev][features[j]]]) # outer [] for protection
                if xx is not []: u_mat[j] = xx
            
            u_mat[~np.isfinite(u_mat)] = null_value # Input protection
            u = torch.tensor(u_mat, dtype=torch.float)

        data = Data(num_nodes=x.shape[0], x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, w=w, u=u)
        dataset.append(data)
    
    #print(__name__ + f'.parse_graph_data: Empty events: {num_empty} / {num_events} = {num_empty/num_events:0.5f} (using only global data u)')        
    
    return dataset
