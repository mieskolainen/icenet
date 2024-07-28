# Common input & data reading routines for HGCAL (CND, TRK modes)
# 
# Mikael Mieskolainen, 2023
# m.mieskolainen@imperial.ac.uk

import pprint
from pprint import pprint

import os
import pickle
from importlib import import_module
import numpy as np
from tqdm import tqdm

from icenet.tools import io
from icenet.tools import aux
from icenet.tools import iceroot

from icehgcal import preprocess
from icehgcal import graphio

# ------------------------------------------
from icenet import print
# ------------------------------------------

# Globals
from configs.hgcal.mctargets import *
from configs.hgcal.mcfilter  import *
from configs.hgcal.cuts import *


def read_data_tracklet(args, runmode):
    """
    (HGCAL-TRK)
    """

    # Create trackster data
    cache_filename = f'{args["datadir"]}/data_{args["__hash__"]}.pkl'

    if args['__use_cache__'] == False or (not os.path.exists(cache_filename)):

        if runmode != "genesis":
            raise Exception(__name__ + f'Data not in cache (or __use_cache__ == False) but --runmode is not "genesis"')
        
        data      = preprocess.event_loop(files=args['root_files'], graph_param=args['graph_param'], maxevents=args['maxevents']) 
        X         = graphio.parse_graph_data_trackster(data=data, graph_param=args['graph_param'], weights=None)
        
        # Pickle to disk
        with open(cache_filename, "wb") as fp:
            pickle.dump([X, args], fp)
            print(f'Saved output to cache: "{cache_filename}"', 'yellow')

    else:
        with open(cache_filename, 'rb') as handle:
            print(f'Loading from cache: "{cache_filename}"', 'yellow')
            X, genesis_args = pickle.load(handle)

            print(f'Cached data was generated with arguments:', 'yellow')
            pprint(genesis_args)

    return X

def process_tracklet_data(args, X):
    """
    (HGCAL-TRK)
    """

    ### Edge weight re-weighting
    if args['reweight']:
        if args['reweight_param']['equal_frac']:

            print(f'Computing binary edge ratio balancing re-weights ...')

            # Count number of false/true edges
            num = [0,0]
            for i in tqdm(range(len(X))):
                for j in range(len(X[i].y)):
                    num[X[i].y[j]] += 1

            # Re-weight
            EQ = num[0]/num[1] if (args['reweight_param']['reference_class'] == 0) else num[1]/num[0]

            print(f'EQ = {EQ}')

            # Apply weights
            for i in tqdm(range(len(X))):
                for j in range(len(X[i].y)):
                    if X[i].y[j] != args['reweight_param']['reference_class']:
                        X[i].w[j] *= EQ
    
    ### Split
    X_trn, X_val, X_tst = io.split_data_simple(X=X, frac=args['frac'])
    data = {}
    data['trn'] = {'data': None, 'data_kin': None, 'data_deps': None, 'data_tensor': None, 'data_graph': X_trn}
    data['val'] = {'data': None, 'data_kin': None, 'data_deps': None, 'data_tensor': None, 'data_graph': X_val}
    data['tst'] = {'data': None, 'data_kin': None, 'data_deps': None, 'data_tensor': None, 'data_graph': X_tst}

    return data


def load_root_file(root_path, ids=None, entry_start=0, entry_stop=None, maxevents=None, args=None):
    """ 
    (HGCAL-CND)

    Loads the root files
    
    Args:
        root_path : paths to root files (list)
    
    Returns:
        X:     columnar data
        Y:     class labels
        W:     event weights
        ids:   columnar variable string (list)
        info:  trigger and pre-selection acceptance x efficiency information (dict)
    """
    inputvars = import_module("configs." + args["rootname"] + "." + args["inputvars"])
    
    if type(root_path) is list:
        root_path = root_path[0] # Remove [] list, we expect only the path here
    
    # -----------------------------------------------

    # ** Pick the variables **
    
    param = {
        "entry_start": entry_start,
        "entry_stop":  entry_stop,
        "maxevents":   maxevents,
        "args":        args,
        "load_ids":    inputvars.MVA_SCALAR_VARS
    }
    
    tree = args['tree_name']

    # =================================================================
    # *** BACKGROUND MC ***

    filename = args["input"]['class_0']
    rootfile = io.glob_expand_files(datasets=filename, datapath=root_path)
    
    X_B,ids  = process_root(rootfile=rootfile, tree=tree, isMC=True, **param)
    Y_B      = np.zeros(X_B.shape[0])


    # =================================================================
    # *** SIGNAL MC ***

    filename = args["input"]['class_1']
    rootfile = io.glob_expand_files(datasets=filename, datapath=root_path)

    X_S,ids  = process_root(rootfile=rootfile, tree=tree, isMC=True, **param)
    Y_S = np.ones(X_S.shape[0])
    
    
    # =================================================================
    # Finally combine

    X = np.concatenate((X_B, X_S), axis=0)
    Y = np.concatenate((Y_B, Y_S), axis=0)
    
    # Trivial weights
    W = np.ones(len(X))

    # =================================================================
    
    # ** Crucial -- randomize order to avoid problems with other functions **
    rand = np.random.permutation(len(X))
    X    = X[rand].squeeze() # Squeeze removes additional [] dimension
    Y    = Y[rand].squeeze()
    W    = W[rand].squeeze()
    
    # TBD add cut statistics etc. here
    info = {}
    
    return {'X':X, 'Y':Y, 'W':W, 'ids':ids, 'info':info}


def process_root(rootfile, tree, load_ids, isMC, entry_start, entry_stop, maxevents, args):
    """
    (HGCAL-CND)
    """

    CUTFUNC    = globals()[args['cutfunc']]
    FILTERFUNC = globals()[args['filterfunc']]

    X,ids      = iceroot.load_tree(rootfile=rootfile, tree=tree,
        entry_start=entry_start, entry_stop=entry_stop, maxevents=maxevents, ids=load_ids, library='np')
    
    """
    # @@ Filtering done here @@
    mask = FILTERFUNC(X=X, ids=ids, isMC=isMC, xcorr_flow=args['xcorr_flow'])
    plots.plot_selection(X=X, mask=mask, ids=ids, args=args, label=f'<filter>_{isMC}', varlist=CUT_VARS)
    print(f'isMC = {isMC} | <filterfunc> before: {len(X)}, after: {sum(mask)} events ', 'green')
    
    X   = X[mask]
    prints.printbar()
    
    # @@ Observable cut selections done here @@
    mask = CUTFUNC(X=X, ids=ids, isMC=isMC, xcorr_flow=args['xcorr_flow'])
    plots.plot_selection(X=X, mask=mask, ids=ids, args=args, label=f'<cutfunc>_{isMC}', varlist=CUT_VARS)
    print(f"isMC = {isMC} | <cutfunc>: before: {len(X)}, after: {sum(mask)} events \n", 'green')

    X   = X[mask]
    io.showmem()
    prints.printbar()
    """

    return X, ids


def splitfactor(x, y, w, ids, args):
    """
    (HGCAL-CND)
    
    Transform data into different datatypes.
    
    Args:
        data:  jagged arrays
        args:  arguments dictionary
    
    Returns:
        dictionary with different data representations
    """
    inputvars = import_module("configs." + args["rootname"] + "." + args["inputvars"])
    
    data = io.IceXYW(x=x, y=y, w=w, ids=ids)

    ### Pick active variables out
    scalar_vars = aux.process_regexp_ids(all_ids=ids, ids=eval('inputvars.' + args['inputvar_scalar']))

    # -------------------------------------------------------------------------
    ### Pick kinematic variables out
    data_kin = None
    
    #if inputvars.KINEMATIC_VARS is not None:
    #    vars = aux.process_regexp_ids(all_ids=ids, ids=inputvars.KINEMATIC_VARS)
    #    data_kin      = data[vars]
    #    data_kin.x    = data_kin.x.astype(np.float32)

    # -------------------------------------------------------------------------
    ### DeepSets representation
    data_deps   = None
    
    # -------------------------------------------------------------------------
    ### Tensor representation
    data_tensor = None

    # -------------------------------------------------------------------------
    ## Graph representation
    data_graph = None
    
    data_graph = graphio.parse_graph_data_candidate(X=data.x, Y=data.y, weights=data.w, ids=data.ids,
        features=scalar_vars, graph_param=args['graph_param'])
    
    # --------------------------------------------------------------------
    ### Finally pick active scalar variables out

    data.x = None # To protect other routines (TBD see global impact --> comment this line)
    
    return {'data': data, 'data_kin': data_kin, 'data_deps': data_deps, 'data_tensor': data_tensor, 'data_graph': data_graph}

# ========================================================================
# ========================================================================
