# Common input & data reading routines for the HNL analysis
# 
# Mikael Mieskolainen, 2023
# m.mieskolainen@imperial.ac.uk

import numpy as np
import copy
import pandas as pd
from importlib import import_module

from icenet.tools import io
from icenet.tools import aux

# ------------------------------------------
from icenet import print
# ------------------------------------------

# GLOBALS
#from configs.hnl.cuts import *
#from configs.hnl.filter import *


def load_root_file(root_path, ids=None, entry_start=0, entry_stop=None, maxevents=None, args=None):
    """ Loads the root file.
    
    Args:
        root_path : paths to root files (list)
    
    Returns:
        X:     columnar data
        Y:     class labels
        W:     event weights
        ids:   columnar variable string (list)
        info:  trigger and pre-selection acceptance x efficiency information (dict)
    """
    #inputvars = import_module("configs." + args["rootname"] + "." + args["inputvars"])
    
    if type(root_path) is list:
        root_path = root_path[0] # Remove [] list, we expect only the path here
    
    # -----------------------------------------------
    #param = {
    #    'entry_start': entry_start,
    #    "entry_stop":  entry_stop,
    #    "maxevents":   maxevents,
    #    "args": args
    #}

    # =================================================================
    # *** MC (signal and background) ***
    
    frames  = []
    mcfiles = io.glob_expand_files(datasets=args["mcfile"], datapath=root_path)
    
    for f in mcfiles:
        new_frame = copy.deepcopy(pd.read_parquet(f))
        frames.append(new_frame)
        print(f'{f} | N = {len(new_frame)}')
    
    frame = pd.concat(frames)
    
    ids = frame.keys()
    print(ids)
    
    X   = frame[ids].to_numpy()
    W   = frame['event_weight'].to_numpy()
    Y   = frame['label'].to_numpy().astype(int)

    ## Print some diagnostics
    label_type = frame['label_type'].to_numpy().astype(int)
    
    print(f'Number of events: {len(X)}')
    
    for c in np.unique(Y):
        print(f'class[{c}] has unique "label_type" = {np.unique(label_type[Y==c])}')        
        print(f'class[{c}] mean(event_weight) = {np.mean(W[Y==c]):0.3f}, std(event_weight) = {np.std(W[Y==c]):0.3f}')
    
    ## ** Set all weights to one **
    W   = np.ones(len(W))
    
    # 'label_type' has values:
    #   hnl          = 1
    #   wjets_label  = 2
    #   dyjets_label = 3
    #   ttbar_label  = 4

    # ** Crucial -- randomize order to avoid problems with other functions **
    rand = np.random.permutation(len(X))
    X    = X[rand].squeeze() # Squeeze removes additional [] dimension
    Y    = Y[rand].squeeze()
    W    = W[rand].squeeze()

    # Apply maxevents cutoff
    maxevents = np.min([maxevents, len(X)])
    X, Y, W = X[0:maxevents], Y[0:maxevents], W[0:maxevents]
    
    # TBD add cut statistics etc. here
    info = {}
    
    return {'X':X, 'Y':Y, 'W':W, 'ids':ids, 'info':info}


def splitfactor(x, y, w, ids, args):
    """
    Transform data into different datatypes.
    
    Args:
        data:  jagged arrays
        args:  arguments dictionary
    
    Returns:
        dictionary with different data representations
    """
    inputvars = import_module("configs." + args["rootname"] + "." + args["inputvars"])
    
    data = io.IceXYW(x=x, y=y, w=w, ids=ids)

    # -------------------------------------------------------------------------
    ### Pick kinematic variables out
    data_kin = None

    if inputvars.KINEMATIC_VARS is not None:
        vars       = aux.process_regexp_ids(all_ids=data.ids, ids=inputvars.KINEMATIC_VARS)
        data_kin   = data[vars]
        data_kin.x = data_kin.x.astype(np.float32)
    
    # -------------------------------------------------------------------------
    data_deps   = None

    # -------------------------------------------------------------------------
    data_tensor = None

    # -------------------------------------------------------------------------
    data_graph  = None
    
    # -------------------------------------------------------------------------
    # Mutual information regularization targets
    
    vars    = aux.process_regexp_ids(all_ids=data.ids, ids=inputvars.MI_VARS)
    data_MI = data[vars].x.astype(np.float32)
    
    # --------------------------------------------------------------------
    ### Finally pick active scalar variables out
    
    vars   = aux.process_regexp_ids(all_ids=data.ids, ids=eval('inputvars.' + args['inputvar_scalar']))
    data   = data[vars]
    data.x = data.x.astype(np.float32)
    
    return {'data':        data,
            'data_MI':     data_MI,
            'data_kin':    data_kin,
            'data_deps':   data_deps,
            'data_tensor': data_tensor,
            'data_graph':  data_graph}
