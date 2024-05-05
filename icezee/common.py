# Common input & data reading routines for ZEE
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import copy
import pickle
from importlib import import_module
import pandas as pd

from termcolor import colored, cprint

from icenet.tools import io
from icenet.tools import aux

# GLOBALS
#from configs.zee.cuts import *
#from configs.zee.filter import *


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
    inputvars = import_module("configs." + args["rootname"] + "." + args["inputvars"])
    
    if type(root_path) is list:
        root_path = root_path[0] # Remove [] list, we expect only the path here
    
    # -----------------------------------------------
    #param = {
    #    'entry_start': entry_start,
    #    "entry_stop":  entry_stop,
    #    "maxevents":   maxevents,
    #    "args": args
    #}

    LOAD_VARS = inputvars.LOAD_VARS
    
    
    # =================================================================
    # *** MC ***
    
    mc_file  = f'{root_path}/{args["mcfile"]}'
    print(__name__ + f'.load_root_file: {mc_file}')
    
    frame_mc = pd.read_parquet(mc_file)
    
    print(f'Total number of events in file: {len(frame_mc)}')
    print(f'ids: {frame_mc.keys()}')
    
    # Pre-computed weights (kinematic re-weight x gen event weight x ...)
    W_MC = frame_mc[['rw_weights']].to_numpy().squeeze()
    W_MC = W_MC / np.sum(W_MC) * len(W_MC)

    X_MC = frame_mc[LOAD_VARS].to_numpy()
    Y_MC = np.zeros(len(X_MC)).astype(int)
    
    print(f'X_MC.shape = {X_MC.shape}')
    print(f'W_MC.shape = {W_MC.shape}')
    
    
    # =================================================================
    # *** Data ***
    
    data_file  = f'{root_path}/{args["datafile"]}'
    print(__name__ + f'.load_root_file: {data_file}')
    
    frame_data = pd.read_parquet(data_file)
    
    print(f'Total number of events in file: {len(frame_data)}')
    print(f'ids: {frame_data.keys()}')
    
    # Unit weighted events for data
    X_data = frame_data[LOAD_VARS].to_numpy()
    W_data = np.ones(len(X_data))
    Y_data = np.ones(len(X_data)).astype(int)
    
    print(f'X_data.shape = {X_data.shape}')
    print(f'W_data.shape = {W_data.shape}')
    
    
    # =================================================================
    # Combine
    
    X   = np.vstack((X_MC, X_data))
    W   = np.concatenate((W_MC, W_data))
    Y   = np.concatenate((Y_MC, Y_data))
    ids = LOAD_VARS
    
    # =================================================================
    
    
    ## Print some diagnostics
    print(__name__ + f'.load_root_file: Number of events: {len(X)}')
    
    for c in np.unique(Y):      
        print(__name__ + f'.load_root_file: class[{c}] mean(event_weight) = {np.mean(W[Y==c]):0.3E}, std(event_weight) = {np.std(W[Y==c]):0.3E}')
    
    # ** Crucial -- randomize order to avoid problems with other functions **
    rand = np.random.permutation(len(X))
    X    = X[rand].squeeze() # Squeeze removes additional [] dimension
    Y    = Y[rand].squeeze()
    W    = W[rand].squeeze()

    # Apply maxevents cutoff
    maxevents = np.min([args['maxevents'], len(X)])
    print(__name__ + f'.load_root_file: Applying maxevents cutoff {maxevents}')
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
    
    #vars    = aux.process_regexp_ids(all_ids=data.ids, ids=inputvars.MI_VARS)
    #data_MI = data[vars].x.astype(np.float32)
    data_MI = None
    
    # --------------------------------------------------------------------
    ### Finally pick active scalar variables out
    
    vars   = aux.process_regexp_ids(all_ids=data.ids, ids=eval('inputvars.' + args['inputvar_scalar']))
    data   = data[vars]
    data.x = data.x.astype(np.float32)
    
    return {'data': data, 'data_MI': data_MI, 'data_kin': data_kin, 'data_deps': data_deps, 'data_tensor': data_tensor, 'data_graph': data_graph}
