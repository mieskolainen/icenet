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
    
    if maxevents is None:
        maxevents = int(1e10)
        cprint(__name__ + f'.load_root_file: "maxevents" is None, set {maxevents}', 'yellow')
    
    
    # -------------------------------------------------------------------------
    # *** MC ***
    
    frames = []
    
    mcfiles = io.glob_expand_files(datasets=args["mcfile"], datapath=root_path)
    
    for f in mcfiles:
        new_frame = copy.deepcopy(pd.read_parquet(f))
        frames.append(new_frame)
        cprint(__name__ + f'.load_root_file: {f} | N = {len(new_frame)}', 'yellow')
        ids = list(new_frame.keys()); ids.sort()
        print(ids)
    
    frame_mc = pd.concat(frames)
    
    print(f'Total number of events: {len(frame_mc):0.1E}')
    
    X_MC = frame_mc[LOAD_VARS].to_numpy()
    
    ## Pre-computed weights (gen event weight x CMS weights)
    W_MC = frame_mc[['weight']].to_numpy().squeeze()
    #W_MC_rw = frame_mc[['rw_weights']].to_numpy().squeeze()
    #W_MC    = W_MC / W_MC_rw # Extract out raw "gen" weights
    W_MC = W_MC / np.sum(W_MC) * len(W_MC)
    
    # Use all events with weight 1
    #W_MC = np.ones(len(X_MC))
    
    # Label = 0
    Y_MC = np.zeros(len(X_MC)).astype(int)
    
    # ** Drop negative weight events **
    """
    ind  = W_MC < 0
    if np.sum(ind) > 0:
        cprint(__name__ + f'.load_root_file: Dropping negative weight events ({np.sum(ind)/len(ind):0.3f})', 'red')
        W_MC = W_MC[~ind] # Boolean NOT
        X_MC = X_MC[~ind]
        Y_MC = Y_MC[~ind]
    """
    
    print(f'X_MC.shape = {X_MC.shape}')
    print(f'W_MC.shape = {W_MC.shape}')
    
    
    # -------------------------------------------------------------------------
    # *** Data ***
    
    frames = []
    
    datafiles = io.glob_expand_files(datasets=args["datafile"], datapath=root_path)
    
    for f in datafiles:
        new_frame = copy.deepcopy(pd.read_parquet(f))
        frames.append(new_frame)
        cprint(__name__ + f'.load_root_file: {f} | N = {len(new_frame)}', 'yellow')
        ids = list(new_frame.keys()); ids.sort()
        print(ids)
    
    frame_data = pd.concat(frames)
    
    print(f'Total number of events: {len(frame_data):0.1E}')
    
    # -------------------------------------------------------------------------
    # ** Special treatment -- different naming in data **
    NEW_LOAD_VARS = copy.deepcopy(LOAD_VARS)
    for i in range(len(LOAD_VARS)):
        if LOAD_VARS[i] == 'probe_pfChargedIso':
            cprint(f'Changing variable {LOAD_VARS[i]} name in data', 'red')
            NEW_LOAD_VARS[i] = 'probe_pfChargedIsoPFPV'
    # -------------------------------------------------------------------------
    
    X_data = frame_data[NEW_LOAD_VARS].to_numpy()
    W_data = np.ones(len(X_data))
    
    # Label = 1
    Y_data = np.ones(len(X_data)).astype(int)
    
    print(f'X_data.shape = {X_data.shape}')
    print(f'W_data.shape = {W_data.shape}')
    
    
    # -------------------------------------------------------------------------
    # Combine MC and Data samples
    
    X   = np.vstack((X_MC, X_data))
    W   = np.concatenate((W_MC, W_data))
    Y   = np.concatenate((Y_MC, Y_data))
    ids = LOAD_VARS
    
    
    # -------------------------------------------------------------------------
    ## Print some diagnostics
    print(__name__ + f'.load_root_file: Number of events: {len(X)}')
    
    for c in np.unique(Y):      
        print(__name__ + f'.load_root_file: class[{c}] weight[mean,std,min,max] = {np.mean(W[Y==c]):0.3E}, {np.std(W[Y==c]):0.3E}, {np.min(W[Y==c]):0.3E}, {np.max(W[Y==c]):0.3E}')
    
    # ** Crucial -- randomize order to avoid problems with other functions **
    rand = np.random.permutation(len(X))
    X    = X[rand].squeeze() # Squeeze removes additional [] dimension
    Y    = Y[rand].squeeze()
    W    = W[rand].squeeze()
    
    # Apply maxevents cutoff
    maxevents = np.min([maxevents, len(X)])
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
    
    data_MI = None
    
    if inputvars.MI_VARS is not None:
        vars    = aux.process_regexp_ids(all_ids=data.ids, ids=inputvars.MI_VARS)
        data_MI = data[vars].x.astype(np.float32)
    
    # -------------------------------------------------------------------------
    ### Finally pick active scalar variables out
    
    vars   = aux.process_regexp_ids(all_ids=data.ids, ids=eval('inputvars.' + args['inputvar_scalar']))
    data   = data[vars]
    data.x = data.x.astype(np.float32)
    
    # -------------------------------------------------------------------------
    ### ** DEBUG TEST -- special transform for "zero-inflated" variables **
    """
    special_var = ['probe_esEffSigmaRR',
                   'probe_pfChargedIso',
                   'probe_ecalPFClusterIso',
                   'probe_trkSumPtHollowConeDR03',
                   'probe_trkSumPtSolidConeDR04']
    
    # Thresholds
    dT          = [0.1, 0.1, 0.1, 0.1, 0.1]
    
    for i, v in enumerate(special_var):
        
        try:
            ind = data.find_ind(v)
        except:
            cprint(__name__ + f'.splitfactor: Could not find variable "{v}" -- continue', 'red')
        
        cprint(__name__ + f'.splitfactor: Pre-transforming variable "{v}" with dT < {dT[i]}', 'magenta')
        
        mask              = np.abs(data.x[:,ind] < dT[i])
        data.x[mask, ind] = np.random.triangular(left=-1, mode=-0.75, right=dT[i], size=np.sum(mask))
        data.x[:,ind]     = np.log1p(data.x[:,ind] + 1)
        
        # Change the variable name
        data.ids[ind] = f'DQL__{v}'
    """
    # -------------------------------------------------------------------------
    
    return {'data': data, 'data_MI': data_MI, 'data_kin': data_kin, 'data_deps': data_deps, 'data_tensor': data_tensor, 'data_graph': data_graph}
