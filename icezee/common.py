# Common input & data reading routines for ZEE
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import copy
from importlib import import_module
import pandas as pd
import os

from icenet.tools import io
from icenet.tools import aux

# ------------------------------------------
from icenet import print
# ------------------------------------------

# GLOBALS
#from configs.zee.cuts import *
#from configs.zee.filter import *

def truncate(X,Y,W,maxevents):

    # Apply maxevents cutoff
    maxevents = np.min([maxevents, len(X)])
    if maxevents < len(X):
        print(f'Applying maxevents cutoff {maxevents}')
        X, Y, W = X[0:maxevents], Y[0:maxevents], W[0:maxevents]
    
    return X,Y,W   

def load_helper(mcfiles, datafiles, maxevents, args):
    
    inputvars = import_module("configs." + args["rootname"] + "." + args["inputvars"])
    LOAD_VARS = inputvars.LOAD_VARS
    
    # -------------------------------------------------------------------------
    # *** MC ***
    
    frames = []
    
    for f in mcfiles:
        new_frame = copy.deepcopy(pd.read_parquet(f))
        frames.append(new_frame)
        print(f'{f} | N = {len(new_frame)}', 'yellow')
        ids = list(new_frame.keys()); ids.sort()
        print(ids)
    
    frame_mc = pd.concat(frames)
    
    print(f'Total number of events: {len(frame_mc):0.1E}')
    
    X_MC = frame_mc[LOAD_VARS].to_numpy()
    
    # Label = 0
    Y_MC = np.zeros(len(X_MC)).astype(int)
    
    ## Pre-computed weights (gen event weight x CMS weights)
    W_MC = frame_mc[['weight']].to_numpy().squeeze()
    
    print(f'X_MC.shape = {X_MC.shape}')
    print(f'W_MC.shape = {W_MC.shape}')
    
    # Apply maxevents cutoff
    X_MC, Y_MC, W_MC = truncate(X=X_MC, Y=Y_MC, W=W_MC, maxevents=maxevents)
    
    # -------------------------------------------------------------------------
    # *** Data ***
    
    frames = []
    
    for f in datafiles:
        new_frame = copy.deepcopy(pd.read_parquet(f))
        frames.append(new_frame)
        print(f'{f} | N = {len(new_frame)}', 'yellow')
        ids = list(new_frame.keys()); ids.sort()
        print(ids)
    
    frame_data = pd.concat(frames)
    
    print(f'Total number of events: {len(frame_data):0.1E}')
    
    # -------------------------------------------------------------------------
    # ** Special treatment -- different naming in data **
    NEW_LOAD_VARS = copy.deepcopy(LOAD_VARS)
    for i in range(len(LOAD_VARS)):
        if LOAD_VARS[i] == 'probe_pfChargedIso':
            print(f'Changing variable {LOAD_VARS[i]} name in data', 'red')
            NEW_LOAD_VARS[i] = 'probe_pfChargedIsoPFPV'
    # -------------------------------------------------------------------------
    
    X_data = frame_data[NEW_LOAD_VARS].to_numpy()
    W_data = np.ones(len(X_data))
    
    # Label = 1
    Y_data = np.ones(len(X_data)).astype(int)
    
    print(f'X_data.shape = {X_data.shape}')
    print(f'W_data.shape = {W_data.shape}')
    
    # Apply maxevents cutoff
    X_data, Y_data, W_data = truncate(X=X_data, Y=Y_data, W=W_data, maxevents=maxevents)
    
    # -------------------------------------------------------------------------
    # Combine MC and Data samples
    
    X   = np.vstack((X_MC, X_data))
    Y   = np.concatenate((Y_MC, Y_data))
    W   = np.concatenate((W_MC, W_data))
    
    ids = LOAD_VARS # We use these
    
    ## -------------------------------------------------
    # ** Drop negative weight (MC) events **
    if args['drop_negative']:
        ind = W < 0
        if np.sum(ind) > 0:
            print(f'Dropping negative weight events ({np.sum(ind)/len(ind):0.3f})', 'red')
            X = X[~ind]
            W = W[~ind] # Boolean NOT
            Y = Y[~ind]
    
    # Re-nenormalize MC to the event count
    ind    = (Y == 0)
    W[ind] = W[ind] / np.sum(W[ind]) * len(W[ind])
    ## -------------------------------------------------
    
    # -------------------------------------------------------------------------
    # ** Randomize MC vs Data order to avoid problems with other functions **
    # ** No need to randomize in this application. We have fixed (eval, validate, test) file structure **
    
    #rand = np.random.permutation(len(X))
    #X    = X[rand].squeeze() # Squeeze removes additional [] dimension
    #Y    = Y[rand].squeeze()
    #W    = W[rand].squeeze()
    # -------------------------------------------------------------------------
    
    # -------------------------------------------------------------------------
    ## Add event weights as an aux variable called `raw_weight` for plots etc.
    
    X   = np.hstack((X, W[None].T))
    ids = ids + ['raw_weight']
    
    # -------------------------------------------------------------------------
    
    ## Print some diagnostics
    print(f'Total number of events: {len(X)}')
    
    for c in np.unique(Y):
        print(f'class[{c}] | N = {np.sum(Y == c)} | weight[mean,std,min,max] = {np.mean(W[Y==c]):0.3E}, {np.std(W[Y==c]):0.3E}, {np.min(W[Y==c]):0.3E}, {np.max(W[Y==c]):0.3E}')
    
    return X,Y,W,ids


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
    
    if type(root_path) is list:
        root_path = root_path[0] # Remove [] list, we expect only the path here
    
    # -----------------------------------------------
    #param = {
    #    'entry_start': entry_start,
    #    "entry_stop":  entry_stop,
    #    "maxevents":   maxevents,
    #    "args": args
    #}

    if maxevents is None:
        maxevents = int(1e10)
        print(f'"maxevents" is None, set {maxevents}', 'yellow')
    
    # -------------------------------------------------------------------------
    # We do splitting according to dictionary (trn, val, tst) structure
    
    running_split = {}
    X = {}
    Y = {}
    W = {}
    N_prev = 0
    
    for mode in ['trn', 'val', 'tst']:
        
        # Glob expansion type (be careful not to have "label noise" under the folder or subfolders)
        mc_files  = io.glob_expand_files(datasets=args["mcfile"][mode],   datapath=root_path)
        da_files  = io.glob_expand_files(datasets=args["datafile"][mode], datapath=root_path)
        
        # Simple fixed one file
        #mc_files = [os.path.join(root_path, args['mcfile'][mode][0])]
        #da_files = [os.path.join(root_path, args['mcfile'][mode][0])]
        
        print(f'Found mcfiles:   {mc_files}')
        print(f'Found datafiles: {da_files}')
        
        X[mode],Y[mode],W[mode],ids = load_helper(mcfiles=mc_files, datafiles=da_files, maxevents=maxevents, args=args)
        running_split[mode] = np.arange(N_prev, len(X[mode]) + N_prev)
        N_prev += len(X[mode]) #!
    
    # Combine
    X = np.vstack((X['trn'], X['val'], X['tst']))
    Y = np.concatenate((Y['trn'], Y['val'], Y['tst']))
    W = np.concatenate((W['trn'], W['val'], W['tst']))
    
    # Aux info here
    info = {'running_split': running_split}
    
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
    ### Pick kinematic variables out (note also 'raw_weight')
    data_kin = None

    if inputvars.KINEMATIC_VARS is not None:
        vars       = aux.process_regexp_ids(all_ids=data.ids, ids=inputvars.KINEMATIC_VARS + ['raw_weight'])
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
    ### ** Special transform for specific MVA input variables **
    
    if args['pre_transform']['active']:
        
        var  = args['pre_transform']['var']
        func = args['pre_transform']['func']
        
        for i, v in enumerate(var):
            
            try:
                ind = data.find_ind(v)
            except:
                print(f'Could not find variable "{v}" -- continue', 'red')
                continue
            
            print(f'Pre-transform: {func[i]} of variable "{v}"', 'magenta')
            
            x = data.x[:,ind] # x used in the transform
            data.x[:,ind] = eval(func[i])
            
            # Change the variable name [+ have the same original variables in data_kin]
            data.ids[ind] = f'TRF__{v}'
    
    return {'data': data, 'data_MI': data_MI, 'data_kin': data_kin, 'data_deps': data_deps, 'data_tensor': data_tensor, 'data_graph': data_graph}
