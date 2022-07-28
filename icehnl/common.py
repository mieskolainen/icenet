# Common input & data reading routines for the HNL analysis
# 
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

import numpy as np
import uproot
from tqdm import tqdm
import psutil
import copy
import os
import pickle

from termcolor import colored, cprint

from icenet.tools import io
from icenet.tools import aux
from icenet.tools import plots
from icenet.tools import prints
from icenet.tools import process
from icenet.tools import iceroot

# GLOBALS
from configs.hnl.mvavars import *
#from configs.hnl.cuts import *
#from configs.hnl.filter import *


def load_root_file(root_path, ids=None, entry_start=0, entry_stop=None, args=None):
    """ Loads the root file.
    
    Args:
        root_path : paths to root files
    
    Returns:
        X,Y       : input, output matrices
        ids       : variable names
    """

    # -----------------------------------------------
    param = {
        'entry_start': entry_start,
        "entry_stop":  entry_stop,
        "args": args
    }

    # =================================================================
    # *** MC (signal and background) ***
    
    rootfile = f'{root_path}/{args["mcfile"]}'
    
    with open(rootfile, 'rb') as f:
        frame = pickle.load(f)

    ids = frame.keys()

    X   = frame[ids].to_numpy()
    W   = frame['event_weight'].to_numpy()
    Y   = frame['label'].to_numpy().astype(int)

    ## Print some diagnostics
    label_type = frame['label_type'].to_numpy().astype(int)
    
    for c in np.unique(Y):
        print(__name__ + f'.load_root_file: class[{c}] has unique "label_type" = {np.unique(label_type[Y==c])}')        
        print(__name__ + f'.load_root_file: class[{c}] mean(event_weight) = {np.mean(W[Y==c]):0.3f}, std(event_weight) = {np.std(W[Y==c]):0.3f}')
    
    # 'label_type' has values:
    #   hnl          = 1
    #   wjets_label  = 2
    #   dyjets_label = 3
    #   ttbar_label  = 4

    # ** Crucial -- randomize order to avoid problems with other functions **
    rind = np.random.permutation(len(X))
    X    = X[rind].squeeze() # Squeeze removes additional [] dimension
    Y    = Y[rind].squeeze()
    W    = W[rind].squeeze()

    # Apply maxevents cutoff
    maxevents = np.min([args['maxevents'], len(X)])
    X, Y, W = X[0:maxevents], Y[0:maxevents], W[0:maxevents]
    
    return X, Y, W, ids


def splitfactor(x, y, w, ids, args):
    """
    Transform data into different datatypes.
    
    Args:
        data:  jagged arrays
        args:  arguments dictionary
    
    Returns:
        dictionary with different data representations
    """
    
    data = io.IceXYW(x=x, y=y, w=w, ids=ids)
    
    # -------------------------------------------------------------------------
    ### Pick kinematic variables out
    data_kin = None

    if KINEMATIC_VARS is not None:
        k_ind, k_vars = io.pick_vars(data, KINEMATIC_VARS)
        
        data_kin      = copy.deepcopy(data)
        data_kin.x    = data.x[:, k_ind].astype(np.float)
        data_kin.ids  = k_vars

    # -------------------------------------------------------------------------
    data_deps   = None

    # -------------------------------------------------------------------------
    data_tensor = None

    # -------------------------------------------------------------------------
    data_graph  = None
    
    # -------------------------------------------------------------------------
    # Mutual information regularization targets
    MI_ind, MI_vars = io.pick_vars(data, globals()['MI_VARS'])
    data_MI = data.x[:, MI_ind].astype(np.float)
    
    # --------------------------------------------------------------------
    ### Finally pick active scalar variables out
    s_ind, s_vars = io.pick_vars(data, globals()[args['inputvar']])
    
    data.x   = data.x[:, s_ind].astype(np.float)
    data.ids = s_vars
    
    return {'data': data, 'data_MI': data_MI, 'data_kin': data_kin, 'data_deps': data_deps, 'data_tensor': data_tensor, 'data_graph': data_graph}
