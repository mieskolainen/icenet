# Common input & data reading routines for the DQCD analysis
# 
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk


import numpy as np
import uproot
from tqdm import tqdm
import psutil
import copy
import os

from termcolor import colored, cprint


from icenet.tools import io
from icenet.tools import aux
from icenet.tools import plots
from icenet.tools import prints
from icenet.tools import process
from icenet.tools import iceroot

from icedqcd import graphio


# GLOBALS
from configs.dqcd.mvavars import *
from configs.dqcd.cuts import *
from configs.dqcd.filter import *


def load_root_file(root_path, ids=None, entry_start=0, entry_stop=None, args=None):
    """ Loads the root file with signal events from MC and background from DATA.
    
    Args:
        root_path : paths to root files
    
    Returns:
        X,Y       : input, output matrices
        ids       : variable names
    """

    # -----------------------------------------------

    param = {
        "tree":        "Events",
        "entry_start": entry_start,
        "entry_stop":  entry_stop,
        "args":        args,
        "load_ids":    LOAD_VARS,
        "isMC":        True
    }

    # =================================================================
    # *** SIGNAL MC *** (first signal, so we can use it's theory conditional parameters)

    proc = args["input"]['class_1']
    X_S, Y_S, W_S, VARS_S = iceroot.read_multiple_MC(class_id=1,
        process_func=process_root, processes=proc, root_path=root_path, param=param)
    
    
    # =================================================================
    # *** BACKGROUND MC ***
    
    proc = args["input"]['class_0']
    X_B, Y_B, W_B, VARS_B = iceroot.read_multiple_MC(class_id=0,
        process_func=process_root, processes=proc, root_path=root_path, param=param)
    
    
    # =================================================================
    # Sample conditional theory parameters for the background as they are distributed in signal sample
        
    for var in MODEL_VARS:
        
        print(__name__ + f'.load_root_file: Sampling theory conditional parameter "{var}" for the background')

        # Random-sample values for the background as in the signal MC
        p   = ak.to_numpy(W_S / ak.sum(W_S)).squeeze() # probability per event entry
        new = np.random.choice(ak.to_numpy(X_S[var]).squeeze(), size=len(X_B), replace=True, p=p)
    
        X_B[var] = ak.Array(new)

    # =================================================================
    # *** Finally combine ***

    X = ak.concatenate((X_B, X_S), axis=0)
    Y = ak.concatenate((Y_B, Y_S), axis=0)
    W = ak.concatenate((W_B, W_S), axis=0)

    # ** Crucial -- randomize order to avoid problems with other functions **
    rind = np.random.permutation(len(X))

    X    = X[rind]
    Y    = Y[rind]
    W    = W[rind]
    
    print(__name__ + f'.common.load_root_file: len(X) = {len(X)}')

    return X, Y, W, VARS_S


def process_root(X, ids, isMC, args, **extra):
    
    CUTFUNC    = globals()[args['cutfunc']]
    FILTERFUNC = globals()[args['filterfunc']]

    # @@ Filtering done here @@
    mask = FILTERFUNC(X=X, ids=ids, isMC=isMC, xcorr_flow=args['xcorr_flow'])
    plots.plot_selection(X=X, mask=mask, ids=ids, plotdir=args['plotdir'], label=f'<filterfunc>_{isMC}', varlist=PLOT_VARS, library='ak')
    cprint(__name__ + f'.process_root: isMC = {isMC} | <filterfunc>  before: {len(X)}, after: {sum(mask)} events ({sum(mask)/len(X):0.6f})', 'green')
    
    X   = X[mask]
    prints.printbar()

    # @@ Observable cut selections done here @@
    mask = CUTFUNC(X=X, ids=ids, isMC=isMC, xcorr_flow=args['xcorr_flow'])
    plots.plot_selection(X=X, mask=mask, ids=ids, plotdir=args['plotdir'], label=f'<cutfunc>_{isMC}', varlist=PLOT_VARS, library='ak')
    cprint(__name__ + f".process_root: isMC = {isMC} | <cutfunc>: before: {len(X)}, after: {sum(mask)} events ({sum(mask)/len(X):0.6f}) \n", 'green')

    X   = X[mask]
    io.showmem()
    prints.printbar()

    return X, ids


def splitfactor(x, y, w, ids, args):
    """
    Transform data into different datatypes.
    
    Args:
        data:  jagged arrays
        args:  arguments dictionary
    
    Returns:
        dictionary with different data representations
    """
    data   = io.IceXYW(x=x, y=y, w=w, ids=ids)
    
    data.y = ak.to_numpy(data.y)
    data.w = ak.to_numpy(data.w)

    # -------------------------------------------------------------------------
    ### Pick kinematic variables out
    data_kin = None
    
    if KINEMATIC_VARS is not None:

        data_kin      = copy.deepcopy(data)
        data_kin.x    = aux.ak2numpy(x=data.x, fields=KINEMATIC_VARS)
        data_kin.ids  = KINEMATIC_VARS

    # -------------------------------------------------------------------------
    ## Graph representation
    data_graph = None

    features   = globals()[args['inputvar_scalar']]
    data_graph = graphio.parse_graph_data(X=data.x, Y=data.y, weights=data.w, ids=data.ids, 
        features=features, graph_param=args['graph_param'])

    # -------------------------------------------------------------------------
    ## Tensor representation
    data_tensor = None

    # -------------------------------------------------------------------------
    ## Turn jagged data to "long-vector" matrix representation

    ### Pick active scalar variables out
    scalar_vars   = globals()[args['inputvar_scalar']]
    jagged_vars   = globals()[args['inputvar_jagged']]
    
    # Create tuplet expanded jagged variable names
    all_jagged_vars = []
    jagged_maxdim   = []
    jagged_totdim   = int(0)
    for i in range(len(jagged_vars)):

        sf             = jagged_vars[i].split('_')
        thisdim        = int(args['jagged_maxdim'][sf[0]])
        jagged_totdim += thisdim

        for j in range(thisdim):
            all_jagged_vars.append( f'{jagged_vars[i]}[{j}]' )
            jagged_maxdim.append(thisdim)

    # Update representation
    arg = {
        'scalar_vars'  : scalar_vars,
        'jagged_vars'  : jagged_vars,
        'jagged_maxdim': jagged_maxdim,
        'jagged_totdim': jagged_totdim
    }
    mat      = aux.jagged2matrix(data.x, **arg)

    data.x   = ak.to_numpy(mat)
    data.y   = ak.to_numpy(data.y)
    data.ids = scalar_vars + all_jagged_vars
    # --------------------------------------------------------------------------
    
    # --------------------------------------------------------------------------
    # Create DeepSet style representation from the "long-vector" content
    data_deps = None
    data_deps = copy.deepcopy(data)
    
    M = len(jagged_vars)              # Number of (jagged) variables per event
    D = args['jagged_maxdim']['sv']   # Tuplet feature vector dimension
    data_deps.x   = aux.longvec2matrix(X=data.x[:, len(scalar_vars):], M=M, D=D)
    data_deps.y   = data_deps.y
    data_deps.ids = all_jagged_vars
    # --------------------------------------------------------------------------
    
    
    return {'data': data, 'data_kin': data_kin, 'data_deps': data_deps, 'data_tensor': data_tensor, 'data_graph': data_graph}
