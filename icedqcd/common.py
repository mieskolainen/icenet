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


# GLOBALS
from configs.dqcd.mvavars import *
from configs.dqcd.cuts import *
from configs.dqcd.filter import *


def load_root_file(root_path, ids=None, entry_start=0, entry_stop=None, class_id=None, args=None):
    """ Loads the root file with signal events from MC and background from DATA.
    
    Args:
        root_path : paths to root files
        class_id  : class ids
    
    Returns:
        X,Y       : input, output matrices
        ids       : variable names
    """
    
    # -----------------------------------------------

    param = {
        "entry_start": entry_start,
        "entry_stop":  entry_stop,
        "args":        args
    }
    
    # =================================================================
    # *** BACKGROUND MC ***

    filename = args["MC_input"]['background']
    rootfile = io.glob_expand_files(datasets=filename, datapath=root_path)

    X_B, VARS = process_root(rootfile=rootfile, tree='Events', isMC=True, **param)
    Y_B = np.zeros(X_B.shape[0])


    # =================================================================
    # *** SIGNAL MC ***

    filename = args["MC_input"]['signal']
    rootfile = io.glob_expand_files(datasets=filename, datapath=root_path)

    X_S, VARS = process_root(rootfile=rootfile, tree='Events', isMC=True, **param)
    Y_S = np.ones(X_S.shape[0])
    
    
    # =================================================================
    # Finally combine

    X = np.concatenate((X_B, X_S), axis=0)
    Y = np.concatenate((Y_B, Y_S), axis=0)
    
    
    # ** Crucial -- randomize order to avoid problems with other functions **
    arr  = np.arange(X.shape[0])
    rind = np.random.shuffle(arr)

    X    = X[rind, ...].squeeze() # Squeeze removes additional [] dimension
    Y    = Y[rind].squeeze()
    
    # =================================================================
    # Custom treat specific variables

    """
    ind      = NEW_VARS.index('x_hlt_pms2')
    X[:,ind] = np.clip(a=np.asarray(X[:,ind]), a_min=-1e10, a_max=1e10)
    """
    
    return X, Y, VARS


def process_root(rootfile, tree, isMC, entry_start, entry_stop, args):

    CUTFUNC    = globals()[args['cutfunc']]
    FILTERFUNC = globals()[args['filterfunc']]

    X,ids      = iceroot.load_tree(rootfile=rootfile, tree=tree, entry_start=entry_start, entry_stop=entry_stop, ids=LOAD_VARS)
    
    # @@ Filtering done here @@
    ind = FILTERFUNC(X=X, ids=ids, isMC=isMC, xcorr_flow=args['xcorr_flow'])
    plots.plot_selection(X=X, ind=ind, ids=ids, args=args, label=f'<filter>_{isMC}', varlist=PLOT_VARS)
    cprint(__name__ + f'.process_root: isMC = {isMC} | <filterfunc> before: {len(X)}, after: {sum(ind)} events ', 'green')
    
    X   = X[ind]
    prints.printbar()


    # @@ Observable cut selections done here @@
    ind = CUTFUNC(X=X, ids=ids, isMC=isMC, xcorr_flow=args['xcorr_flow'])
    plots.plot_selection(X=X, ind=ind, ids=ids, args=args, label=f'<cutfunc>_{isMC}', varlist=PLOT_VARS)
    cprint(__name__ + f".process_root: isMC = {isMC} | <cutfunc>: before: {len(X)}, after: {sum(ind)} events \n", 'green')

    X   = X[ind]
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

    data = io.IceXYW(x=x, y=y, w=w, ids=ids)

    # -------------------------------------------------------------------------
    ### Pick kinematic variables out
    data_kin = None
    
    if KINEMATIC_ID is not None:
        k_ind, k_vars   = io.pick_vars(data, KINEMATIC_ID)
        
        data_kin     = copy.deepcopy(data)
        data_kin.x   = data.x[:, k_ind].astype(np.float)
        data_kin.ids = k_vars

    # -------------------------------------------------------------------------
    ## Graph representation
    data_graph  = None

    # -------------------------------------------------------------------------
    ## Tensor representation
    data_tensor = None

    # -------------------------------------------------------------------------
    ## Turn jagged to "long-vector" matrix representation

    ### Pick active scalar variables out
    scalar_ind, scalar_vars = io.pick_vars(data, globals()[args['inputvar_scalar']])
    jagged_ind, jagged_vars = io.pick_vars(data, globals()[args['inputvar_jagged']])
    
    jagged_maxdim = args['jagged_maxdim']*np.ones(len(jagged_vars), dtype=int)
    
    # Create tuplet expanded jagged variable names
    all_jagged_vars = []
    for i in range(len(jagged_vars)):
        for j in range(jagged_maxdim[i]):
            all_jagged_vars.append( f'{jagged_vars[i]}[{j}]' )
    
    # Update representation
    arg = {
        'scalar_vars'  :  scalar_ind,
        'jagged_vars'  :  jagged_ind,
        'jagged_maxdim':  jagged_maxdim,
        'library'      :  'np'
    }
    data.x   = aux.jagged2matrix(data.x, **arg)
    data.ids = scalar_vars + all_jagged_vars
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Create DeepSet style representation from the "long-vector" content
    data_deps = None

    if args['deps_on']:
        data_deps = copy.deepcopy(data)

        M = args['jagged_maxdim']      # Number of (jagged) tuplets per event
        D = len(jagged_ind)            # Tuplet feature vector dimension

        data_deps.x   = aux.longvec2matrix(X=data.x[:, len(scalar_ind):], M=M, D=D)
        data_deps.ids = all_jagged_vars
    # --------------------------------------------------------------------------


    return {'data': data, 'data_kin': data_kin, 'data_deps': data_deps, 'data_tensor': data_tensor, 'data_graph': data_graph}
