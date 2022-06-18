# Common input & data reading routines for the HLT electron trigger studies
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
from configs.trg.mvavars import *
from configs.trg.cuts import *
from configs.trg.filter import *


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
        'entry_start': entry_start,
        "entry_stop":  entry_stop,
        "args": args
    }

    # =================================================================
    # *** MC (signal) ***
    
    rootfile      = f'{root_path}/{args["mcfile"]}'
    
    # e1
    X_MC, VARS_MC = process_root(rootfile=rootfile, tree='tree', isMC='mode_e1', **param)
    
    X_MC_e1 = X_MC[:, [VARS_MC.index(name.replace("x_", "e1_")) for name in NEW_VARS]]
    Y_MC_e1 = np.ones(X_MC_e1.shape[0])
    
    
    # e2
    X_MC, VARS_MC = process_root(rootfile=rootfile, tree='tree', isMC='mode_e2', **param)
    
    X_MC_e2 = X_MC[:, [VARS_MC.index(name.replace("x_", "e2_")) for name in NEW_VARS]]
    Y_MC_e2 = np.ones(X_MC_e2.shape[0])
    
    
    # =================================================================
    # *** DATA (background) ***

    rootfile          = f'{root_path}/{args["datafile"]}'
    X_DATA, VARS_DATA = process_root(rootfile=rootfile, tree='tree', isMC='data', **param)

    X_DATA = X_DATA[:, [VARS_DATA.index(name.replace("x_", "")) for name in NEW_VARS]]
    Y_DATA = np.zeros(X_DATA.shape[0])
    
    
    # =================================================================
    # Finally combine

    X = np.concatenate((X_MC_e1, X_MC_e2, X_DATA), axis=0)
    Y = np.concatenate((Y_MC_e1, Y_MC_e2, Y_DATA), axis=0)


    # ** Crucial -- randomize order to avoid problems with other functions **
    arr  = np.arange(X.shape[0])
    rind = np.random.shuffle(arr)

    X    = X[rind, ...].squeeze() # Squeeze removes additional [] dimension
    Y    = Y[rind].squeeze()
    
    # =================================================================
    # Custom treat specific variables

    ind      = NEW_VARS.index('x_hlt_pms2')
    X[:,ind] = np.clip(a=np.asarray(X[:,ind]), a_min=-1e10, a_max=1e10)

    return X, Y, NEW_VARS


def process_root(rootfile, tree, isMC, args, entry_start=0, entry_stop=None):

    CUTFUNC    = globals()[args['cutfunc']]
    FILTERFUNC = globals()[args['filterfunc']]


    events  = uproot.open(f'{rootfile}:{tree}')
    ids     = events.keys()
    X,ids = iceroot.process_tree(events=events, ids=ids, entry_start=entry_start, entry_stop=entry_stop)


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
        k_ind, k_vars = io.pick_vars(data, KINEMATIC_ID)
        
        data_kin      = copy.deepcopy(data)
        data_kin.x    = data.x[:, k_ind].astype(np.float)
        data_kin.ids  = k_vars

    # -------------------------------------------------------------------------
    data_deps   = None

    # -------------------------------------------------------------------------
    data_tensor = None

    # -------------------------------------------------------------------------
    data_graph  = None

    # --------------------------------------------------------------------
    ### Finally pick active scalar variables out
    s_ind, s_vars = io.pick_vars(data, globals()[args['inputvar']])
    
    data.x   = data.x[:, s_ind].astype(np.float)
    data.ids = s_vars
    
    return {'data': data, 'data_kin': data_kin, 'data_deps': data_deps, 'data_tensor': data_tensor, 'data_graph': data_graph}
