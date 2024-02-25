# Common input & data reading routines for the HLT electron trigger studies
# 
# Mikael Mieskolainen, 2023
# m.mieskolainen@imperial.ac.uk

import numpy as np
import copy
from importlib import import_module
from termcolor import colored, cprint

from icenet.tools import io
from icenet.tools import aux
from icenet.tools import prints
from icenet.tools import iceroot

# GLOBALS
from configs.trg.cuts import *
from configs.trg.filter import *


def load_root_file(root_path, ids=None, entry_start=0, entry_stop=None, maxevents=None, args=None):
    """ Loads the root files.
    
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
        root_path = root_path[0] # Remove [] list, we expect only the path here (no files)
    
    # -----------------------------------------------
    param = {
        'entry_start': entry_start,
        "entry_stop":  entry_stop,
        "maxevents":   maxevents,
        "args": args
    }

    # =================================================================
    # *** MC (signal) ***
    
    rootfile      = f'{root_path}/{args["mcfile"]}'
    
    # e1
    X_MC, VARS_MC = process_root(rootfile=rootfile, tree='tree', isMC='mode_e1', **param)
    
    X_MC_e1 = X_MC[:, [VARS_MC.index(name.replace("x_", "e1_")) for name in inputvars.NEW_VARS]]
    Y_MC_e1 = np.ones(X_MC_e1.shape[0], dtype=int)
    
    
    # e2
    X_MC, VARS_MC = process_root(rootfile=rootfile, tree='tree', isMC='mode_e2', **param)
    
    X_MC_e2 = X_MC[:, [VARS_MC.index(name.replace("x_", "e2_")) for name in inputvars.NEW_VARS]]
    Y_MC_e2 = np.ones(X_MC_e2.shape[0], dtype=int)
    
    
    # =================================================================
    # *** DATA (background) ***

    rootfile          = f'{root_path}/{args["datafile"]}'


    X_DATA, VARS_DATA = process_root(rootfile=rootfile, tree='tree', isMC='data', **param)

    X_DATA = X_DATA[:, [VARS_DATA.index(name.replace("x_", "")) for name in inputvars.NEW_VARS]]
    Y_DATA = np.zeros(X_DATA.shape[0], dtype=int)
    
    
    # =================================================================
    # Finally combine

    X = np.concatenate((X_MC_e1, X_MC_e2, X_DATA), axis=0)
    Y = np.concatenate((Y_MC_e1, Y_MC_e2, Y_DATA), axis=0)
    
    # Trivial weights
    W = np.ones(len(X))

    # ** Crucial -- randomize order to avoid problems with other functions **
    rand = np.random.permutation(len(X))
    X    = X[rand].squeeze() # Squeeze removes additional [] dimension
    Y    = Y[rand].squeeze()
    W    = W[rand].squeeze()
    
    # =================================================================
    # Custom treat specific variables

    ind      = inputvars.NEW_VARS.index('x_hlt_pms2')
    X[:,ind] = np.clip(a=np.asarray(X[:,ind]), a_min=-1e10, a_max=1e10)

    # TBD add info about cut stats etc
    info = {}
    
    return {'X':X, 'Y':Y, 'W':W, 'ids':inputvars.NEW_VARS, 'info':info}


def process_root(rootfile, tree, isMC, args, entry_start=0, entry_stop=None, maxevents=None):

    CUTFUNC    = globals()[args['cutfunc']]
    FILTERFUNC = globals()[args['filterfunc']]

    # Load files
    X,ids = iceroot.load_tree(rootfile=rootfile, tree=tree,
        entry_start=entry_start, entry_stop=entry_stop, maxevents=maxevents, ids=None, library='np')

    # @@ Filtering done here @@
    mask = FILTERFUNC(X=X, ids=ids, isMC=isMC, xcorr_flow=args['xcorr_flow'])
    #plots.plot_selection(X=X, mask=mask, ids=ids, plotdir=args['plotdir'], label=f'<filterfunc>_{isMC}', varlist=CUT_VARS)
    cprint(__name__ + f'.process_root: isMC = {isMC} | <filterfunc> before: {len(X)}, after: {sum(mask)} events ', 'green')
    
    X   = X[mask]
    prints.printbar()
    
    # @@ Observable cut selections done here @@
    mask = CUTFUNC(X=X, ids=ids, isMC=isMC, xcorr_flow=args['xcorr_flow'])
    #plots.plot_selection(X=X, mask=mask, ids=ids, plotdir=args['plotdir'], label=f'<cutfunc>_{isMC}', varlist=CUT_VARS)
    cprint(__name__ + f".process_root: isMC = {isMC} | <cutfunc>    before: {len(X)}, after: {sum(mask)} events \n", 'green')

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
    
    vars   = aux.process_regexp_ids(all_ids=ids, ids=eval('inputvars.' + args['inputvar_scalar']))
    data   = data[vars]
    data.x = data.x.astype(np.float32)
    
    return {'data': data, 'data_MI': data_MI, 'data_kin': data_kin, 'data_deps': data_deps, 'data_tensor': data_tensor, 'data_graph': data_graph}
