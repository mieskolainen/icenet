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


def init(MAXEVENTS=None):
    """ Initialize the input data.
    
    Args:
        Implicit commandline and yaml file input.
    
    Returns:
        jagged array data, arguments
    """
    
    args, cli = process.read_config('./configs/dqcd')
    features  = globals()[args['imputation_param']['var']]
    
    
    ### SET random seed
    print(__name__ + f'.init: Setting random seed: {args["rngseed"]}')
    np.random.seed(args['rngseed'])

    # --------------------------------------------------------------------
    ### SET GLOBALS (used only in this file)
    global ARGS
    ARGS = args

    if MAXEVENTS is not None:
        ARGS['MAXEVENTS'] = MAXEVENTS
    
    print(__name__ + f'.init: inputvar   =  {args["inputvar"]}')
    print(__name__ + f'.init: cutfunc    =  {args["cutfunc"]}')
    #print(__name__ + f'.init: targetfunc =  {args["targetfunc"]}')
    # --------------------------------------------------------------------

    ### Load data

    # Background (0) and signal (1)
    class_id = [0,1]
    data     = io.DATASET(func_loader=load_root_file, files=args['root_files'], class_id=class_id, frac=args['frac'], rngseed=args['rngseed'])
    

    # @@ Imputation @@
    if args['imputation_param']['active']:

        special_values = args['imputation_param']['values'] # possible special values
        print(__name__ + f': Imputing data for special values {special_values} for variables in <{args["imputation_param"]["var"]}>')

        # Choose active dimensions
        dim = np.array([i for i in range(len(data.ids)) if data.ids[i] in features], dtype=int)

        # Parameters
        param = {
            "dim":        dim,
            "values":     special_values,
            "labels":     data.ids,
            "algorithm":  args['imputation_param']['algorithm'],
            "fill_value": args['imputation_param']['fill_value'],
            "knn_k":      args['imputation_param']['knn_k']
        }
        
        # NOTE, UPDATE NEEDED: one should save here 'imputer_trn' to a disk -> can be used with data
        data.trn.x, imputer_trn = io.impute_data(X=data.trn.x, imputer=None,        **param)
        data.tst.x, _           = io.impute_data(X=data.tst.x, imputer=imputer_trn, **param)
        data.val.x, _           = io.impute_data(X=data.val.x, imputer=imputer_trn, **param)
        
    else:
        True
        # No imputation, but fix spurious NaN / Inf
        #data.trn.x[np.logical_not(np.isfinite(data.trn.x))] = 0
        #data.val.x[np.logical_not(np.isfinite(data.val.x))] = 0
        #data.tst.x[np.logical_not(np.isfinite(data.tst.x))] = 0

    cprint(__name__ + f""".common: Process RAM usage: {io.process_memory_use():0.2f} GB 
        [total RAM in use: {psutil.virtual_memory()[2]} %]""", 'red')
    
    return data, args, features


def load_root_file(root_path, ids=None, entrystart=0, entrystop=None, class_id = [], args=None):
    """ Loads the root file with signal events from MC and background from DATA.
    
    Args:
        root_path : paths to root files
        class_id  : class ids
    
    Returns:
        X,Y       : input, output matrices
        ids       : variable names
    """
    
    # -----------------------------------------------
    # ** GLOBALS **

    if args is None:
        args = ARGS

    if entrystop is None:
        entrystop = args['MAXEVENTS']

    # -----------------------------------------------

    param = {
        "entry_start": entrystart,
        "entry_stop":  entrystop,
        "args": args
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

    Y   = iceroot.load_tree(rootfile=rootfile, tree=tree, entry_start=entry_start, entry_stop=entry_stop, ids=LOAD_VARS)
    ids = [i for i in Y.keys()]
    X   = np.empty((len(Y[ids[0]]), len(ids)), dtype=object) 
    for i in range(len(ids)):
        X[:,i] = Y[ids[i]]
    
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


def splitfactor(data, args):
    """
    Split electron ID data into different datatypes.
    
    Args:
        data:  jagged arrays
        args:  arguments dictionary
    
    Returns:
        scalar (vector) data
        kinematic data
    """
    
    ### Pick kinematic variables out
    if KINEMATIC_ID is not None:
        k_ind, k_vars   = io.pick_vars(data, KINEMATIC_ID)
        
        data_kin        = copy.deepcopy(data)
        data_kin.trn.x  = data.trn.x[:, k_ind].astype(np.float)
        data_kin.val.x  = data.val.x[:, k_ind].astype(np.float)
        data_kin.tst.x  = data.tst.x[:, k_ind].astype(np.float)
        data_kin.ids    = k_vars

    ### Pick active scalar variables out
    s_ind, s_vars = io.pick_vars(data, globals()[args['inputvar']])
    
    data.trn.x    = data.trn.x[:, s_ind].astype(np.float)
    data.val.x    = data.val.x[:, s_ind].astype(np.float)
    data.tst.x    = data.tst.x[:, s_ind].astype(np.float)
    data.ids      = s_vars

    return data, data_kin
