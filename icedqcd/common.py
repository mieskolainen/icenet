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
        root_path: paths to root files
    
    Returns:
        X,Y      : data matrices
        ids      : variable names
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

    INFO = {'class_1': None, 'class_0': None}

    # =================================================================
    # *** SIGNAL MC *** (first signal, so we can use it's theory conditional parameters)

    proc = args["input"]['class_1']
    X_S, Y_S, W_S, VARS_S, INFO['class_1'] = iceroot.read_multiple_MC(class_id=1,
        process_func=process_root, processes=proc, root_path=root_path, param=param)
    
    
    # =================================================================
    # *** BACKGROUND MC ***
    
    proc = args["input"]['class_0']
    X_B, Y_B, W_B, VARS_B, INFO['class_0'] = iceroot.read_multiple_MC(class_id=0,
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
    
    return X, Y, W, VARS_S, INFO


def process_root(X, ids, isMC, args, **extra):
    
    FILTERFUNC = globals()[args['filterfunc']]    
    CUTFUNC    = globals()[args['cutfunc']]
    
    stats = {'filterfunc': None, 'cutfunc': None}
    
    # @@ Filtering done here @@
    mask = FILTERFUNC(X=X, isMC=isMC, xcorr_flow=args['xcorr_flow'])
    stats['filterfunc'] = {'before': len(X), 'after': sum(mask)}
    
    plots.plot_selection(X=X, mask=mask, ids=ids, plotdir=args['plotdir'], label=f'<filterfunc>_{isMC}', varlist=PLOT_VARS, library='ak')
    cprint(__name__ + f'.process_root: isMC = {isMC} | <filterfunc>  before: {len(X)}, after: {sum(mask)} events ({sum(mask)/len(X):0.6f})', 'green')
    X = X[mask]
    prints.printbar()
    
    # @@ Observable cut selections done here @@
    mask = CUTFUNC(X=X, xcorr_flow=args['xcorr_flow'])
    stats['cutfunc'] = {'before': len(X), 'after': sum(mask)}
    
    plots.plot_selection(X=X, mask=mask, ids=ids, plotdir=args['plotdir'], label=f'<cutfunc>_{isMC}', varlist=PLOT_VARS, library='ak')
    cprint(__name__ + f".process_root: isMC = {isMC} | <cutfunc>     before: {len(X)}, after: {sum(mask)} events ({sum(mask)/len(X):0.6f}) \n", 'green')
    X = X[mask]
    prints.printbar()
    
    io.showmem()
    
    return X, ids, stats


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

    ### Pick active variables out
    scalar_vars = aux.process_regexp_ids(all_ids=aux.unroll_ak_fields(x=x, order='first'),  ids=globals()[args['inputvar_scalar']])
    jagged_vars = aux.process_regexp_ids(all_ids=aux.unroll_ak_fields(x=x, order='second'), ids=globals()[args['inputvar_jagged']])
    
    # Individually
    muon_vars = aux.process_regexp_ids(all_ids=aux.unroll_ak_fields(x=x, order='second'), ids=MVA_MUON_VARS)
    jet_vars  = aux.process_regexp_ids(all_ids=aux.unroll_ak_fields(x=x, order='second'), ids=MVA_JET_VARS)
    sv_vars   = aux.process_regexp_ids(all_ids=aux.unroll_ak_fields(x=x, order='second'), ids=MVA_SV_VARS)
    cpf_vars  = aux.process_regexp_ids(all_ids=aux.unroll_ak_fields(x=x, order='second'), ids=MVA_CPF_VARS)
    npf_vars  = aux.process_regexp_ids(all_ids=aux.unroll_ak_fields(x=x, order='second'), ids=MVA_NPF_VARS)
    pf_vars   = aux.process_regexp_ids(all_ids=aux.unroll_ak_fields(x=x, order='second'), ids=MVA_PF_VARS)

    ### ** Remove conditional variables **
    if args['use_conditional'] == False:
        for var in globals()['MODEL_VARS']:
            try:
                scalar_vars.remove(var)
                print(__name__ + f'.splitfactor: Removing model conditional var "{var}"" from scalar_vars')
            except:
                continue

    # -------------------------------------------------------------------------
    ### Pick kinematic variables out
    data_kin = None
    
    if KINEMATIC_VARS is not None:

        kinematic_vars = aux.process_regexp_ids(all_ids=aux.unroll_ak_fields(x=x, order='first'), ids=KINEMATIC_VARS)

        data_kin       = copy.deepcopy(data)
        data_kin.x     = aux.ak2numpy(x=data.x, fields=kinematic_vars)
        data_kin.ids   = kinematic_vars

    # -------------------------------------------------------------------------
    ## Graph representation
    data_graph = None

    #node_features = {'muon': muon_vars, 'jet': jet_vars, 'cpf': cpf_vars, 'npf': npf_vars, 'sv': sv_vars}
    node_features = {'muon': muon_vars, 'jet': jet_vars}
    
    data_graph = graphio.parse_graph_data(X=data.x, Y=data.y, weights=data.w, ids=data.ids, 
        features=scalar_vars, node_features=node_features, graph_param=args['graph_param'])

    # -------------------------------------------------------------------------
    ## Tensor representation
    data_tensor = None

    # -------------------------------------------------------------------------
    ## Turn jagged data to a "long-vector" zero-padded matrix representation

    data = aux.jagged_ak_to_numpy(data=data, scalar_vars=scalar_vars,
                       jagged_vars=jagged_vars, jagged_maxdim=args['jagged_maxdim'],
                       null_value=args['imputation_param']['fill_value'])

    # --------------------------------------------------------------------------
    # Create DeepSet style representation from the "long-vector" content
    data_deps = None
    
    """
    ## ** TBD. This should be generalized to handle multiple different length sets **
    
    data_deps = copy.deepcopy(data)

    M = len(jagged_vars)              # Number of (jagged) variables per event
    D = args['jagged_maxdim']['sv']   # Tuplet feature vector dimension
    data_deps.x   = aux.longvec2matrix(X=data.x[:, len(scalar_vars):], M=M, D=D)
    data_deps.y   = data_deps.y
    data_deps.ids = all_jagged_vars
    """
    # --------------------------------------------------------------------------
    
    return {'data': data, 'data_kin': data_kin, 'data_deps': data_deps, 'data_tensor': data_tensor, 'data_graph': data_graph}
