# Common input & data reading routines for the DQCD analysis
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

import numpy as np
import psutil
import copy
import os

import ray
from ray.actor import ActorHandle
from tqdm import tqdm

import time
import multiprocessing

from termcolor import colored, cprint

from icenet.tools import raytools
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


def load_root_file(root_path, ids=None, entry_start=0, entry_stop=None, maxevents=None, args=None):
    """ Loads the root files
    
    Args:
        root_path: path to root files
    
    Returns:
        X:     jagged columnar data
        Y:     class labels
        W:     event weights
        ids:   columnar variables string (list)
        info:  trigger, MC xs, pre-selection acceptance x efficiency information (dict)
    """

    if type(root_path) is list:
        root_path = root_path[0] # Remove [] list
    
    # -----------------------------------------------

    param = {
        "tree":        "Events",
        "entry_start": entry_start,
        "entry_stop":  entry_stop,
        "maxevents":    maxevents,
        "args":        args,
        "load_ids":    LOAD_VARS,
        "isMC":        True
    }

    INFO = {'class_0': None, 'class_1': None}

    # =================================================================
    # *** SIGNAL MC ***
    
    proc = args["input"]['class_1']
    X_S, Y_S, W_S, ind, INFO['class_1'] = iceroot.read_multiple_MC(class_id=1,
        process_func=process_root, processes=proc, root_path=root_path, param=param)
    
    # =================================================================
    # *** BACKGROUND MC ***
    
    proc = args["input"]['class_0']
    X_B, Y_B, W_B, ind, INFO['class_0'] = iceroot.read_multiple_MC(class_id=0,
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
    rand = np.random.permutation(len(X))
    
    X    = X[rand]
    Y    = Y[rand]
    W    = W[rand]

    print(__name__ + f'.common.load_root_file: len(X) = [{len(X[Y==0])}, {len(X[Y==1])}]')
    
    return {'X':X, 'Y':Y, 'W':W, 'ids':ids, 'info': INFO}


def process_root(X, args, ids=None, isMC=None, return_mask=False, **kwargs):
    """
    Apply selections
    """

    FILTERFUNC = globals()[args['filterfunc']]    
    CUTFUNC    = globals()[args['cutfunc']]
    
    stats = {'filterfunc': None, 'cutfunc': None}
    
    # @@ Filtering done here @@
    fmask = FILTERFUNC(X=X, isMC=isMC, xcorr_flow=args['xcorr_flow'])
    stats['filterfunc'] = {'before': len(X), 'after': sum(fmask)}
    
    #plots.plot_selection(X=X, mask=mask, ids=ids, plotdir=args['plotdir'], label=f'<filterfunc>_{isMC}', varlist=PLOT_VARS, library='ak')
    cprint(__name__ + f'.process_root: isMC = {isMC} | <filterfunc>  before: {len(X)}, after: {sum(fmask)} events ({sum(fmask)/(len(X)+1E-12):0.6f})', 'green')
    prints.printbar()
    
    X_new = X[fmask]
    
    # @@ Observable cut selections done here @@
    cmask = CUTFUNC(X=X_new, xcorr_flow=args['xcorr_flow'])
    stats['cutfunc'] = {'before': len(X_new), 'after': sum(cmask)}
    
    #plots.plot_selection(X=X, mask=mask, ids=ids, plotdir=args['plotdir'], label=f'<cutfunc>_{isMC}', varlist=PLOT_VARS, library='ak')
    cprint(__name__ + f".process_root: isMC = {isMC} | <cutfunc>     before: {len(X_new)}, after: {sum(cmask)} events ({sum(cmask)/(len(X_new)+1E-12):0.6f}) \n", 'green')
    prints.printbar()
    io.showmem()
    
    X_final = X_new[cmask]

    if return_mask == False:
        return X_final, ids, stats
    else:
        fmask_np = fmask.to_numpy()
        fmask_np[fmask_np] = cmask # cmask is evaluated for which fmask == True
        
        return fmask_np


def splitfactor(x, y, w, ids, args, skip_graph=False):
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
    
    if not skip_graph:
        
        #node_features = {'muon': muon_vars, 'jet': jet_vars, 'cpf': cpf_vars, 'npf': npf_vars, 'sv': sv_vars}
        node_features = {'muon': muon_vars, 'jet': jet_vars, 'sv': sv_vars}
        
        """
        start_time = time.time()
        data_graph_ = graphio.parse_graph_data(X=data.x, Y=data.y, weights=data.w, ids=data.ids, 
            features=scalar_vars, node_features=node_features, graph_param=args['graph_param'])

        print(f'single_results: {time.time()-start_time:0.1f} sec')
        """
        
        ## ------------------------------------------
        # Parallel processing of graph objects with Ray
        
        start_time  = time.time()
        
        big_chunk_size = 10000
        num_workers    = multiprocessing.cpu_count()
        big_chunks     = int(np.ceil(len(data.x) / big_chunk_size))
        
        chunk_ind   = aux.split_start_end(range(len(data.x)), num_workers * big_chunks)
        print(chunk_ind)

        data_graph = []
        job_index  = 0
        for _ in tqdm(range(big_chunks)):

            ray.init(num_cpus=num_workers, _temp_dir=f'{os.getcwd()}/tmp/')

            graph_futures = []
            obj_ref       = ray.put(data.x)

            for _ in range(num_workers):
                
                entry_start, entry_stop = chunk_ind[job_index][0], chunk_ind[job_index][-1]
                
                graph_futures.append( \
                    graphio.parse_graph_data_ray.remote( \
                        obj_ref, data.ids, scalar_vars, node_features, args['graph_param'], data.y, data.w, entry_start, entry_stop)
                )

                job_index += 1

            data_graph += sum(ray.get(graph_futures), []) # Join split array results
            ray.shutdown()

        print(f'ray_results: {time.time() - start_time:0.1f} sec')
        io.showmem()

    # -------------------------------------------------------------------------
    ## Tensor representation
    data_tensor = None
    
    # -------------------------------------------------------------------------
    ## Turn jagged data to a "long-vector" zero-padded matrix representation
    
    data = aux.jagged_ak_to_numpy(data=data, scalar_vars=scalar_vars,
                       jagged_vars=jagged_vars, jagged_maxdim=args['jagged_maxdim'],
                       null_value=args['imputation_param']['fill_value'])
    io.showmem()

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
