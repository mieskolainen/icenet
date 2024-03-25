# Common input & data reading routines for the DQCD analysis
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import copy
import os

import ray
from tqdm import tqdm

import time
import multiprocessing
from importlib import import_module

from termcolor import colored, cprint

from icenet.tools import io
from icenet.tools import aux
from icenet.tools import prints
from icenet.tools import iceroot
from icenet.algo import analytic
from icefit import dequantize
from icedqcd import graphio

# GLOBALS
from configs.dqcd.cuts   import *
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
    inputvars = import_module("configs." + args["rootname"] + "." + args["inputvars"])
    
    if type(root_path) is list:
        root_path = root_path[0] # Remove [] list
    
    # -----------------------------------------------

    param = {
        "tree":        "Events",
        "entry_start": entry_start,
        "entry_stop":  entry_stop,
        "maxevents":   maxevents,
        "args":        args,
        "load_ids":    inputvars.LOAD_VARS
    }
    
    INFO = {}
    X    = {}
    Y    = {}
    W    = {}
    
    # =================================================================
    # *** BACKGROUND MC, SIGNAL MC, DOMAIN ADAPTATION ... ***
    
    for key in args["input"].keys(): # input from yamlgen generated yml
        
        class_id = int(key.split("_")[1])
        proc     = args["input"][key] 
        
        X[key], Y[key], W[key], ind, INFO[key] = iceroot.read_multiple(class_id=class_id,
            process_func=process_root, processes=proc, root_path=root_path, param=param)
    
    # =================================================================
    # Sample conditional theory parameters as they are distributed in signal sample
    
    sig_class = "class_1" # fixed here
    p = ak.to_numpy(W[sig_class] / ak.sum(W[sig_class])).squeeze() # probability per event entry

    for key in X.keys(): # Loop over classes
        if key != sig_class:
            
            # We pick all variables at once
            # (so N-dim random variable N-tuplets are sampled 1-to-1 as in signal class)
            var = inputvars.MODEL_VARS
            
            # Reshape into numpy array
            N = len(X[sig_class])
            A = np.zeros((N, len(var)), dtype=np.float32)
            for i in range(len(var)):
                A[:,i] = ak.to_numpy(X[sig_class][var[i]]).squeeze()
            
            # Sample
            rng = np.random.default_rng()
            new = rng.choice(A, axis=0, size=len(X[key]), p=p, replace=True)
            
            print(__name__ + f'.load_root_file: Sampling theory conditional parameter "{var}" for "{key}"')
            
            # Set conditional variables 'MODEL_'
            for i in range(len(var)):
                X[key][var[i]] = ak.Array(new[:,i])
            
            # "Mirror" copy variables to 'GEN_' (for ROC plots etc. in the evaluation stage)
            var_new = [s.replace('MODEL', 'GEN') for s in var]
            for i in range(len(var)):
                X[key][var_new[i]] = ak.Array(new[:,i])
    
    # =================================================================
    # *** Finally combine ***
    
    X = ak.concatenate(X.values(), axis=0)
    Y = ak.concatenate(Y.values(), axis=0)
    W = ak.concatenate(W.values(), axis=0)
    
    # ** Crucial -- randomize order to avoid problems with other functions **
    rand = np.random.permutation(len(X))
    
    X    = X[rand]
    Y    = Y[rand]
    W    = W[rand]

    print(__name__ + f'.common.load_root_file: Event counts per class')
    unique, counts = np.unique(Y, return_counts=True)
    print(np.asarray((unique, counts)).T)
    
    return {'X':X, 'Y':Y, 'W':W, 'ids':ids, 'info': INFO}


def process_root(X, args, ids=None, isMC=None, return_mask=False, class_id=None, **kwargs):
    """
    Apply selections
    """

    FILTERFUNC = globals()[args['filterfunc']]    
    CUTFUNC    = globals()[args['cutfunc']]
    
    stats = {'filterfunc': None, 'cutfunc': None}
    
    # @@ Filtering done here @@
    fmask = FILTERFUNC(X=X, isMC=isMC, class_id=class_id, xcorr_flow=args['xcorr_flow'])
    stats['filterfunc'] = {'before': len(X), 'after': sum(fmask)}
    
    #plots.plot_selection(X=X, mask=mask, ids=ids, plotdir=args['plotdir'], label=f'<filterfunc>_{isMC}', varlist=CUT_VARS, library='ak')
    cprint(__name__ + f'.process_root: isMC = {isMC} | <filterfunc>  before: {len(X)}, after: {sum(fmask)} events ({sum(fmask)/(len(X)+1E-12):0.6f})', 'green')
    prints.printbar()
    
    X_new = X[fmask]
    
    # @@ Observable cut selections done here @@
    cmask = CUTFUNC(X=X_new, xcorr_flow=args['xcorr_flow'])
    stats['cutfunc'] = {'before': len(X_new), 'after': sum(cmask)}
    
    #plots.plot_selection(X=X, mask=mask, ids=ids, plotdir=args['plotdir'], label=f'<cutfunc>_{isMC}', varlist=CUT_VARS, library='ak')
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

def splitfactor(x, y, w, ids, args, skip_graph=True, use_dequantize=True):
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

    if data.y is not None:
        data.y = ak.to_numpy(data.y).astype(np.float32)
    
    if data.w is not None:
        data.w = ak.to_numpy(data.w).astype(np.float32)
    
    # -------------------------------------------------------------------------
    
    ### Pick active variables out
    scalar_vars = aux.process_regexp_ids(all_ids=aux.unroll_ak_fields(x=x, order='first'),  ids=eval('inputvars.' + args['inputvar_scalar']))
    jagged_vars = aux.process_regexp_ids(all_ids=aux.unroll_ak_fields(x=x, order='second'), ids=eval('inputvars.' + args['inputvar_jagged']))
    
    # Individually for GNNs
    muon_vars   = aux.process_regexp_ids(all_ids=aux.unroll_ak_fields(x=x, order='second'), ids=inputvars.MVA_MUON_VARS)
    jet_vars    = aux.process_regexp_ids(all_ids=aux.unroll_ak_fields(x=x, order='second'), ids=inputvars.MVA_JET_VARS)
    sv_vars     = aux.process_regexp_ids(all_ids=aux.unroll_ak_fields(x=x, order='second'), ids=inputvars.MVA_SV_VARS)

    muonsv_vars = aux.process_regexp_ids(all_ids=aux.unroll_ak_fields(x=x, order='second'), ids=inputvars.MVA_MUONSV_VARS)
    cpf_vars    = aux.process_regexp_ids(all_ids=aux.unroll_ak_fields(x=x, order='second'), ids=inputvars.MVA_CPF_VARS)
    npf_vars    = aux.process_regexp_ids(all_ids=aux.unroll_ak_fields(x=x, order='second'), ids=inputvars.MVA_NPF_VARS)
    pf_vars     = aux.process_regexp_ids(all_ids=aux.unroll_ak_fields(x=x, order='second'), ids=inputvars.MVA_PF_VARS)


    # -------------------------------------------------------------------------
    ### ** Conditional variables **
    
    if args['use_conditional'] == False:
        for var in inputvars.MODEL_VARS:
            try:
                scalar_vars.remove(var)
                cprint(__name__ + f'.splitfactor: Removing model conditional var "{var}"" from scalar_vars', 'red')
            except:
                continue
    else:
        
        # Dequantization for training (turn off for deployment)
        if args['conditional_param']['dequantize_common'] and use_dequantize:
            
            n_interp = args['conditional_param']['n_interp']
            kind     = args['conditional_param']['kind']
            
            for var in inputvars.MODEL_VARS:
                
                cprint(__name__ + f'.split_factor: Dequantizing conditional "{var}" with iDQF', 'yellow')
                data.x[var] = dequantize.iDQF(x=ak.to_numpy(ak.ravel(data.x[var])), n_interp=n_interp, kind=kind)
    
    # -------------------------------------------------------------------------
    ## ** Collection filter **
    
    for d in args['jagged_filter']:
        
        expr = 'data.x.' + d['condition'].strip() # strip to remove leading/trailing spaces
        cprint(__name__ + f'.splitfactor: Filtering collection {d} with {expr}', 'yellow')
        
        filter_ind = eval(expr)
        data.x[d['name']] = data.x[d['name']][filter_ind]
    
    # -------------------------------------------------------------------------
    ## ** Collection entry re-ordering sort **
    
    for d in args['jagged_order']:

        cprint(__name__ + f'.splitfactor: Collection re-ordering {d}', 'yellow')
        
        sort_ind = ak.argsort(data.x[d['name']][d['var']], ascending=d['ascending'])
        data.x[d['name']] = data.x[d['name']][sort_ind]
        
    # -------------------------------------------------------------------------
    ## ** Custom variables added to collections **
    
    ## \DeltaR
    data.x['muonSV', 'deltaR'] = \
        analytic.deltaR(x=data.x['muonSV'], eta1='mu1eta', eta2='mu2eta', phi1='mu1phi', phi2='mu2phi')
    jagged_vars.append('muonSV_deltaR')
    muonsv_vars.append('muonSV_deltaR')
    
    ## Invariant Mass
    data.x['muonSV', 'mass'] = \
        analytic.invmass(x=data.x['muonSV'], pt1='mu1pt', pt2='mu2pt',
                         eta1='mu1eta', eta2='mu2eta', phi1='mu1phi', phi2='mu2phi',
                         m1_const=0.1057, m2_const=0.1057)
    
    jagged_vars.append('muonSV_mass')
    muonsv_vars.append('muonSV_mass')
    
    print(data.x['muonSV'].fields)
    
    # -------------------------------------------------------------------------
    ### Pick kinematic variables out
    data_kin = None
    
    if inputvars.KINEMATIC_VARS is not None:
        
        kinematic_vars = aux.process_regexp_ids(all_ids=aux.unroll_ak_fields(x=x, order='first'),
                                                ids=inputvars.KINEMATIC_VARS)

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
        num_workers    = multiprocessing.cpu_count() // 2
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
                        obj_ref, data.ids, scalar_vars, node_features, args['graph_param'],
                        data.y, data.w, entry_start, entry_stop)
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
    
    data.x, data.ids = aux.jagged_ak_to_numpy(arr=data.x, scalar_vars=scalar_vars,
                        jagged_vars=jagged_vars, jagged_maxdim=args['jagged_maxdim'],
                        null_value=args['imputation_param']['fill_value'])
    io.showmem()
    
    # -------------------------------------------------------------------------
    # Mutual information regularization targets
    
    vars = aux.process_regexp_ids(all_ids=data.ids, ids=inputvars.MI_VARS)
    data_MI = data[vars].x.astype(np.float32)
    
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
    
    
    return {'data': data, 'data_MI': data_MI, 'data_kin': data_kin, 'data_deps': data_deps, 'data_tensor': data_tensor, 'data_graph': data_graph}
