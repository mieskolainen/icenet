# Common input & data reading routines for the electron ID
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import uproot
from importlib import import_module
import time
import ray
import os

from icenet.tools import io, aux, prints, iceroot
from iceid import graphio

# ------------------------------------------
from icenet import print
# ------------------------------------------

# Globals
from configs.eid.mctargets import *
from configs.eid.mcfilter  import *
from configs.eid.cuts import *


def load_root_file(root_path, ids=None, entry_start=0, entry_stop=None, maxevents=None, args=None, library='np'):
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
    
    if type(root_path) is not list:
        root_path = [root_path] # Make sure it is a list, even if one file only
    
    # -----------------------------------------------
    CUTFUNC    = globals()[args['cutfunc']]
    TARFUNC    = globals()[args['targetfunc']]
    FILTERFUNC = globals()[args['filterfunc']]
    # -----------------------------------------------

    print('')
    print(f'Loading root file {root_path}', 'yellow')
    
    # Check is it MC (based on the first file and first event)
    file   = uproot.open(root_path[0])
    events = file[args['tree_name']]
    isMC   = bool(events.arrays('is_mc')[0]['is_mc'])

    # --------------------------------------------------------------
    # Load all files

    LOAD_VARS = inputvars.LOAD_VARS # Which variables do we read
    
    X,ids = iceroot.load_tree(rootfile=root_path, tree=args['tree_name'],
        entry_start=entry_start, entry_stop=entry_stop, maxevents=maxevents, ids=LOAD_VARS, library=library,
        num_cpus=args['num_cpus'])
    Y = None
    # --------------------------------------------------------------
    
    print(f'X.shape = {X.shape}')
    io.showmem()
    prints.printbar()

    # =================================================================
    # *** MC ONLY ***

    if isMC:

        # @@ MC class target definition here @@
        print(f'Computing MC <targetfunc> ...', 'yellow')
        Y = TARFUNC(X=X, ids=ids, xcorr_flow=args['xcorr_flow']).astype(np.int32)
        print(__name__ + f'Y.shape = {Y.shape}')
        
        # @@ MC filtering done here @@
        print(f'Computing MC <filterfunc> ...', 'yellow')
        mask_mc = FILTERFUNC(X=X, ids=ids, xcorr_flow=args['xcorr_flow'])
        print(f'<filterfunc> | before: {len(X)}, after: {sum(mask_mc)} events', 'green')
        prints.printbar()
        
        X = X[mask_mc]
        Y = Y[mask_mc].squeeze() # Remove useless dimension
    
    # =================================================================

    # @@ Observable cut selections done here @@
    print(f'Computing <cutfunc> ...', 'yellow')
    cmask = CUTFUNC(X=X, ids=ids, xcorr_flow=args['xcorr_flow'])
    print(f"<cutfunc> | before: {len(X)}, after: {np.sum(cmask)} events \n", 'green')
    
    X = X[cmask]
    if isMC: Y = Y[cmask]
    
    io.showmem()
    prints.printbar()
    file.close()

    # Trivial weights
    W = np.ones(len(X))

    # TBD add cut statistics etc. info here
    info = {}
    
    # ** Crucial -- randomize order to avoid problems with other functions **
    rand = np.random.permutation(len(X))
    X    = X[rand].squeeze() # Squeeze removes additional [] dimension
    Y    = Y[rand].squeeze()
    W    = W[rand].squeeze()

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
    
    ### Pick active variables out
    scalar_vars = aux.process_regexp_ids(all_ids=ids, ids=eval('inputvars.' + args['inputvar_scalar']))

    if args['inputvar_image'] is not None:
        image_vars = aux.process_regexp_ids(all_ids=ids, ids=eval('inputvars.' + args['inputvar_image']))
    else:
        image_vars = None

    # -------------------------------------------------------------------------
    ### Pick kinematic variables out
    data_kin = None
    
    if inputvars.KINEMATIC_VARS is not None:

        vars       = aux.process_regexp_ids(all_ids=data.ids, ids=inputvars.KINEMATIC_VARS)
        data_kin   = data[vars]
        data_kin.x = data_kin.x.astype(np.float32)
    
    # -------------------------------------------------------------------------
    ### MI variables
    data_MI = None
    
    # -------------------------------------------------------------------------
    ### DeepSets representation
    data_deps = None
    
    # -------------------------------------------------------------------------
    ### Tensor representation
    data_tensor = None
    
    if image_vars is not None:
        data_tensor = graphio.parse_tensor_data(X=data.x, ids=ids, image_vars=image_vars, args=args)
    
    # -------------------------------------------------------------------------
    ## Graph representation
    data_graph = None

    if args['graph_param']['num_workers'] == 1:
        
        data_graph = graphio.parse_graph_data(X=data.x, ids=data.ids, features=scalar_vars,
            graph_param=args['graph_param'], Y=data.y, weights=data.w, entry_start=None, entry_stop=None)
    
    else:
        
        # Parallel processing of graph objects with Ray
        
        start_time  = time.time()
        
        num_workers = args['graph_param']['num_workers']
        chunk_ind   = aux.split_start_end(range(len(data.x)), num_workers)
        print(chunk_ind)

        data_graph = []
        job_index  = 0
        
        ray.init(num_cpus=num_workers, _temp_dir=f'{os.getcwd()}/tmp/')
        
        graph_futures = []
        obj_ref       = ray.put(data.x) # ** Ray seems not able to handle numpy(object) array without copy ..

        for _ in range(num_workers):
            
            entry_start, entry_stop = chunk_ind[job_index][0], chunk_ind[job_index][-1]
            
            graph_futures.append( \
                graphio.parse_graph_data_ray.remote( \
                    obj_ref, data.ids, scalar_vars, args['graph_param'],
                    data.y, data.w, entry_start, entry_stop)
            )

            job_index += 1

        data_graph += sum(ray.get(graph_futures), []) # Join split array results
        ray.shutdown()

        print(f'ray_results: {time.time() - start_time:0.1f} sec')
        io.showmem()
    
    # --------------------------------------------------------------------
    ### Finally pick active scalar variables out
    
    vars   = aux.process_regexp_ids(all_ids=data.ids, ids=scalar_vars)
    data   = data[vars]
    data.x = data.x.astype(np.float32)
    
    return {'data':        data,
            'data_MI':     data_MI,
            'data_kin':    data_kin,
            'data_deps':   data_deps,
            'data_tensor': data_tensor,
            'data_graph':  data_graph}
