# Common input & data reading routines for the electron ID
#
# m.mieskolainen@imperial.ac.uk, 2024

import copy
import numpy as np
import uproot
from importlib import import_module
from termcolor import colored, cprint
import multiprocessing
import time
from tqdm import tqdm
import ray
import os
import awkward as ak

from icenet.tools import io
from icenet.tools import aux
from icenet.tools import prints
from icenet.tools import iceroot

# Globals
from configs.lptele.mctargets import *
from configs.lptele.mcfilter  import *
from configs.lptele.filter  import *
from configs.lptele.cuts import *

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

    # ----------
        
    param = {
        "tree":        "ntuplizer/tree",
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

    # ----------

    for key in args["input"].keys(): # input from yamlgen generated yml
        class_id = int(key.split("_")[1])
        proc     = args["input"][key] 
        
        X[key], Y[key], W[key], ind, INFO[key] = iceroot.read_multiple(
            class_id=class_id,
            process_func=process_root,
            processes=proc,
            root_path=root_path,
            param=param)
        
    X = ak.concatenate(X.values(), axis=0)
    Y = ak.concatenate(Y.values(), axis=0)
    W = ak.concatenate(W.values(), axis=0)
    
    rand = np.random.permutation(len(X)) # Randomize order, crucial!
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
    
    cprint(__name__ + f'.process_root: isMC = {isMC} | <filterfunc>  before: {len(X)}, after: {sum(fmask)} events ({sum(fmask)/(len(X)+1E-12):0.6f})', 'green')
    prints.printbar()
    
    X_new = X[fmask]
    
    # @@ Observable cut selections done here @@
    cmask = CUTFUNC(X=X_new, xcorr_flow=args['xcorr_flow'])
    stats['cutfunc'] = {'before': len(X_new), 'after': sum(cmask)}
    
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

def splitfactor(x, y, w, ids, args):
    """
    Transform data into different datatypes.
    
    Args:
        data:  jagged arrays
        args:  arguments dictionary
    
    Returns:
        dictionary with different data representations
    """

    # ----------
    # Init
    inputvars = import_module("configs." + args["rootname"] + "." + args["inputvars"])
    
    data = io.IceXYW(x=x, y=y, w=w, ids=ids)

    if data.y is not None:
        data.y = ak.to_numpy(data.y).astype(np.float32)
    
    if data.w is not None:
        data.w = ak.to_numpy(data.w).astype(np.float32)

    # ----------
    # Pick active variables out
    scalar_vars = aux.process_regexp_ids(all_ids=aux.unroll_ak_fields(x=x, order='first'),  ids=eval('inputvars.' + args['inputvar_scalar']))

    # ----------
    # Extract active kinematic variables
    data_kin = None
    if inputvars.KINEMATIC_VARS is not None:
        kinematic_vars = aux.process_regexp_ids(
            all_ids=aux.unroll_ak_fields(x=x, order='first'),
            ids=inputvars.KINEMATIC_VARS)
        data_kin       = copy.deepcopy(data)
        data_kin.x     = aux.ak2numpy(x=data.x, fields=kinematic_vars)
        data_kin.ids   = kinematic_vars
        
    # ----------
    # Unnecessary representations
    data_MI     = None # Mutual information
    data_deps   = None # DeepSets
    data_tensor = None # Tensor
    data_graph  = None # Graph
    
    # ----------
    return {
        'data': data,
        'data_kin': data_kin,
        'data_deps': data_deps,
        'data_tensor': data_tensor,
        'data_graph': data_graph
        }

