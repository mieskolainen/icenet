# Common input & data reading routines for the electron ID
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

import copy
import math
import argparse
import pprint
import psutil
import os
import datetime
import json
import pickle
import sys

import numpy as np
import torch
import uproot
import awkward as ak

import multiprocessing


from termcolor import colored, cprint
from tqdm import tqdm

from icenet.tools import io
from icenet.tools import aux
from icenet.tools import plots
from icenet.tools import prints
from icenet.tools import process
from icenet.tools import iceroot

from iceid import graphio


from configs.eid.mctargets import *
from configs.eid.mcfilter  import *

from configs.eid.mvavars import *
from configs.eid.cuts import *


def load_root_file(root_path, ids=None, entry_start=0, entry_stop=None, args=None, library='np'):
    """ Loads a single root file.
    
    Args:
        root_path : paths to root file
    
    Returns:
        X,Y       : input, output data (with jagged content)
        ids       : variable names
    """
    
    # -----------------------------------------------
    CUTFUNC    = globals()[args['cutfunc']]
    TARFUNC    = globals()[args['targetfunc']]
    FILTERFUNC = globals()[args['filterfunc']]
    # -----------------------------------------------
    
    print('\n')
    cprint( __name__ + f'.load_root_file: Loading root file {root_path}', 'yellow')
    file   = uproot.open(root_path)
    events = file[args['tree_name']]

    # Check is it MC (based on the first event)
    isMC   = bool(events.arrays('is_mc')[0]['is_mc'])

    # --------------------------------------------------------------
    load_ids = iceroot.process_regexp_ids(ids=ids, all_ids=events.keys())
    X,ids    = iceroot.events_to_jagged_numpy(events=events, ids=load_ids, entry_start=entry_start, entry_stop=entry_stop)
    Y = None
    # --------------------------------------------------------------

    print(__name__ + f'.load_root_file: X.shape = {X.shape}')
    io.showmem()
    prints.printbar()

    # =================================================================
    # *** MC ONLY ***

    if isMC:

        # @@ MC target definition here @@
        cprint(__name__ + f'.load_root_file: Computing MC <targetfunc> ...', 'yellow')
        Y = TARFUNC(events, entry_start=entry_start, entry_stop=entry_stop, new=True)
        Y = np.asarray(Y).T
        print(__name__ + f'.load_root_file: Y.shape = {Y.shape}')
        
        # For info
        labels1 = ['is_e', 'is_egamma']
        aux.count_targets(events=events, ids=labels1, entry_start=entry_start, entry_stop=entry_stop, new=True)
        prints.printbar()
        
        # @@ MC filtering done here @@
        indmc = FILTERFUNC(X=X, ids=ids, xcorr_flow=args['xcorr_flow'])
        cprint(__name__ + f'.load_root_file: <filterfunc> | before: {len(X)}, after: {sum(indmc)} events', 'green')
        prints.printbar()
        
        X = X[indmc]
        Y = Y[indmc].squeeze() # Remove useless dimension
    
    # =================================================================

    # @@ Observable cut selections done here @@
    cprint(colored(__name__ + f'.load_root_file: Computing <cutfunc> ...'), 'yellow')
    cind = CUTFUNC(X=X, ids=ids, xcorr_flow=args['xcorr_flow'])
    cprint(__name__ + f".load_root_file: <cutfunc> | before: {len(X)}, after: {np.sum(cind)} events \n", 'green')
    
    X = X[cind]
    if isMC: Y = Y[cind]
    
    io.showmem()
    prints.printbar()
    file.close()

    # No weights
    W = None

    return X, Y, W, ids


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
    
    if KINEMATIC_VARS is not None:
        k_ind, k_vars = io.pick_vars(data, KINEMATIC_VARS)
        
        data_kin     = copy.deepcopy(data)
        data_kin.x   = data.x[:, k_ind].astype(np.float)
        data_kin.ids = k_vars

    # -------------------------------------------------------------------------
    ### DeepSets representation
    data_deps = None
    
    # -------------------------------------------------------------------------
    ### Tensor representation
    data_tensor = None

    if args['image_on']:
        data_tensor = graphio.parse_tensor_data(X=data.x, ids=ids, image_vars=globals()['CMSSW_MVA_IMAGE_VARS'], args=args)
    
    # -------------------------------------------------------------------------
    ## Graph representation
    data_graph = None

    if args['graph_on']:
        
        features   = globals()[args['inputvar']]
        data_graph = graphio.parse_graph_data(X=data.x, Y=data.y, weights=data.w, ids=data.ids, 
            features=features, global_on=args['graph_param']['global_on'], coord=args['graph_param']['coord'])
    
    # --------------------------------------------------------------------
    ### Finally pick active scalar variables out
    s_ind, s_vars = io.pick_vars(data, globals()[args['inputvar']])
    
    data.x   = data.x[:, s_ind].astype(np.float)
    data.ids = s_vars
    
    return {'data': data, 'data_kin': data_kin, 'data_deps': data_deps, 'data_tensor': data_tensor, 'data_graph': data_graph}


# ========================================================================
# ========================================================================

def init_multiprocess(maxevents=None):
    """ Initialize electron ID data [UNTESTED FUNCTION]

    Args:
        Implicit commandline and yaml file input.
    
    Returns:
        jagged array data, arguments
    """

    args, cli = process.read_config(config_path='./configs/eid')
    features  = globals()[args['imputation_param']['var']]
    
    ### SET random seed
    print(__name__ + f'.init: Setting random seed {args["rngseed"]}')
    np.random.seed(args['rngseed'])

    # --------------------------------------------------------------------
    ### SET GLOBALS (used only in this file)
    global ARGS
    ARGS = args

    if maxevents is not None:
        ARGS['maxevents'] = maxevents
    
    print(__name__ + f'.init: inputvar   =  {args["inputvar"]}')
    print(__name__ + f'.init: cutfunc    =  {args["cutfunc"]}')
    print(__name__ + f'.init: targetfunc =  {args["targetfunc"]}')
    # --------------------------------------------------------------------

    ### Load data
    #data     = io.DATASET(func_loader=, files=args['root_files'], frac=args['frac'], rngseed=args['rngseed'])
    
    CPU_count = None

    if CPU_count is None:
        CPU_count = int(np.ceil(multiprocessing.cpu_count()/2))

    # Loop over all files
    for i in range(len(args['root_files'])):

        file     = uproot.open(args['root_files'][i])
        num_events = int(file["ntuplizer"]["tree"].numentries)
        file.close()

        # Truncate upto max events
        num_events = np.min([args['maxevents'], num_events])
        N_cpu    = 1 if num_events <= 128 else CPU_count

        # Create blocks
        block_ind = aux.split_start_end(range(num_events), N_cpu)

        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        # Extend with other variables
        procs = []
        for k in range(len(block_ind)):
            inputs = {
                'root_path'   : args['root_files'][i],
                'ids'         : None,
                'entry_start' : block_ind[k][0],
                'entry_stop'  : block_ind[k][1],
                'image_on'    : args['image_on'],
                'graph_on'    : args['graph_on'],
                'args'        : args
            }

            p = multiprocessing.Process(target=load_root_file_multiprocess, args=(k, inputs, return_dict))
            procs.append(p)
            p.start()

        # Join processes
        for k in range(len(procs)):
            procs[k].join()

        # Join data
        #
        # ... add here ...
        io.showmem('yellow')

        # Fuse data
        N_tot = 0
        N_ind = []
        for k in tqdm(range(len(procs))):
            N_this = return_dict[k]['N']
            N_ind.append([N_tot, N_tot+N_this])
            N_tot += N_this
        
        print(N_ind)
        print(__name__ + ': Combining multiprocess data ...')

        td = return_dict[0]['X_tensor'].shape[1:4] # Tensor dimensions

        data = {
            'X'        : np.zeros((N_tot, return_dict[0]['X'].shape[1]), dtype=object),
            'Y'        : np.zeros((N_tot), dtype=np.long),
            'X_tensor' : np.zeros((N_tot, td[0], td[1], td[2]), dtype=np.float),
            'X_graph'  : np.zeros((N_tot), dtype=object),
            'ids'     : return_dict[0]['ids']
        }

        for k in tqdm(range(len(procs))):

            ind = np.arange(N_ind[k][0], N_ind[k][1])
            data['X'][ind,...]        = return_dict[k]['X']
            data['Y'][ind]            = return_dict[k]['Y']
            data['X_tensor'][ind,...] = return_dict[k]['X_tensor']
            data['X_graph'][ind]      = return_dict[k]['X_graph']

        # Torch conversions
        data['X_graph']  = graphio.graph2torch(data['X_graph'])
    
    return data, args, features


def load_root_file_multiprocess(procnumber, inputs, return_dict, library='np'):
    """
    [UNTESTED FUNCTION]
    """

    print('\n\n\n\n\n')
    print(inputs)


    root_path   = inputs['root_path']
    ids         = inputs['ids']
    entry_start = inputs['entry_start']
    entry_stop  = inputs['entry_stop']
    graph_on    = inputs['graph_on']
    image_on    = inputs['image_on']
    args        = inputs['args']


    """ Loads the root file.
    
    Args:
        root_path : paths to root files
    
    Returns:
        X,Y       : input, output matrices
        ids      : variable names
    """

    # -----------------------------------------------
    # ** GLOBALS **

    if args is None:
        args = ARGS

    CUTFUNC    = globals()[args['cutfunc']]
    TARFUNC    = globals()[args['targetfunc']]
    FILTERFUNC = globals()[args['filterfunc']]

    if entry_stop is None:
        entry_stop = args['maxevents']
    # -----------------------------------------------

    
    ### From root trees
    print('\n')
    cprint( __name__ + f'.load_root_file: Loading with uproot from file ' + root_path, 'yellow')
    cprint( __name__ + f'.load_root_file: entry_start = {entry_start}, entry_stop = {entry_stop}')

    file = uproot.open(root_path)
    events = file["ntuplizer"]["tree"]
    
    print(events)
    print(events.name)
    print(events.title)
    #cprint(__name__ + f'.load_root_file: events.numentries = {events.numentries}', 'green')

    ### All variables
    if ids is None:
        #ids = events.keys()
        ids = [x.decode() for x in events.keys()]# if b'image_' not in x]

    # Check is it MC (based on the first event)
    X_test = events.array('is_mc', entry_start=entry_start, entry_stop=entry_stop, library=library)

    print(X_test)

    isMC   = bool(X_test[0]) # Take the first event
    N      = len(X_test)
    print(__name__ + f'.load_root_file: isMC = {isMC}')
    print(__name__ + f'.load_root_file: N    = {N}')    


    # Now read the data
    print(__name__ + '.load_root_file: Loading root file variables ...')

    # --------------------------------------------------------------
    # Important to lead variables one-by-one (because one single np.assarray call takes too much RAM)

    # Needs to be of object type numpy array to hold arbitrary objects (such as jagged arrays) !
    X = np.empty((N, len(ids)), dtype=object) 

    for j in tqdm(range(len(ids))):
        x = events.array(ids[j], entry_start=entry_start, entry_stop=entry_stop, library=library)
        X[:,j] = np.asarray(x)
    # --------------------------------------------------------------
    Y = None


    print(__name__ + f'common: X.shape = {X.shape}')
    io.showmem()

    prints.printbar()

    # =================================================================
    # *** MC ONLY ***

    if isMC:

        # @@ MC target definition here @@
        cprint(__name__ + f'.load_root_file: Computing MC <targetfunc> ...', 'yellow')
        Y = TARFUNC(events, entry_start=entry_start, entry_stop=entry_stop)
        Y = np.asarray(Y).T

        print(__name__ + f'.load_root_file: Y.shape = {Y.shape}')

        # For info
        labels1 = ['is_e', 'is_egamma']
        aux.count_targets(events=events, ids=labels1, entry_start=entry_start, entry_stop=entry_stop)

        prints.printbar()

        # @@ MC filtering done here @@
        cprint(__name__ + f'.load_root_file: Computing MC <filterfunc> ...', 'yellow')
        indmc = FILTERFUNC(X=X, ids=ids, xcorr_flow=args['xcorr_flow'])

        cprint(__name__ + f'.load_root_file: Prior MC <filterfunc>: {len(X)} events', 'green')
        cprint(__name__ + f'.load_root_file: After MC <filterfunc>: {sum(indmc)} events ', 'green')
        prints.printbar()
        
        
        X = X[indmc]
        Y = Y[indmc].squeeze() # Remove useless dimension
    # =================================================================
    
    # -----------------------------------------------------------------
    # @@ Observable cut selections done here @@
    cprint(colored(__name__ + f'.load_root_file: Computing <cutfunc> ...'), 'yellow')
    cind = CUTFUNC(X=X, ids=ids, xcorr_flow=args['xcorr_flow'])
    # -----------------------------------------------------------------
    
    N_before = X.shape[0]

    ### Select events
    X = X[cind]
    if isMC: Y = Y[cind]

    N_after = X.shape[0]
    cprint(__name__ + f".load_root_file: Prior <cutfunc> selections: {N_before} events ", 'green')
    cprint(__name__ + f".load_root_file: Post  <cutfunc> selections: {N_after} events ({N_after / N_before:.3f})", 'green')
    print('')
    prints.printbar()

    # ** REMEMBER TO CLOSE **
    file.close()

    # --------------------------------------------------------------------
    ### Datatype conversions

    X_tensor = None
    X_graph  = None

    if image_on:
        X_tensor = graphio.parse_tensor_data(X=X, ids=ids, image_vars=globals()['CMSSW_MVA_IMAGE_VARS'], args=args)

    if graph_on:
        X_graph  = graphio.parse_graph_data_np(X=X, Y=Y, ids=ids, features=globals()[args['imputation_param']['var']])

    io.showmem()

    return_dict[procnumber] = {'X': X, 'Y': Y, 'ids': ids, "X_tensor": X_tensor, "X_graph": X_graph, 'N': X.shape[0]}

