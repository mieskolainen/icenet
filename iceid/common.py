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
#import uproot4
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


from iceid import graphio


from configs.eid.mctargets import *
from configs.eid.mcfilter  import *

from configs.eid.mvavars import *
from configs.eid.cuts import *


def init(MAXEVENTS=None):
    """ Initialize electron ID data.

    Args:
        Implicit commandline and yaml file input.
    
    Returns:
        jagged array data, arguments
    """
    
    args, cli = process.read_config(config_path='./configs/eid')
    features  = globals()[args['imputation_param']['var']]
    
    ### SET random seed
    print(__name__ + f'.init: Setting random seed: {args["rngseed"]}')
    np.random.seed(args['rngseed'])
    
    # --------------------------------------------------------------------    
    print(__name__ + f'.init: inputvar   =  {args["inputvar"]}')
    print(__name__ + f'.init: cutfunc    =  {args["cutfunc"]}')
    print(__name__ + f'.init: targetfunc =  {args["targetfunc"]}')
    # --------------------------------------------------------------------

    ### Load data

    # Background (0) and signal (1)
    class_id = [0,1]

    files = io.glob_expand_files(datapath=cli.datapath, datasets=cli.datasets)
    args['root_files'] = files

    if MAXEVENTS is None:
        MAXEVENTS = args['MAXEVENTS']
    load_args = {'entry_stop': MAXEVENTS,
                 'args': args}

    data = io.IceTriplet(func_loader=load_root_file, files=files, load_args=load_args,
        class_id=class_id, frac=args['frac'], rngseed=args['rngseed'])
    
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
        # No imputation, but fix spurious NaN / Inf
        data.trn.x[np.logical_not(np.isfinite(data.trn.x))] = 0
        data.val.x[np.logical_not(np.isfinite(data.val.x))] = 0
        data.tst.x[np.logical_not(np.isfinite(data.tst.x))] = 0

    cprint(__name__ + f""".common: Process RAM usage: {io.process_memory_use():0.2f} GB 
        [total RAM in use: {psutil.virtual_memory()[2]} %]""", 'red')

    return data, args, features


def splitfactor(data, args):
    """
    Split electron ID data into different datatypes.
    
    Args:
        data:        jagged numpy arrays
        args:        arguments dictionary
    
    Returns:
        data:        scalar (vector) data
        data_tensor: tensor data (images)
        data_kin:    kinematic data
    """
    
    ### Pick kinematic variables out
    k_ind, k_vars    = io.pick_vars(data, KINEMATIC_ID)
    
    data_kin         = copy.deepcopy(data)
    data_kin.trn.x   = data.trn.x[:, k_ind].astype(np.float)
    data_kin.val.x   = data.val.x[:, k_ind].astype(np.float)
    data_kin.tst.x   = data.tst.x[:, k_ind].astype(np.float)
    data_kin.ids     = k_vars

    data_tensor      = None

    if args['image_on']:

        ### Pick active jagged array / "image" variables out
        j_ind, j_vars    = io.pick_vars(data, globals()['CMSSW_MVA_ID_IMAGE'])
        
        data_image       = copy.deepcopy(data)
        data_image.trn.x = data.trn.x[:, j_ind]
        data_image.val.x = data.val.x[:, j_ind]
        data_image.tst.x = data.tst.x[:, j_ind]
        data_image.ids  = j_vars 

        # Use single channel tensors
        if   args['image_param']['channels'] == 1:
            xyz = [['image_clu_eta', 'image_clu_phi', 'image_clu_e']]

        # Use multichannel tensors
        elif args['image_param']['channels'] == 2:
            xyz  = [['image_clu_eta', 'image_clu_phi', 'image_clu_e'], 
                    ['image_pf_eta',  'image_pf_phi',  'image_pf_p']]
        else:
            raise Except(__name__ + f'.splitfactor: Unknown [image_param][channels] parameter')

        eta_binedges = args['image_param']['eta_bins']
        phi_binedges = args['image_param']['phi_bins']    

        # Pick tensor data out
        cprint(__name__ + f'.splitfactor: jagged2tensor processing ...', 'yellow')

        data_tensor = {}
        data_tensor['trn'] = aux.jagged2tensor(X=data_image.trn.x, ids=j_vars, xyz=xyz, x_binedges=eta_binedges, y_binedges=phi_binedges)
        data_tensor['val'] = aux.jagged2tensor(X=data_image.val.x, ids=j_vars, xyz=xyz, x_binedges=eta_binedges, y_binedges=phi_binedges)
        data_tensor['tst'] = aux.jagged2tensor(X=data_image.tst.x, ids=j_vars, xyz=xyz, x_binedges=eta_binedges, y_binedges=phi_binedges)
    

    ### Pick active scalar variables out
    s_ind, s_vars = io.pick_vars(data, globals()[args['inputvar']])
    
    data.trn.x    = data.trn.x[:, s_ind].astype(np.float)
    data.val.x    = data.val.x[:, s_ind].astype(np.float)
    data.tst.x    = data.tst.x[:, s_ind].astype(np.float)
    data.ids      = s_vars
    
    return data, data_tensor, data_kin


def load_root_file(root_path, ids=None, class_id=None, entry_start=0, entry_stop=None, args=None, library='np'):
    """ Loads the root file.
    
    Args:
        root_path : paths to root files
        class_id  : class ids
    
    Returns:
        X,Y       : input, output matrices
        ids       : variable names
    """

    # -----------------------------------------------
    CUTFUNC    = globals()[args['cutfunc']]
    TARFUNC    = globals()[args['targetfunc']]
    FILTERFUNC = globals()[args['filterfunc']]
    # -----------------------------------------------

    ### From root trees
    print('\n')
    cprint( __name__ + f'.load_root_file: Loading with uproot from file ' + root_path, 'yellow')
    cprint( __name__ + f'.load_root_file: entry_start = {entry_start}, entry_stop = {entry_stop}')

    file   = uproot.open(root_path)
    events = file["ntuplizer"]["tree"]
    
    print(events)
    print(events.name)
    print(events.title)

    ### All variables
    if ids is None:
        ids = events.keys() #[x for x in events.keys()]
    #VARS_scalar = [x.decode() for x in events.keys() if b'image_' not in x]
    #print(ids)
    
    # Check is it MC (based on the first event)
    X_test = events.arrays('is_mc', entry_start=entry_start, entry_stop=entry_stop)

    isMC   = bool(X_test[0]['is_mc'])
    N      = len(X_test)
    print(__name__ + f'.load_root_file: isMC: {isMC}')

    # --------------------------------------------------------------
    # Important to lead variables one-by-one (because one single np.assarray call takes too much RAM)

    # Needs to be of object type numpy array to hold arbitrary objects (such as jagged arrays) !
    X = np.empty((N, len(ids)), dtype=object) 

    for j in tqdm(range(len(ids))):
        x = events.arrays(ids[j], entry_start=entry_start, entry_stop=entry_stop, library="np", how=list)
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
        Y = TARFUNC(events, entry_start=entry_start, entry_stop=entry_stop, new=True)
        Y = np.asarray(Y).T
        print(__name__ + f'common: Y.shape = {Y.shape}')

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

    return X, Y, ids




# ========================================================================
# ========================================================================

def init_multiprocess(MAXEVENTS=None):
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

    if MAXEVENTS is not None:
        ARGS['MAXEVENTS'] = MAXEVENTS
    
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
        N_events = int(file["ntuplizer"]["tree"].numentries)
        file.close()

        # Truncate upto max events
        N_events = np.min([args['MAXEVENTS'], N_events])
        N_cpu    = 1 if N_events <= 128 else CPU_count

        # Create blocks
        block_ind = aux.split_start_end(range(N_events), N_cpu)

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
        entry_stop = args['MAXEVENTS']
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
        X_tensor = graphio.parse_tensor_data(X=X, ids=ids, image_vars=globals()['CMSSW_MVA_ID_IMAGE'], args=args)

    if graph_on:
        X_graph  = graphio.parse_graph_data_np(X=X, Y=Y, ids=ids, features=globals()[args['imputation_param']['var']])

    io.showmem()

    return_dict[procnumber] = {'X': X, 'Y': Y, 'ids': ids, "X_tensor": X_tensor, "X_graph": X_graph, 'N': X.shape[0]}

