# Common input & data reading routines for HGCAL (CND, TRK modes)
# 
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

import copy
import math
import argparse
import pprint
from pprint import pprint

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

from icehgcal import preprocess
from icehgcal import graphio

from configs.hgcal.mctargets import *
from configs.hgcal.mcfilter  import *

from configs.hgcal.mvavars import *
from configs.hgcal.cuts import *


def read_data_tracklet(args, runmode):
    """
    (HGCAL-TRK)
    """

    # Create trackster data
    cache_filename = f'{args["datadir"]}/data_{args["__hash__"]}.pkl'

    if args['__use_cache__'] == False or (not os.path.exists(cache_filename)):

        if runmode != "genesis":
            raise Exception(__name__ + f'.read_data_tracklet: Data not in cache (or __use_cache__ == False) but --runmode is not "genesis"')
        
        data      = preprocess.event_loop(files=args['root_files'], graph_param=args['graph_param'], maxevents=args['maxevents']) 
        X         = graphio.parse_graph_data_trackster(data=data, graph_param=args['graph_param'], weights=None)
        
        # Pickle to disk
        with open(cache_filename, "wb") as fp:
            pickle.dump([X, args], fp)
            cprint(__name__ + f'.read_data_tracklet: Saved output to cache: "{cache_filename}"', 'yellow')

    else:
        with open(cache_filename, 'rb') as handle:
            cprint(__name__ + f'.read_data_tracklet: Loading from cache: "{cache_filename}"', 'yellow')
            X, genesis_args = pickle.load(handle)

            cprint(__name__ + f'.read_data_tracklet: Cached data was generated with arguments:', 'yellow')
            pprint(genesis_args)

    return X

def process_tracklet_data(args, X):
    """
    (HGCAL-TRK)
    """

    ### Edge weight re-weighting
    if args['reweight']:
        if args['reweight_param']['equal_frac']:

            print(__name__ + f'.process_tracklet_data: Computing binary edge ratio balancing re-weights ...')

            # Count number of false/true edges
            num = [0,0]
            for i in tqdm(range(len(X))):
                for j in range(len(X[i].y)):
                    num[X[i].y[j]] += 1

            # Re-weight
            EQ = num[0]/num[1] if (args['reweight_param']['reference_class'] == 0) else num[1]/num[0]

            print(__name__ + f'.process_tracklet_data: EQ = {EQ}')

            # Apply weights
            for i in tqdm(range(len(X))):
                for j in range(len(X[i].y)):
                    if X[i].y[j] != args['reweight_param']['reference_class']:
                        X[i].w[j] *= EQ
    
    ### Split
    X_trn, X_val, X_tst = io.split_data_simple(X=X, frac=args['frac'])
    data = {}
    data['trn'] = {'data': None, 'data_kin': None, 'data_deps': None, 'data_tensor': None, 'data_graph': X_trn}
    data['val'] = {'data': None, 'data_kin': None, 'data_deps': None, 'data_tensor': None, 'data_graph': X_val}
    data['tst'] = {'data': None, 'data_kin': None, 'data_deps': None, 'data_tensor': None, 'data_graph': X_tst}

    return data


def load_root_file(root_path, ids=None, entry_start=0, entry_stop=None, maxevents=None, args=None):
    """ 
    (HGCAL-CND)

    Loads the root files
    
    Args:
        root_path : paths to root files (list)
    
    Returns:
        X,Y       : input, output matrices
        ids       : variable names
    """

    if type(root_path) is list:
        root_path = root_path[0] # Remove [] list, we expect only the path here
    
    # -----------------------------------------------

    # ** Pick the variables **
    ids = MVA_SCALAR_VARS
    
    param = {
        "entry_start": entry_start,
        "entry_stop":  entry_stop,
        "maxevents":   maxevents,
        "args":        args,
        "load_ids":    ids     
    }
    
    tree = args['tree_name']

    # =================================================================
    # *** BACKGROUND MC ***

    filename = args["input"]['class_0']
    rootfile = io.glob_expand_files(datasets=filename, datapath=root_path)
    
    X_B,ids  = process_root(rootfile=rootfile, tree=tree, isMC=True, **param)
    Y_B      = np.zeros(X_B.shape[0])


    # =================================================================
    # *** SIGNAL MC ***

    filename = args["input"]['class_1']
    rootfile = io.glob_expand_files(datasets=filename, datapath=root_path)

    X_S,ids  = process_root(rootfile=rootfile, tree=tree, isMC=True, **param)
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
    
    # No weights
    W = None

    # TBD add cut statistics etc. here
    info = {}

    return X, Y, W, ids, info


def process_root(rootfile, tree, load_ids, isMC, entry_start, entry_stop, maxevents, args):
    """
    (HGCAL-CND)
    """

    CUTFUNC    = globals()[args['cutfunc']]
    FILTERFUNC = globals()[args['filterfunc']]

    X,ids      = iceroot.load_tree(rootfile=rootfile, tree=tree,
        entry_start=entry_start, entry_stop=entry_stop, maxevents=maxevents, ids=load_ids, library='np')
    
    """
    # @@ Filtering done here @@
    mask = FILTERFUNC(X=X, ids=ids, isMC=isMC, xcorr_flow=args['xcorr_flow'])
    plots.plot_selection(X=X, mask=mask, ids=ids, args=args, label=f'<filter>_{isMC}', varlist=PLOT_VARS)
    cprint(__name__ + f'.process_root: isMC = {isMC} | <filterfunc> before: {len(X)}, after: {sum(mask)} events ', 'green')
    
    X   = X[mask]
    prints.printbar()
    
    # @@ Observable cut selections done here @@
    mask = CUTFUNC(X=X, ids=ids, isMC=isMC, xcorr_flow=args['xcorr_flow'])
    plots.plot_selection(X=X, mask=mask, ids=ids, args=args, label=f'<cutfunc>_{isMC}', varlist=PLOT_VARS)
    cprint(__name__ + f".process_root: isMC = {isMC} | <cutfunc>: before: {len(X)}, after: {sum(mask)} events \n", 'green')

    X   = X[mask]
    io.showmem()
    prints.printbar()
    """

    return X, ids


def splitfactor(x, y, w, ids, args):
    """
    (HGCAL-CND)
    
    Transform data into different datatypes.
    
    Args:
        data:  jagged arrays
        args:  arguments dictionary
    
    Returns:
        dictionary with different data representations
    """

    data = io.IceXYW(x=x, y=y, w=w, ids=ids)

    ### Pick active variables out
    scalar_vars = aux.process_regexp_ids(all_ids=ids, ids=globals()[args['inputvar_scalar']])

    # -------------------------------------------------------------------------
    ### Pick kinematic variables out
    data_kin = None
    
    #if KINEMATIC_VARS is not None:
    #    k_ind, k_vars = io.pick_vars(data, aux.process_regexp_ids(all_ids=ids, ids=KINEMATIC_VARS))
    #    
    #    data_kin      = copy.deepcopy(data)
    #    data_kin.x    = data.x[:, k_ind].astype(np.float)
    #    data_kin.ids  = k_vars

    # -------------------------------------------------------------------------
    ### DeepSets representation
    data_deps   = None
    
    # -------------------------------------------------------------------------
    ### Tensor representation
    data_tensor = None
    """
    if args['image_on']:
        data_tensor = graphio.parse_tensor_data(X=data.x, ids=ids, image_vars=globals()['CMSSW_MVA_IMAGE_VARS'], args=args)
    """
    # -------------------------------------------------------------------------
    ## Graph representation
    data_graph = None
    
    data_graph = graphio.parse_graph_data_candidate(X=data.x, Y=data.y, weights=data.w, ids=data.ids,
        features=scalar_vars, graph_param=args['graph_param'])
    
    # --------------------------------------------------------------------
    ### Finally pick active scalar variables out
    data.x = None
    
    return {'data': data, 'data_kin': data_kin, 'data_deps': data_deps, 'data_tensor': data_tensor, 'data_graph': data_graph}

# ========================================================================
# ========================================================================
