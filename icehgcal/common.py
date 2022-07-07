# Common input & data reading routines for HGCAL
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

from icehgcal import graphio


from configs.hgcal.mctargets import *
from configs.hgcal.mcfilter  import *

from configs.hgcal.mvavars import *
from configs.hgcal.cuts import *



def load_root_file(root_path, ids=None, entry_start=0, entry_stop=None, args=None):
    """ Loads the root file.
    
    Args:
        root_path : paths to root files
    
    Returns:
        X,Y       : input, output matrices
        ids       : variable names
    """

    # -----------------------------------------------

    # ** Pick the variables **
    ids = MVA_SCALAR_VARS
    
    param = {
        "entry_start": entry_start,
        "entry_stop":  entry_stop,
        "args":        args,
        "load_ids":    ids     
    }
    
    tree = args['tree_name']


    # =================================================================
    # *** BACKGROUND MC ***

    filename = args["input"]['class_0']
    rootfile = io.glob_expand_files(datasets=filename, datapath=root_path)
    
    X_B, VARS = process_root(rootfile=rootfile, tree=tree, isMC=True, **param)
    Y_B = np.zeros(X_B.shape[0])


    # =================================================================
    # *** SIGNAL MC ***

    filename = args["input"]['class_1']
    rootfile = io.glob_expand_files(datasets=filename, datapath=root_path)

    X_S, VARS = process_root(rootfile=rootfile, tree=tree, isMC=True, **param)
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
    
    # No weights
    W = None

    return X, Y, W, VARS


def process_root(rootfile, tree, load_ids, isMC, entry_start, entry_stop, args):

    CUTFUNC    = globals()[args['cutfunc']]
    FILTERFUNC = globals()[args['filterfunc']]

    X,ids      = iceroot.load_tree(rootfile=rootfile, tree=tree, entry_start=entry_start, entry_stop=entry_stop, ids=load_ids)
    
    """
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
    """

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

    data = io.IceXYW(x=x, y=y, w=w, ids=ids)

    # -------------------------------------------------------------------------
    ### Pick kinematic variables out
    data_kin = None
    """
    if KINEMATIC_VARS is not None:
        k_ind, k_vars = io.pick_vars(data, KINEMATIC_VARS)
        
        data_kin     = copy.deepcopy(data)
        data_kin.x   = data.x[:, k_ind].astype(np.float)
        data_kin.ids = k_vars
    """
    # -------------------------------------------------------------------------
    ### DeepSets representation
    data_deps = None
    
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
    
    features   = globals()[args['inputvar']]
    data_graph = graphio.parse_graph_data(X=data.x, Y=data.y, weights=data.w, ids=data.ids, 
        features=features, graph_param=args['graph_param'])
    
    # --------------------------------------------------------------------
    ### Finally pick active scalar variables out
    """
    s_ind, s_vars = io.pick_vars(data, globals()[args['inputvar']])
    
    data.x   = data.x[:, s_ind].astype(np.float)
    data.ids = s_vars
    """
    data.x = None
    
    return {'data': data, 'data_kin': data_kin, 'data_deps': data_deps, 'data_tensor': data_tensor, 'data_graph': data_graph}


# ========================================================================
# ========================================================================
