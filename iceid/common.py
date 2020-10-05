# Common input & data reading routines for the electron ID
#
# Mikael Mieskolainen, 2020
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
import yaml
import numpy as np
import torch
import uproot
from termcolor import colored, cprint

from tqdm import tqdm

from icenet.tools import io
from icenet.tools import aux
from icenet.tools import plots
from icenet.tools import prints

from configs.eid.mctargets import *
from configs.eid.mcfilter  import *

from configs.eid.mvavars import *
from configs.eid.cuts import *

def read_config():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type = str, default='tune0')
    parser.add_argument("--datapath", type = str, default=".")
    parser.add_argument("--datasets", type = str, default="0")

    cli = parser.parse_args()

    # Input is [0,1,2,..]
    cli.datasets = cli.datasets.split(',')

    ## Read configuration
    args = {}
    config_yaml_file = cli.config + '.yml'
    with open('./configs/eid/' + config_yaml_file, 'r') as stream:
        try:
            args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    args['config'] = cli.config
    print(args)
    print('')
    print('torch.__version__: ' + torch.__version__)

    return args

def init(MAXEVENTS=None):
    """ Initialize electron ID data.

    Args:
        Implicit commandline and yaml file input.
    
    Returns:
        jagged array data, arguments
    """
    
    args = read_config()

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
    paths = []
    for i in cli.datasets:
        paths.append(cli.datapath + '/output_' + str(i) + '.root')

    # Background (0) and signal (1)
    class_id = [0,1]
    data     = io.DATASET(func_loader=load_root_file_new, files=paths, class_id=class_id, frac=args['frac'], rngseed=args['rngseed'])
    
    # @@ Imputation @@
    if args['imputation_param']['active']:

        special_values = args['imputation_param']['values'] # possible special values
        print(__name__ + f': Imputing data for special values {special_values} for variables in <{args["imputation_param"]["var"]}>')

        # Choose active dimensions
        INPUTVAR = globals()[args['imputation_param']['var']]
        dim = np.array([i for i in range(len(data.VARS)) if data.VARS[i] in INPUTVAR], dtype=int)

        # Parameters
        param = {
            "dim":        dim,
            "values":     special_values,
            "labels":     data.VARS,
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
    
    return data, args, INPUTVAR


def compute_reweights(data, args):
    """ Compute (eta,pt) reweighting coefficients.

    Args:
        data    : training data object
        args    : arguments object
    Returns:
        weights : array of re-weights
    """

    # Re-weighting variables
    PT           = data.trn.x[:,data.VARS.index('trk_pt')]
    ETA          = data.trn.x[:,data.VARS.index('trk_eta')]
    pt_binedges  = np.linspace(
                         args['reweight_param']['bins_pt'][0],
                         args['reweight_param']['bins_pt'][1],
                         args['reweight_param']['bins_pt'][2])
    eta_binedges = np.linspace(
                         args['reweight_param']['bins_eta'][0],
                         args['reweight_param']['bins_eta'][1],
                         args['reweight_param']['bins_eta'][2])

    print(__name__ + f".compute_reweights: reference_class: <{args['reweight_param']['reference_class']}>")

    ### Compute 2D-pdfs for each class
    N_class = 2
    pdf     = {}
    for c in range(N_class):
        pdf[c] = aux.pdf_2D_hist(X_A=PT[data.trn.y==c], X_B=ETA[data.trn.y==c], binedges_A=pt_binedges, binedges_B=eta_binedges)

    pdf['binedges_A'] = pt_binedges
    pdf['binedges_B'] = eta_binedges

    # Compute event-by-event weights
    if args['reweight_param']['reference_class'] != -1:
        
        trn_weights = aux.reweightcoeff2D(
            X_A = PT, X_B = ETA, pdf = pdf, y = data.trn.y, N_class=N_class,
            equal_frac       = args['reweight_param']['equal_frac'],
            reference_class  = args['reweight_param']['reference_class'],
            max_reg          = args['reweight_param']['max_reg'])
    else:
        # No re-weighting
        weights_doublet = np.zeros((data.trn.x.shape[0], N_class))
        for c in range(N_class):    
            weights_doublet[data.trn.y == c, c] = 1
        trn_weights = np.sum(weights_doublet, axis=1)

    # Compute the sum of weights per class for the output print
    frac = np.zeros(N_class)
    sums = np.zeros(N_class)
    for c in range(N_class):
        frac[c] = np.sum(data.trn.y == c)
        sums[c] = np.sum(trn_weights[data.trn.y == c])
    
    print(__name__ + f'.compute_reweights: sum(trn.y==c): {frac}')
    print(__name__ + f'.compute_reweights: sum(trn_weights[trn.y==c]): {sums}')
    print(__name__ + f'.compute_reweights: [done]\n')
    
    return trn_weights


def splitfactor(data, args):
    """
    Split electron ID data into different datatypes.
    
    Args:
        data:  jagged arrays
        args:  arguments dictionary
    
    Returns:
        scalar (vector) data
        tensor data (images)
        kinematic data
    """

    ### Pick kinematic variables out
    k_ind, k_vars    = io.pick_vars(data, KINEMATIC_ID)
    
    data_kin         = copy.deepcopy(data)
    data_kin.trn.x   = data.trn.x[:, k_ind].astype(np.float)
    data_kin.val.x   = data.val.x[:, k_ind].astype(np.float)
    data_kin.tst.x   = data.tst.x[:, k_ind].astype(np.float)
    data_kin.VARS    = k_vars
    
    ### Pick active jagged array / "image" variables out
    j_ind, j_vars    = io.pick_vars(data, globals()['CMSSW_MVA_ID_IMAGE'])
    
    data_image       = copy.deepcopy(data)
    data_image.trn.x = data.trn.x[:, j_ind]
    data_image.val.x = data.val.x[:, j_ind]
    data_image.tst.x = data.tst.x[:, j_ind]
    data_image.VARS  = j_vars 

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
    data_tensor['trn'] = aux.jagged2tensor(X=data_image.trn.x, VARS=j_vars, xyz=xyz, x_binedges=eta_binedges, y_binedges=phi_binedges)
    data_tensor['val'] = aux.jagged2tensor(X=data_image.val.x, VARS=j_vars, xyz=xyz, x_binedges=eta_binedges, y_binedges=phi_binedges)
    data_tensor['tst'] = aux.jagged2tensor(X=data_image.tst.x, VARS=j_vars, xyz=xyz, x_binedges=eta_binedges, y_binedges=phi_binedges)
    
    ### Pick active scalar variables out
    s_ind, s_vars = io.pick_vars(data, globals()[args['inputvar']])
    
    data.trn.x    = data.trn.x[:, s_ind].astype(np.float)
    data.val.x    = data.val.x[:, s_ind].astype(np.float)
    data.tst.x    = data.tst.x[:, s_ind].astype(np.float)
    data.VARS     = s_vars

    return data, data_tensor, data_kin


def load_root_file_new(root_path, entrystart=0, entrystop=None, class_id = []):
    """ Loads the root file.
    
    Args:
        root_path : paths to root files
        class_id  : class ids
    
    Returns:
        X,Y       : input, output matrices
        VARS      : variable names
    """

    # -----------------------------------------------
    # ** GLOBALS **

    CUTFUNC    = globals()[ARGS['cutfunc']]
    TARFUNC    = globals()[ARGS['targetfunc']]
    FILTERFUNC = globals()[ARGS['filterfunc']]

    if entrystop is None:
        entrystop = ARGS['MAXEVENTS']
    # -----------------------------------------------

    def showmem():
        cprint(__name__ + f""".load_root_file: Process RAM usage: {io.process_memory_use():0.2f} GB 
            [total RAM in use {psutil.virtual_memory()[2]} %]""", 'red')
    
    ### From root trees
    print('\n')
    cprint( __name__ + f'.load_root_file: Loading with uproot from file ' + root_path, 'yellow')
    cprint( __name__ + f'.load_root_file: entrystart = {entrystart}, entrystop = {entrystop}')

    file = uproot.open(root_path)
    events = file["ntuplizer"]["tree"]

    print(events.name)
    print(events.title)
    cprint(__name__ + f'.load_root_file: events.numentries = {events.numentries}', 'green')

    ### All variables
    VARS   = [x.decode() for x in events.keys()]
    #VARS_scalar = [x.decode() for x in events.keys() if b'image_' not in x]

    # Turn into dictionaries
    X_dict = events.arrays(VARS, namedecode = "utf-8", entrystart=entrystart, entrystop=entrystop)


    # Is it MC (decision based on the first event)
    isMC = X_dict['is_mc'][0]

    showmem()

    # -----------------------------------------------------------------
    ### Convert the input into an array of size (dimensions x events)
    # Note that dimensions here can contain jagged information

    print(__name__ + '.load_root_file: Constructing X array ...')

    X = np.array([X_dict[j] for j in VARS])
    X = X.T
    Y = None
    
    X_dict.clear() # Free memory
    print(f'X.shape = {X.shape}')
    showmem()

    prints.printbar()

    # =================================================================
    # *** MC ONLY ***

    if isMC:

        # @@ MC target definition here @@
        cprint(__name__ + f'.load_root_file: Computing MC <targetfunc> ...', 'yellow')
        Y = TARFUNC(events, entrystart=entrystart, entrystop=entrystop)

        # For info
        labels1 = ['is_e', 'is_egamma']
        aux.count_targets(events=events, names=labels1, entrystart=entrystart, entrystop=entrystop)

        prints.printbar()

        # @@ MC filtering done here @@
        cprint(__name__ + f'.load_root_file: Computing MC <filterfunc> ...', 'yellow')
        indmc = FILTERFUNC(X=X, VARS=VARS, xcorr_flow=ARGS['xcorr_flow'])

        cprint(__name__ + f'.load_root_file: Prior MC <filterfunc>: {len(X)} events', 'green')
        cprint(__name__ + f'.load_root_file: After MC <filterfunc>: {sum(indmc)} events ', 'green')
        prints.printbar()

        Y = Y[indmc]
        X = X[indmc]
    # =================================================================
    
    # -----------------------------------------------------------------
    # @@ Observable cut selections done here @@
    cprint(colored(__name__ + f'.load_root_file: Computing <cutfunc> ...'), 'yellow')
    cind = CUTFUNC(X=X, VARS=VARS, xcorr_flow=ARGS['xcorr_flow'])
    # -----------------------------------------------------------------

    N_before = X.shape[0]

    ### Select events
    X = X[cind]
    if isMC: Y = Y[cind]

    N_after = X.shape[0]
    cprint(__name__ + f".load_root_file: Prior <cutfunc> selections: {N_before} events ", 'green')
    cprint(__name__ + f".load_root_file: Post  <cutfunc> selections: {N_after} events ({N_after / N_before:.3f})", 'green')
    print('')

    showmem()
    prints.printbar()
    
    return X, Y, VARS