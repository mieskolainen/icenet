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
#import uproot4
import uproot
import awkward as ak

import multiprocessing

# Command line arguments
from glob import glob
from braceexpand import braceexpand


from termcolor import colored, cprint
from tqdm import tqdm

from icenet.tools import io
from icenet.tools import aux
from icenet.tools import plots
from icenet.tools import prints

from iceid import graphio

from configs.eid.mctargets import *
from configs.eid.mcfilter  import *

from configs.eid.mvavars import *
from configs.eid.cuts import *

def read_config():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type = str, default='tune0')
    parser.add_argument("--datapath", type = str, default=".")
    parser.add_argument("--datasets", type = str, default="*")

    cli = parser.parse_args()
    
    # -------------------------------------------------------------------
    ## Read configuration
    args = {}
    config_yaml_file = cli.config + '.yml'
    with open('./configs/eid/' + config_yaml_file, 'r') as stream:
        try:
            args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
    args['config'] = cli.config

    # -------------------------------------------------------------------
    ### Set image and graph constructions on/off
    args['graph_on'] = False
    args['image_on'] = False

    for i in range(len(args['active_models'])):
        ID    = args['active_models'][i]
        param = args[f'{ID}_param']

        if ('graph' in param['train']) or ('graph' in param['predict']):
            args['graph_on'] = True
        if ('image' in param['train']) or ('image' in param['predict']):
            args['image_on'] = True

    print('\n')
    cprint(__name__ + f'.read_config: graph_on = {args["graph_on"]}', 'yellow')
    cprint(__name__ + f'.read_config: image_on = {args["image_on"]}', 'yellow')    

    # -------------------------------------------------------------------
    # Do brace expansion
    datasets = list(braceexpand(cli.datasets))

    # Parse input files into a list
    args['root_files'] = list()
    for data in datasets:
        filepath = glob(cli.datapath + '/' + data + '.root')
        if filepath != []:
            for i in range(len(filepath)):
                args['root_files'].append(filepath[i])
    # -------------------------------------------------------------------
    
    print(args)
    print('')
    print(" torch.__version__: " + torch.__version__)
    print("")
    print(" Try 'filename_*' ")
    print(" Try 'filename_[0-99]' ")
    print(" Try 'filename_0' ")
    print(" Try 'filename_{0,3,4}' ")
    print(" Google <glob wildcards> and brace expansion.")
    print("")

    features = globals()[args['imputation_param']['var']]

    return args, cli, features


def init_multiprocess(MAXEVENTS=None):
    """ Initialize electron ID data.

    Args:
        Implicit commandline and yaml file input.
    
    Returns:
        jagged array data, arguments
    """

    args, cli, features = read_config()

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
                'root_path'  : args['root_files'][i],
                'VARS'       : None,
                'entrystart' : block_ind[k][0],
                'entrystop'  : block_ind[k][1],
                'image_on'   : args['image_on'],
                'graph_on'   : args['graph_on'],
                'args'       : args
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
            'VARS'     : return_dict[0]['VARS']
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


def init(MAXEVENTS=None):
    """ Initialize electron ID data.

    Args:
        Implicit commandline and yaml file input.
    
    Returns:
        jagged array data, arguments
    """
    
    args, cli, features = read_config()

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

    # Background (0) and signal (1)
    class_id = [0,1]
    data     = io.DATASET(func_loader=load_root_file_new, files=args['root_files'], class_id=class_id, frac=args['frac'], rngseed=args['rngseed'])
    
    # @@ Imputation @@
    if args['imputation_param']['active']:

        special_values = args['imputation_param']['values'] # possible special values
        print(__name__ + f': Imputing data for special values {special_values} for variables in <{args["imputation_param"]["var"]}>')

        # Choose active dimensions
        dim = np.array([i for i in range(len(data.VARS)) if data.VARS[i] in features], dtype=int)

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
    
    return data, args, features


def compute_reweights(data, args, N_class=2, EPS=1e-12):
    """ Compute (eta,pt) reweighting coefficients.
    Args:
        data    : training data object
        args    : arguments object
    Returns:
        weights : array of re-weights
    """

    ### Re-weighting variables
    RV = {}
    RV['pt']  = data.trn.x[:,data.VARS.index('trk_pt')].astype(np.float)
    RV['eta'] = data.trn.x[:,data.VARS.index('trk_eta')].astype(np.float)
    
    
    ### Pre-transform
    for var in ['pt', 'eta']:
        mode = args['reweight_param'][f'transform_{var}']

        if   mode == 'log10':
            RV[var] = np.log10(RV[var] + EPS)

            # Bins
            args['reweight_param'][f'bins_{var}'][0] = np.log10(args['reweight_param'][f'bins_{var}'][0] + EPS)
            args['reweight_param'][f'bins_{var}'][1] = np.log10(args['reweight_param'][f'bins_{var}'][1])

        elif mode == 'sqrt':
            RV[var] = np.sqrt(RV[var])

            # Bins
            args['reweight_param'][f'bins_{var}'][0] = np.sqrt(args['reweight_param'][f'bins_{var}'][0])
            args['reweight_param'][f'bins_{var}'][1] = np.sqrt(args['reweight_param'][f'bins_{var}'][1])

        elif mode == 'square':
            RV[var] = RV[var]**2

            # Bins
            args['reweight_param'][f'bins_{var}'][0] = (args['reweight_param'][f'bins_{var}'][0])**2
            args['reweight_param'][f'bins_{var}'][1] = (args['reweight_param'][f'bins_{var}'][1])**2

        elif mode == None:
            True
        else:
            raise Except(__name__ + '.compute_reweights: Unknown pre-transform')

    # Binning setup
    binedges = {}
    for var in ['pt', 'eta']:
        if   args['reweight_param'][f'binmode_{var}'] == 'linear':
            binedges[var] = np.linspace(
                                 args['reweight_param'][f'bins_{var}'][0],
                                 args['reweight_param'][f'bins_{var}'][1],
                                 args['reweight_param'][f'bins_{var}'][2])

        elif args['reweight_param'][f'binmode_{var}'] == 'log':
            binedges[var] = np.logspace(
                                 np.log10(np.max([args['reweight_param'][f'bins_{var}'][0], EPS])),
                                 np.log10(args['reweight_param'][f'bins_{var}'][1]),
                                 args['reweight_param'][f'bins_{var}'][2], base=10)
        else:
            raise Except(__name__ + ': Unknown re-weight binning mode')
    
    print(__name__ + f".compute_reweights: reference_class: <{args['reweight_param']['reference_class']}>")


    ### Compute 2D-PDFs for each class
    pdf     = {}
    for c in range(N_class):
        pdf[c] = aux.pdf_2D_hist(X_A=RV['pt'][data.trn.y==c], X_B=RV['eta'][data.trn.y==c],
                                    binedges_A=binedges['pt'], binedges_B=binedges['eta'])

    pdf['binedges_A'] = binedges['pt']
    pdf['binedges_B'] = binedges['eta']


    # Compute event-by-event weights
    if args['reweight_param']['reference_class'] != -1:
        
        trn_weights = aux.reweightcoeff2D(
            X_A = RV['pt'], X_B = RV['eta'], pdf = pdf, y = data.trn.y, N_class=N_class,
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
    
    print(__name__ + f'.compute_reweights: sum[trn.y==c]: {frac}')
    print(__name__ + f'.compute_reweights: sum[trn_weights[trn.y==c]]: {sums}')
    print(__name__ + f'.compute_reweights: [done] \n')
    
    return trn_weights


def compute_reweights_XY(X, Y, VARS, args):
    """ Compute (eta,pt) reweighting coefficients.

    Args:
        data    : training data object
        args    : arguments object
    Returns:
        weights : array of re-weights
    """

    # ...
    
    return True


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

    data_tensor = {}

    if args['image_on']:

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


def slow_conversion(hdfarray):
    return np.array(hdfarray)


def fast_conversion(hdfarray, shape, dtype):
    a    = np.empty(shape=shape, dtype=dtype)
    a[:] = hdfarray[:]
    return a


def load_root_file_multiprocess(procnumber, inputs, return_dict):
    
    print('\n\n\n\n\n')
    print(inputs)

    # root_path, VARS=None, entrystart=0, entrystop=None, output_graph=True, output_tensor=False, args=None

    root_path  = inputs['root_path']
    VARS       = inputs['VARS']
    entrystart = inputs['entrystart']
    entrystop  = inputs['entrystop']
    graph_on   = inputs['graph_on']
    image_on   = inputs['image_on']
    args       = inputs['args']


    """ Loads the root file.
    
    Args:
        root_path : paths to root files
    
    Returns:
        X,Y       : input, output matrices
        VARS      : variable names
    """

    # -----------------------------------------------
    # ** GLOBALS **

    if args is None:
        args = ARGS

    CUTFUNC    = globals()[args['cutfunc']]
    TARFUNC    = globals()[args['targetfunc']]
    FILTERFUNC = globals()[args['filterfunc']]

    if entrystop is None:
        entrystop = args['MAXEVENTS']
    # -----------------------------------------------

    
    ### From root trees
    print('\n')
    cprint( __name__ + f'.load_root_file: Loading with uproot from file ' + root_path, 'yellow')
    cprint( __name__ + f'.load_root_file: entrystart = {entrystart}, entrystop = {entrystop}')

#    file = uproot4.open(root_path)
#    events = file["ntuplizer"]["tree"]

    file = uproot.open(root_path)
    events = file["ntuplizer"]["tree"]


    print(events)
    print(events.name)
    print(events.title)
    #cprint(__name__ + f'.load_root_file: events.numentries = {events.numentries}', 'green')

    ### All variables
    if VARS is None:
        #VARS = events.keys()
        VARS = [x.decode() for x in events.keys()]# if b'image_' not in x]

    # Check is it MC (based on the first event)
    # X_test = events.arrays('is_mc', entry_start=entrystart, entry_stop=entrystop)
    X_test = events.array('is_mc', entrystart=entrystart, entrystop=entrystop)

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
    X = np.empty((N, len(VARS)), dtype=object) 

    for j in tqdm(range(len(VARS))):
        #x = events.arrays(VARS[j], library="np", how=list, entry_start=entrystart, entry_stop=entrystop)
        x = events.array(VARS[j], entrystart=entrystart, entrystop=entrystop)
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
        Y = TARFUNC(events, entrystart=entrystart, entrystop=entrystop)
        Y = np.asarray(Y).T

        print(__name__ + f'.load_root_file: Y.shape = {Y.shape}')

        # For info
        labels1 = ['is_e', 'is_egamma']
        aux.count_targets(events=events, names=labels1, entrystart=entrystart, entrystop=entrystop)

        prints.printbar()

        # @@ MC filtering done here @@
        cprint(__name__ + f'.load_root_file: Computing MC <filterfunc> ...', 'yellow')
        indmc = FILTERFUNC(X=X, VARS=VARS, xcorr_flow=args['xcorr_flow'])

        cprint(__name__ + f'.load_root_file: Prior MC <filterfunc>: {len(X)} events', 'green')
        cprint(__name__ + f'.load_root_file: After MC <filterfunc>: {sum(indmc)} events ', 'green')
        prints.printbar()
        
        
        X = X[indmc]
        Y = Y[indmc].squeeze() # Remove useless dimension
    # =================================================================
    
    # -----------------------------------------------------------------
    # @@ Observable cut selections done here @@
    cprint(colored(__name__ + f'.load_root_file: Computing <cutfunc> ...'), 'yellow')
    cind = CUTFUNC(X=X, VARS=VARS, xcorr_flow=args['xcorr_flow'])
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
        X_tensor = graphio.parse_tensor_data(X=X, VARS=VARS, image_vars=globals()['CMSSW_MVA_ID_IMAGE'], args=args)

    if graph_on:
        X_graph  = graphio.parse_graph_data_np(X=X, Y=Y, VARS=VARS, features=globals()[args['imputation_param']['var']])

    io.showmem()

    return_dict[procnumber] = {'X': X, 'Y': Y, 'VARS': VARS, "X_tensor": X_tensor, "X_graph": X_graph, 'N': X.shape[0]}


def load_root_file_new(root_path, VARS=None, entrystart=0, entrystop=None, class_id = [], args=None):
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

    if args is None:
        args = ARGS

    CUTFUNC    = globals()[args['cutfunc']]
    TARFUNC    = globals()[args['targetfunc']]
    FILTERFUNC = globals()[args['filterfunc']]

    if entrystop is None:
        entrystop = args['MAXEVENTS']
    # -----------------------------------------------

    def showmem():
        cprint(__name__ + f""".load_root_file: Process RAM usage: {io.process_memory_use():0.2f} GB 
            [total RAM in use {psutil.virtual_memory()[2]} %]""", 'red')
    
    ### From root trees
    print('\n')
    cprint( __name__ + f'.load_root_file: Loading with uproot from file ' + root_path, 'yellow')
    cprint( __name__ + f'.load_root_file: entrystart = {entrystart}, entrystop = {entrystop}')

    file = uproot4.open(root_path)
    events = file["ntuplizer"]["tree"]

    print(events)
    print(events.name)
    print(events.title)
    #cprint(__name__ + f'.load_root_file: events.numentries = {events.numentries}', 'green')

    ### All variables
    if VARS is None:
        VARS = events.keys() #[x for x in events.keys()]
    #VARS_scalar = [x.decode() for x in events.keys() if b'image_' not in x]
    #print(VARS)

    # Check is it MC (based on the first event)
    X_test = events.arrays('is_mc', entry_start=entrystart, entry_stop=entrystop)
    
    isMC   = bool(X_test['0'])
    N      = len(X_test)
    print(__name__ + f'.load_root_file: isMC: {isMC}')

    # Now read the data
    print(__name__ + '.load_root_file: Loading root file variables ...')

    # --------------------------------------------------------------
    # Important to lead variables one-by-one (because one single np.assarray call takes too much RAM)

    # Needs to be of object type numpy array to hold arbitrary objects (such as jagged arrays) !
    X = np.empty((N, len(VARS)), dtype=object) 

    for j in tqdm(range(len(VARS))):
        x = events.arrays(VARS[j], library="np", how=list, entry_start=entrystart, entry_stop=entrystop)
        X[:,j] = np.asarray(x)
    # --------------------------------------------------------------
    Y = None


    print(__name__ + f'common: X.shape = {X.shape}')
    showmem()

    prints.printbar()

    # =================================================================
    # *** MC ONLY ***

    if isMC:

        # @@ MC target definition here @@
        cprint(__name__ + f'.load_root_file: Computing MC <targetfunc> ...', 'yellow')
        Y = TARFUNC(events, entrystart=entrystart, entrystop=entrystop, new=True)
        Y = np.asarray(Y).T

        print(__name__ + f'common: Y.shape = {Y.shape}')

        # For info
        labels1 = ['is_e', 'is_egamma']
        aux.count_targets(events=events, names=labels1, entrystart=entrystart, entrystop=entrystop, new=True)

        prints.printbar()

        # @@ MC filtering done here @@
        cprint(__name__ + f'.load_root_file: Computing MC <filterfunc> ...', 'yellow')
        indmc = FILTERFUNC(X=X, VARS=VARS, xcorr_flow=args['xcorr_flow'])

        cprint(__name__ + f'.load_root_file: Prior MC <filterfunc>: {len(X)} events', 'green')
        cprint(__name__ + f'.load_root_file: After MC <filterfunc>: {sum(indmc)} events ', 'green')
        prints.printbar()
        
        
        X = X[indmc]
        Y = Y[indmc].squeeze() # Remove useless dimension
    # =================================================================
    
    # -----------------------------------------------------------------
    # @@ Observable cut selections done here @@
    cprint(colored(__name__ + f'.load_root_file: Computing <cutfunc> ...'), 'yellow')
    cind = CUTFUNC(X=X, VARS=VARS, xcorr_flow=args['xcorr_flow'])
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

    # ** REMEMBER TO CLOSE **
    file.close()

    return X, Y, VARS
