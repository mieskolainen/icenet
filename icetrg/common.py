# Common input & data reading routines for the HLT electron trigger studies
# 
# Mikael Mieskolainen, 2021
# m.mieskolainen@imperial.ac.uk


import numpy as np
import uproot
from tqdm import tqdm
import psutil
import copy

from termcolor import colored, cprint

from icenet.tools import io
from icenet.tools import aux
from icenet.tools import plots
from icenet.tools import prints
from icenet.tools import process


# GLOBALS
from configs.trg.mvavars import *
from configs.trg.cuts import *
from configs.trg.filter import *



def init(MAXEVENTS=None):
    """ Initialize electron HLT trigger data.
	
    Args:
        Implicit commandline and yaml file input.
    
    Returns:
        jagged array data, arguments
    """
    
    args, cli = process.read_config('./configs/trg')
    features  = globals()[args['imputation_param']['var']]
    
    
    # --------------------------------------------------------------------
    ### SET GLOBALS (used only in this file)
    global ARGS
    ARGS = args

    if MAXEVENTS is not None:
        ARGS['MAXEVENTS'] = MAXEVENTS
    
    print(__name__ + f'.init: inputvar   =  {args["inputvar"]}')
    print(__name__ + f'.init: cutfunc    =  {args["cutfunc"]}')
    #print(__name__ + f'.init: targetfunc =  {args["targetfunc"]}')
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


def showmem():
    cprint(__name__ + f""".load_root_file: Process RAM usage: {io.process_memory_use():0.2f} GB 
        [total RAM in use {psutil.virtual_memory()[2]} %]""", 'red')


def load_root_file_new(root_path, ids=None, entrystart=0, entrystop=None, class_id = [], args=None):
    """ Loads the root file with signal events from MC and background from DATA.
    
    Args:
        root_path : paths to root files
        class_id  : class ids
    
    Returns:
        X,Y       : input, output matrices
        ids      : variable names
    """

    # -----------------------------------------------
    # ** GLOBALS **

    if args is None:
        args = ARGS

    if entrystop is None:
        entrystop = args['MAXEVENTS']

    # -----------------------------------------------

    
    # =================================================================
    # *** MC ***

    X_MC, VARS_MC     = process_MC(root_path, ids, entrystart, entrystop, class_id, args)

    ind = []
    for name in NEW_VARS:
        ind.append(VARS_MC.index(name))

    X_MC = X_MC[:, ind]
    Y_MC = np.ones(X_MC.shape[0])


    # =================================================================
    # *** DATA ***

    X_DATA, VARS_DATA = process_DATA(root_path, ids, entrystart, entrystop, class_id, args)

    ind = []
    for name in NEW_VARS:
        new_name = name.replace("e1_", "") # Data tree with different variables
        ind.append(VARS_DATA.index(new_name))
    
    X_DATA = X_DATA[:, ind]
    Y_DATA = np.zeros(X_DATA.shape[0])


    # =================================================================
    # Finally combine

    X = np.concatenate((X_MC, X_DATA), axis=0)
    Y = np.concatenate((Y_MC, Y_DATA), axis=0)


    return X, Y, NEW_VARS


def process_DATA(root_path, ids, entrystart, entrystop, class_id, args):

    CUTFUNC    = globals()[args['cutfunc']]
    FILTERFUNC = globals()[args['filterfunc']]
    

    rootfile = f'{root_path}/{args["datafile"]}'


    ### From root trees
    print('\n')
    cprint( __name__ + f'.process_DATA: Loading with uproot from file "{rootfile}"', 'yellow')
    cprint( __name__ + f'.process_DATA: entrystart = {entrystart}, entrystop = {entrystop}')
    
    file   = uproot.open(rootfile)
    events = file["tree"]
    
    
    ### All variables
    if ids is None:
        ids = events.keys() #[x for x in events.keys()]
    print(ids)

    print(events)
    print(events.name)
    print(events.title)
    #cprint(__name__ + f'.load_root_file: events.numentries = {events.numentries}', 'green')

    
    # Check length
    X_test = events.arrays(ids[0], entry_start=entrystart, entry_stop=entrystop)
    N      = len(X_test)
    
    # Now read the data
    print(__name__ + '.process_DATA: Loading root file variables ...')

    # --------------------------------------------------------------
    # Important to take variables one-by-one (because one single np.assarray call takes too much RAM)

    # Needs to be an object type numpy array to hold arbitrary objects (such as jagged arrays) !
    X = np.empty((N, len(ids)), dtype=object) 

    for j in tqdm(range(len(ids))):
        x = events.arrays(ids[j], library="np", how=list, entry_start=entrystart, entry_stop=entrystop)
        X[:,j] = np.asarray(x)
    # --------------------------------------------------------------
    #Y = np.random.randint(2, size=N)
    #Y = np.ones((N,1)) # Signal

    print(__name__ + f'common: X.shape = {X.shape}')
    showmem()

    prints.printbar()

    # ----------------------------------
    # @@ MC filtering done here @@
    cprint(__name__ + f'.process_DATA: Computing DATA <filterfunc> ...', 'yellow')
    ind = FILTERFUNC(X=X, ids=ids, isMC=False, xcorr_flow=args['xcorr_flow'])

    cprint(__name__ + f'.process_DATA: Prior DATA <filterfunc>: {len(X)} events', 'green')
    cprint(__name__ + f'.process_DATA: After DATA <filterfunc>: {sum(ind)} events ', 'green')
    prints.printbar()

    X = X[ind]
    
    # ----------------------------------
    # @@ DATA cuts done here @@

    # @@ Observable cut selections done here @@
    cprint(colored(__name__ + f'.process_DATA: Computing <cutfunc> ...'), 'yellow')
    cind = CUTFUNC(X=X, ids=ids, isMC=False, xcorr_flow=args['xcorr_flow'])
    # -----------------------------------------------------------------
    
    N_before = X.shape[0]

    ### Select events
    X = X[cind]
    #Y = Y[cind]

    N_after = X.shape[0]
    cprint(__name__ + f".process_DATA: Prior <cutfunc> selections: {N_before} events ", 'green')
    cprint(__name__ + f".process_DATA: Post  <cutfunc> selections: {N_after} events ({N_after / N_before:.3f})", 'green')
    print('')

    showmem()
    prints.printbar()

    # ** REMEMBER TO CLOSE **
    file.close()

    return X, ids


def process_MC(root_path, ids, entrystart, entrystop, class_id, args):


    CUTFUNC    = globals()[args['cutfunc']]
    FILTERFUNC = globals()[args['filterfunc']]


    rootfile = f'{root_path}/{args["mcfile"]}'
    
    ### From root trees
    print('\n')
    cprint( __name__ + f'.process_MC: Loading with uproot from file "{rootfile}"', 'yellow')
    cprint( __name__ + f'.process_MC: entrystart = {entrystart}, entrystop = {entrystop}')
    
    file   = uproot.open(rootfile)
    events = file["tree"]
    
    
    ### All variables
    if ids is None:
        ids = events.keys() #[x for x in events.keys()]
    print(ids)

    print(events)
    print(events.name)
    print(events.title)
    #cprint(__name__ + f'.load_root_file: events.numentries = {events.numentries}', 'green')

    
    # Check length
    X_test = events.arrays(ids[0], entry_start=entrystart, entry_stop=entrystop)
    N      = len(X_test)
    
    # Now read the data
    print(__name__ + '.process_MC: Loading root file variables ...')

    # --------------------------------------------------------------
    # Important to take variables one-by-one (because one single np.assarray call takes too much RAM)

    # Needs to be an object type numpy array to hold arbitrary objects (such as jagged arrays) !
    X = np.empty((N, len(ids)), dtype=object) 

    for j in tqdm(range(len(ids))):
        x = events.arrays(ids[j], library="np", how=list, entry_start=entrystart, entry_stop=entrystop)
        X[:,j] = np.asarray(x)
    # --------------------------------------------------------------
    #Y = np.random.randint(2, size=N)
    #Y = np.ones((N,1)) # Signal

    print(__name__ + f'common: X.shape = {X.shape}')
    showmem()

    prints.printbar()

    # ----------------------------------
    # @@ MC filtering done here @@
    cprint(__name__ + f'.process_MC: Computing MC <filterfunc> ...', 'yellow')
    ind = FILTERFUNC(X=X, ids=ids, isMC=True, xcorr_flow=args['xcorr_flow'])

    cprint(__name__ + f'.process_MC: Prior MC <filterfunc>: {len(X)} events', 'green')
    cprint(__name__ + f'.process_MC: After MC <filterfunc>: {sum(ind)} events ', 'green')
    prints.printbar()

    X = X[ind]
    
    # ----------------------------------
    # @@ MC cuts done here @@

    # @@ Observable cut selections done here @@
    cprint(colored(__name__ + f'.process_MC: Computing <cutfunc> ...'), 'yellow')
    cind = CUTFUNC(X=X, ids=ids, isMC=True, xcorr_flow=args['xcorr_flow'])
    # -----------------------------------------------------------------
    
    N_before = X.shape[0]

    ### Select events
    X = X[cind]
    #Y = Y[cind]

    N_after = X.shape[0]
    cprint(__name__ + f".process_MC: Prior <cutfunc> selections: {N_before} events ", 'green')
    cprint(__name__ + f".process_MC: Post  <cutfunc> selections: {N_after} events ({N_after / N_before:.3f})", 'green')
    print('')

    showmem()
    prints.printbar()

    # ** REMEMBER TO CLOSE **
    file.close()

    return X, ids


def splitfactor(data, args):
    """
    Split electron ID data into different datatypes.
    
    Args:
        data:  jagged arrays
        args:  arguments dictionary
    
    Returns:
        scalar (vector) data
        kinematic data
    """
    
    ### Pick kinematic variables out
    k_ind, k_vars    = io.pick_vars(data, KINEMATIC_ID)
    
    data_kin         = copy.deepcopy(data)
    data_kin.trn.x   = data.trn.x[:, k_ind].astype(np.float)
    data_kin.val.x   = data.val.x[:, k_ind].astype(np.float)
    data_kin.tst.x   = data.tst.x[:, k_ind].astype(np.float)
    data_kin.ids    = k_vars

    ### Pick active scalar variables out
    s_ind, s_vars = io.pick_vars(data, globals()[args['inputvar']])
    
    data.trn.x    = data.trn.x[:, s_ind].astype(np.float)
    data.val.x    = data.val.x[:, s_ind].astype(np.float)
    data.tst.x    = data.tst.x[:, s_ind].astype(np.float)
    data.ids     = s_vars

    return data, data_kin
