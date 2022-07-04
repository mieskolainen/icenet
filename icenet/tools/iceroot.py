# ROOT file processing tools
#
# m.mieskolainen@imperial.ac.uk, 2021

import numpy as np
import uproot
import awkward as ak
from tqdm import tqdm
from termcolor import colored, cprint
import re

from icenet.tools import io
from icenet.tools import iceroot
from icenet.tools.icemap import icemap


def read_multiple_MC(process_func, processes, root_path, param, class_id, use_conditional):
    """
    Loop over different MC processes

    Args:
        process_func:  data processing function
        processes:     MC processes dictionary (from yaml)
        root_path:     main path of files
        param:         parameters of 'process_func'
        class_id:      class identifier (integer), e.g. 0, 1, 2 ...
    
    Returns:
        X, Y, W, ids
    """

    for i,key in enumerate(processes):

        print(__name__ + f'.read_multiple_MC: {processes[key]}')

        # --------------------

        datasets     = processes[key]['path']
        xs           = processes[key]['xs']
        model_param  = processes[key]['model_param']
        force_xs     = processes[key]['force_xs']

        # --------------------

        rootfile    = io.glob_expand_files(datasets=datasets, datapath=root_path)

        # Load file
        X__,ids     = iceroot.load_tree(rootfile=rootfile, tree=param['tree'],
                        entry_start=param['entry_start'], entry_stop=param['entry_stop'], ids=param['load_ids'], library='np')
        N_before    = X__.shape[0]
        
        # Apply selections
        X__,ids     = process_func(X=X__, ids=ids, **param)
        N_after     = X__.shape[0]

        eff_acc     = N_after / N_before

        print(__name__ + f'.read_multiple_MC: Process <{key}> | efficiency x acceptance = {eff_acc}')

        Y__         = class_id * np.ones(N_after, dtype=int)

        # -------------------------------------------------
        # Visible cross-section weights
        if not force_xs:
            W__ = (np.ones(N_after, dtype=float) / N_after) * (xs * eff_acc) # Sum over W yields --> (eff_acc * xs)
        
        # Force mode is useful e.g. in training with a mix of signals with different eff x acc, but one wants
        # to have equal proportions for e.g. theory conditional MVA training.
        # (then one uses the same xs value for each process in the steering .yaml file)
        else:
            W__ = (np.ones(N_after, dtype=float) / N_after) * xs # Sum over W yields --> xs

        # -------------------------------------------------
        # Add conditional (theory param) variables
        if use_conditional:

            print(__name__ + f'.read_multiple_MC: Adding conditional theory (model) parameters')
            print(X__.shape)

            for var in model_param.keys():

                value = model_param[var]

                # Pick random between [a,b]
                if type(value) == list:
                    col = value[0]  + (value[1] - value[0]) * np.random.rand(X__.shape[0], 1)
                # Pick specific fixed value
                else:
                    if value is None:
                        value = np.nan
                    col = value  * np.ones((X__.shape[0], 1))                
                
                X__ = np.append(X__, col, axis=1)
                ids.append(f'__model_{var}')
        else:
            print(__name__ + f'.read_multiple_MC: Not using conditional theory (model) parameters ')
        # -------------------------------------------------

        # Concatenate processes
        if i == 0:
            X, Y, W = X__, Y__, W__
        else:
            X = np.concatenate((X, X__), axis=0)
            Y = np.concatenate((Y, Y__), axis=0)
            W = np.concatenate((W, W__), axis=0)
        
    return X,Y,W,ids


def load_tree_stats(rootfile, tree, key=None, verbose=False):
    """
    Load the number of events in a list of rootfiles

    Args:
        rootfile: a list of rootfiles
        tree:     tree name to open
        key:      key (variable name) to use to get the number of events, if None then use the first one
        verbose:  verbose output print
    
    Returns:
        number of events
    """

    if type(rootfile) is not list:
        rootfile = [rootfile]

    num_events = np.zeros(len(rootfile), dtype=int)
    
    for i in range(len(rootfile)):
        file     = uproot.open(rootfile[i])
        events   = file[tree]
        key_name = events.keys()[0] if key is None else key    

        num_events[i] = len(events.arrays(key_name))
        
        if verbose:
            print(__name__ + f'.load_tree_stats: {rootfile[i]}')
            print(__name__ + f'.load_tree_stats: keys(): {events.keys()}')
            print(__name__ + f'.load_tree_stats: values(): {events.values()}')
        
        file.close()

    return num_events


def events_to_jagged_numpy(events, ids, entry_start=0, entry_stop=None, label=None):
    """
    Process uproot tree to a jagged numpy (object) array
    
    Args:
        events:      uproot tree
        ids:         variable names to pick
        entry_start: first event to consider
        entry_stop:  last event to consider
    
    Returns:
        X
    """

    N_all  = len(events.arrays(ids[0]))
    X_test = events.arrays(ids[0], entry_start=entry_start, entry_stop=entry_stop)
    N      = len(X_test)
    X      = np.empty((N, len(ids)), dtype=object) 
    
    if label is not None:
        cprint( __name__ + f'.events_to_jagged_numpy: Loading: {label}', 'yellow')
    cprint( __name__ + f'.events_to_jagged_numpy: Entry_start = {entry_start}, entry_stop = {entry_stop} | realized = {N} ({100*N/N_all:0.3f} % | available = {N_all})', 'green')
    
    for j in range(len(ids)):
        x = events.arrays(ids[j], entry_start=entry_start, entry_stop=entry_stop, library="np", how=list)
        X[:,j] = np.asarray(x)

    return X, ids


def load_tree(rootfile, tree, entry_start=0, entry_stop=None, ids=None, library='np'):
    """
    Load ROOT files
    
    Args:
        rootfile:          Name of root file paths (string or a list of strings)
        tree:              Tree to read out
        entry_start:       First event to read per file
        entry_stop:        Last event to read per file
        ids:               Variable names to read out from the root tree
        library:           Return type 'np' (jagged numpy) or 'ak' (awkward) of the array
    
    Returns:
        array of type 'library'
    """

    if type(rootfile) is not list:
        rootfile = [rootfile]

    cprint(__name__ + f'.load_tree: Opening rootfile {rootfile} with a tree key <{tree}>', 'yellow')

    files = [rootfile[i] + f':{tree}' for i in range(len(rootfile))]
    
    # ----------------------------------------------------------
    ### Select variables
    events  = uproot.open(files[0])
    all_ids = events.keys()
    #events.close() # This will cause memmap problems
    
    load_ids = process_regexp_ids(ids=ids, all_ids=all_ids)

    print(__name__ + f'.load_tree: Loading variables ({len(load_ids)}): \n{load_ids} \n')
    print(__name__ + f'.load_tree: Reading {len(files)} root files ...')
    
    if   library == 'np':


        for i in tqdm(range(len(files))):
            events = uproot.open(files[i])
            param = {'events': events, 'ids': load_ids, 'entry_start': entry_start, 'entry_stop': entry_stop, 'label': files[i]}

            if i == 0:
                X,ids  = events_to_jagged_numpy(**param)
            else:
                temp,_ = events_to_jagged_numpy(**param)
                X = np.concatenate((X, temp), axis=0)
            
            if (entry_stop is not None) and (len(X) > entry_stop):
                X = X[0:entry_stop]
                print(__name__ + f'.load_tree: Maximum event count {entry_stop} reached')
                break

        print(X.shape)

        return X,ids
        """
        # uproot.concatenate does not allow (??) to control maximum number of events per file
        
        Y = uproot.concatenate(files, expressions=load_ids, library=library)

        ids = [i for i in Y.keys()]
        X   = np.empty((len(Y[ids[0]]), len(ids)), dtype=object) 
        for i in range(len(ids)):
            X[:,i] = Y[ids[i]]
        """

    elif library == 'ak':

        param = {'entry_start': entry_start, 'entry_stop': entry_stop, 'library': 'ak', 'how': 'zip'}

        for i in tqdm(range(len(files))):
            events = uproot.open(files[i])
            if i == 0:
                X    = events.arrays(load_ids, **param)
            else:
                temp = events.arrays(load_ids, **param)
                X = ak.concatenate((X,temp), axis=0)
                
            if (entry_stop is not None) and (len(X) > entry_stop):
                X = X[0:entry_stop]
                print(__name__ + f'.load_tree: Maximum event count {entry_stop} reached')
                break

        return X

    else:
        raise Exception(__name__ + f'.load_tree: Unknown library support')


def process_regexp_ids(all_ids, ids=None):
    """
    Process regular expressions for variable names

    Args:
        all_ids: all keys in a tree
        ids:     keys to pick, if None, use all keys

    Returns:
        ids matching regular expressions
    """

    if ids is None:
        load_ids = all_ids
    else:
        load_ids = []
        chosen   = np.zeros(len(all_ids))

        # Loop over our input
        for string in ids:

            # Compile regular expression
            reg = re.compile(string)
            
            # Loop over all keys in the tree
            for i in range(len(all_ids)):
                if re.fullmatch(reg, all_ids[i]) and not chosen[i]:
                    load_ids.append(all_ids[i])
                    chosen[i] = 1

    return load_ids
