# ROOT file processing tools
#
# m.mieskolainen@imperial.ac.uk, 2022

import numpy as np
import uproot
import awkward as ak
from tqdm import tqdm
from termcolor import colored, cprint
import re

from icenet.tools import io
from icenet.tools import aux
from icenet.tools import iceroot
from icenet.tools.icemap import icemap


def read_multiple_MC(process_func, processes, root_path, param, class_id):
    """
    Loop over different MC processes

    Args:
        process_func:  data processing function
        processes:     MC processes dictionary (from yaml)
        root_path:     main path of files
        param:         parameters of 'process_func'
        class_id:      class identifier (integer), e.g. 0, 1, 2 ...
    
    Returns:
        X, Y, W, ids   (awkward array format)
    """

    info = {}

    for i,key in enumerate(processes):

        print(__name__ + f'.read_multiple_MC: {processes[key]}')

        # --------------------
        
        datasets        = processes[key]['path']
        xs              = processes[key]['xs']
        model_param     = processes[key]['model_param']
        force_xs        = processes[key]['force_xs']
        maxevents_scale = processes[key]['maxevents_scale']

        # --------------------

        rootfile    = io.glob_expand_files(datasets=datasets, datapath=root_path)

        # Custom scale event statistics
        maxevents   = np.max([1, int(param['maxevents'] * maxevents_scale)])
        
        # Load file
        X__, ids    = iceroot.load_tree(rootfile=rootfile, tree=param['tree'],
                        entry_start=param['entry_start'], entry_stop=param['entry_stop'],
                        maxevents=maxevents, ids=param['load_ids'], library='ak')
        
        N_before = len(X__)
        
        # Apply selections
        X__,ids,stats = process_func(X=X__, ids=ids, **param)
        N_after  = len(X__)

        eff_acc  = N_after / N_before

        print(__name__ + f'.read_multiple_MC: Process <{key}> | efficiency x acceptance = {eff_acc:0.6f}')

        Y__ = class_id * ak.Array(np.ones(N_after, dtype=int))

        # -------------------------------------------------
        # Visible cross-section weights
        if not force_xs:
            # Sum over W yields --> (eff_acc * xs)
            W__ = (ak.Array(np.ones(N_after, dtype=float)) / N_after) * (xs * eff_acc)
        
        # 'force_xs' mode is useful e.g. in training with a mix of signals with different eff x acc, but one wants
        # to have equal proportions for e.g. theory conditional MVA training.
        # (then one uses the same input 'xs' value for each process in the steering .yaml file)
        else:
            # Sum over W yields --> xs
            W__ = (ak.Array(np.ones(N_after, dtype=float)) / N_after) * xs

        # Save statistics information
        info[key] = {'yaml': processes[key], 'cut_stats': stats, 'eff_acc': eff_acc}

        # -------------------------------------------------
        # Add conditional (theory param) variables
        
        print(__name__ + f'.read_multiple_MC: Adding conditional theory (model) parameters')
        
        for var in model_param.keys():
                
            value = model_param[var]
            
            # Pick random between [a,b]
            if type(value) == list:
                col = value[0]  + (value[1] - value[0]) * ak.Array(np.random.rand(len(X__), 1))
            # Pick specific fixed value
            else:
                if value is None:
                    value = np.nan
                col = value  * ak.Array(np.ones((len(X__), 1)).astype(float))         
            
            # Create new 'record' (column) to ak-array
            col_name      = f'MODEL_{var}'
            X__[col_name] = col
            ids.append(col_name)
        # -------------------------------------------------

        # Concatenate processes
        if i == 0:
            X, Y, W = X__, Y__, W__
        else:
            X = ak.concatenate((X, X__), axis=0)
            Y = ak.concatenate((Y, Y__), axis=0)
            W = ak.concatenate((W, W__), axis=0)
        
    return X,Y,W,ids,info


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


def load_tree(rootfile, tree, entry_start=0, entry_stop=None, maxevents=None, ids=None, library='np'):
    """
    Load ROOT files
    
    Args:
        rootfile:          Name of root file paths (string or a list of strings)
        tree:              Tree to read out
        entry_start:       First event to read per file
        entry_stop:        Last event to read per file
        maxevents:          Maximum number of events in total (over all files)
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
    
    load_ids = aux.process_regexp_ids(ids=ids, all_ids=all_ids)

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
            
            if (maxevents is not None) and (len(X) > maxevents):
                X = X[0:maxevents]
                cprint(__name__ + f'.load_tree: Maximum event count {maxevents} reached', 'red')
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
            
            if (maxevents is not None) and (len(X) > maxevents):
                X = X[0:maxevents]
                cprint(__name__ + f'.load_tree: Maximum event count {maxevents} reached', 'red')
                break

        return X, ak.fields(X)
        
    else:
        raise Exception(__name__ + f'.load_tree: Unknown library support')
