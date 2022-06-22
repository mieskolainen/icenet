# ROOT file processing tools
#
# m.mieskolainen@imperial.ac.uk, 2021

import numpy as np
import uproot
import awkward as ak
from tqdm import tqdm
from termcolor import colored, cprint
import re

from icenet.tools.icemap import icemap


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


def events_to_jagged_numpy(events, ids, entry_start=0, entry_stop=None):
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
    
    cprint( __name__ + f'.process_tree: Entry_start = {entry_start}, entry_stop = {entry_stop} | realized = {N} ({100*N/N_all:0.3f} % | available = {N_all})', 'yellow')
    
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

    cprint(__name__ + f'.load_tree: Opening rootfile <{rootfile}> with key <{tree}>', 'yellow')

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

        param = {'events': events, 'ids': load_ids, 'entry_start': entry_start, 'entry_stop': entry_stop}

        for i in tqdm(range(len(files))):
            events = uproot.open(files[i])
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
