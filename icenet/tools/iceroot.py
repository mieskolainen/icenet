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
        key:      key to use to get the number of events, if None then automatic
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


def process_tree(events, ids, entry_start=0, entry_stop=None):
    """
    Process uproot tree
    
    Args:
        events:      uproot tree
        ids:         variable names to pick
        entry_start: first event to consider
        entry_stop:  last event to consider
    
    Returns:
        numpy array (with jagged content)
    """
    
    N        = len(events.arrays(ids[0]))
    X_test   = events.arrays(ids[0], entry_start=entry_start, entry_stop=entry_stop)
    X        = np.empty((len(X_test), len(ids)), dtype=object) 
    
    cprint( __name__ + f'.process_tree: Entry_start = {entry_start}, entry_stop = {entry_stop} | total = {N}', 'yellow')
    
    for j in tqdm(range(len(ids))):
        x = events.arrays(ids[j], entry_start=entry_start, entry_stop=entry_stop, library="np", how=list)
        X[:,j] = np.asarray(x)

    return X,ids


def load_tree(rootfile, tree, max_num_elements=None, ids=None, library='np'):
    """
    Load ROOT files using uproot 'concatenate' of files
    
    Args:
        rootfile:          Name of root file paths (string or a list of strings)
        tree:              Tree to read out
        max_num_elements:  Max events to read (NOT IMPLEMENTED)
        ids:               Variable names to read out from the root tree
        library:           Return type 'np' (numpy dict) or 'ak' (awkward) of the array
    
    Returns:
        Tree array of type 'library'
    """

    if type(rootfile) is not list:
        rootfile = [rootfile]

    cprint(__name__ + f'.load_tree: Opening rootfile <{rootfile}> with key <{tree}>', 'yellow')

    files = [rootfile[i] + f':{tree}' for i in range(len(rootfile))]
    
    # ----------------------------------------------------------
    ### Select variables
    events  = uproot.open(files[0])
    all_ids = events.keys()
    events.close()

    load_ids = process_regexp_ids(ids=ids, all_ids=all_ids)

    print(__name__ + f'.load_tree: Loading variables ({len(load_ids)}): \n{load_ids} \n')

    return uproot.concatenate(files, expressions=load_ids, library=library)


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
                if re.match(reg, all_ids[i]) and not chosen[i]:
                    load_ids.append(all_ids[i])
                    chosen[i] = 1

    return load_ids
