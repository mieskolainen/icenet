# ROOT file processing tools
#
# m.mieskolainen@imperial.ac.uk, 2021

import numpy as np
import uproot
from tqdm import tqdm
from termcolor import colored, cprint
import re

from icenet.tools.icemap import icemap


def load_tree_stats(rootfile, tree):

    events = uproot.open(rootfile)[tree]

    print(events)
    print(events.name)
    print(events.title)
    #print(events.numentries)


def load_tree(rootfile, tree, entry_start=0, entry_stop=None, ids=None):
    """
    Load ROOT files

    Args:
        rootfile
        tree:
        entry_start:
        entry_stop:
        ids:

    Returns:

    """

    cprint(__name__ + f'.load_tree: Opening rootfile <{rootfile}> with key <{tree}>', 'yellow')

    files = [rootfile[i] + f':{tree}' for i in range(len(rootfile))]
    
    # ----------------------------------------------------------
    ### Select variables
    events = uproot.open(files[0])
    all_ids = events.keys()


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

    print(__name__ + f'.load_tree: All variables     ({len(all_ids)}): \n{all_ids} \n')
    print(__name__ + f'.load_tree: Loading variables ({len(load_ids)}): \n{load_ids} \n')
    # ----------------------------------------------------------

    Y = uproot.concatenate(files, expressions=load_ids, library='np', entry_start=entry_start, entry_stop=entry_stop)
    
    #print(X)

    # Check length
    #X_test = events.arrays(load_ids[0], entry_start=entry_start, entry_stop=entry_stop)
    #N      = len(X_test)
    
    # Needs to be an object type numpy array to hold arbitrary objects (such as jagged arrays) !
    #X = np.empty((N, len(load_ids)), dtype=object) 

    ##for j in tqdm(range(len(load_ids))):
    ##    x = events.arrays(load_ids[j], library="np", how=list, entry_start=entry_start, entry_stop=entry_stop)
    ##    X[:,j] = np.asarray(x)
    ##

    #Y = icemap(x=X, ids=load_ids)

    # Close the file
    events.close()

    return Y