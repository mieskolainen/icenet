# ROOT file processing tools
#
# m.mieskolainen@imperial.ac.uk, 2021

import numpy as np
import uproot
from tqdm import tqdm
from termcolor import colored, cprint

from icenet.tools.icemap import icemap


def load_tree_stats(rootfile, tree):

    events = uproot.open(rootfile)[tree]

    print(events)
    print(events.name)
    print(events.title)
    print(events.numentries)


def load_tree(rootfile, tree, entry_start=0, entry_stop=None, ids=None):
    
    events = uproot.open(rootfile)[tree]
    
    cprint(__name__ + f'.load_tree: Opening rootfile <{rootfile}> with key <{tree}>', 'yellow')

    ### All variables
    if ids is None:
        ids = events.keys() #[x for x in events.keys()]
    
    # Check length
    X_test = events.arrays(ids[0], entry_start=entry_start, entry_stop=entry_stop)
    N      = len(X_test)
    
    # Needs to be an object type numpy array to hold arbitrary objects (such as jagged arrays) !
    X = np.empty((N, len(ids)), dtype=object) 

    for j in tqdm(range(len(ids))):
        x = events.arrays(ids[j], library="np", how=list, entry_start=entry_start, entry_stop=entry_stop)
        X[:,j] = np.asarray(x)

    Y = icemap(x=X, ids=ids)

    # Close the file
    events.close()

    return Y