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


def load_tree(rootfile, tree, max_num_elements=None, ids=None, library='np'):
    """
    Load ROOT files wrapper function
    
    Args:
        rootfile:          Name of root file paths (string or a list of strings)
        tree:              Tree to read out
        max_num_elements:  Max events to read
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

    Y = uproot.concatenate(files, expressions=load_ids, library=library, max_num_elements=max_num_elements)
    

    return Y
