# Data filtering / triggering rules
#
# Note! Physics observable (fiducial / kinematic) cuts are defined in cuts.py, not here.
#
# m.mieskolainen@imperial.ac.uk, 2022

import numpy as np
import numba

from icenet.tools import stx


def filter_nofilter(X, isMC=None, class_id=None, xcorr_flow=False):
    """ All pass
    """
    return ak.Array(np.ones(len(X), dtype=np.bool_)) # Note datatype np.bool_


def filter_standard(X, isMC=None, class_id=None, xcorr_flow=False):
    """ Basic filters.
    
    Args:
    	X    : Number of vectors N x Number of variables D
    	ids  : Variable name array D
        isMC : MC or not
    
    Returns:
        Passing indices mask (N)
    """

    names = []


    # __technical__ recast due to eval() scope
    global O; O = X

    # Remove EGamma candidates
    names += ['O.is_egamma == False']

    # MiniAOD only
    names += ['O.is_aod == False']
    
    if isMC: 

        # Emulate trigger in MC
        MINPT   = 7.0
        MAXETA  = 1.5
        names += [
            f'O.tag_pt > {MINPT}',
            f'np.abs(O.tag_eta) < {MAXETA}',
            ]

        # Keep only signal candidates in MC
        if class_id == 1: names += ['O.is_e == True']

        # Keep only fake candidates in MC
        if class_id == 0: names += ['O.is_e == False']

    else:

        # Keep only signal candidates in data
        if class_id == 1: names += ['O.is_e == True']

        # Keep only fake candidates in data
        if class_id == 0: names += ['O.is_e == False']

        # Keep only fake candidates in data (for domain adaptation)
        if class_id == -2: names += ['O.is_e == False']

    print(names)
            
    # Construct and apply
    cuts  = [eval(names[i], globals()) for i in range(len(names))]
    mask        = stx.apply_cutflow(cut=cuts, names=names, xcorr_flow=xcorr_flow)
    return mask
