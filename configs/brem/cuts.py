# Basic selection cuts, use only variables available in real data.
#
# m.mieskolainen@imperial.ac.uk, 2021

import numpy as np
import numba

from icenet.tools import stx


def cut_nocut(X, ids, xcorr_flow=False):
    """ No cuts """
    return np.ones(X.shape[0], dtype=np.bool_) # # Note datatype np.bool_


def cut_standard(X, xcorr_flow=False):
    """ Function implements basic selections (cuts)

    Args:
        X    : Data matrix (N events x D dimensions)
        ids : Variable name list (D)
    Returns:
        Passing indices mask (N)
    """
    
    # Fiducial cuts
    MINPT   = 1.0
    MAXETA  = 2.5

    # __technical__ recast due to eval() scope
    global O; O = X
    
    names = [
        'O.has_ele == True' ,
        f'O.gsf_pt > {MINPT}',
        f'np.abs(O.gsf_eta) < {MAXETA}',
        ]
    
    # Construct and apply
    cuts  = [eval(names[i], globals()) for i in range(len(names))]
    mask        = stx.apply_cutflow(cut=cuts, names=names, xcorr_flow=xcorr_flow)
    
    return mask

# Add alternative cuts here ...
