# Basic selection cuts, use only variables available in real data.
#
# m.mieskolainen@imperial.ac.uk, 2021

import numpy as np
import numba

from icenet.tools import stx


def cut_nocut(X, ids, xcorr_flow=False):
    """ No cuts """
    return np.ones(X.shape[0], dtype=np.bool_) # # Note datatype np.bool_


def cut_standard(X, ids, xcorr_flow=False):
    """ Function implements basic selections (cuts)

    Args:
        X    : Data matrix (N events x D dimensions)
        ids : Variable name list (D)
    Returns:
        Passing indices mask (N)
    """
    
    # Fiducial cuts
    MINPT   = 0.7
    MAXETA  = 1.5
    
    # Define cuts (syntax accepts: logic ==, >=, <=, <, >, !=, ==, combinators AND and OR,
    #                              also ABS@, POW2@, SQRT@, INV@, BOOL@)
    cutlist = [f'BOOL@has_gsf == True' ,
               f'gsf_pt       > {MINPT}',
               f'ABS@trk_eta < {MAXETA}']
    
    # Construct and apply
    cuts, names = stx.construct_columnar_cuts(X=X, ids=ids, cutlist=cutlist)
    mask        = stx.apply_cutflow(cut=cuts, names=names, xcorr_flow=xcorr_flow)
    
    return mask

# Add alternative cuts here ...
