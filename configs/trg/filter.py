# Data filtering rules
#
# Note! Physics observable (fiducial / kinematic) cuts are defined in cuts.py, not here.
#
# m.mieskolainen@imperial.ac.uk, 2021

import numpy as np
import numba

from icenet.tools import stx


def filter_nofilter(X, ids, isMC, xcorr_flow=False):
    """ All pass """
    return np.ones(X.shape[0], dtype=np.bool_) # Note datatype np.bool_


def filter_standard(X, ids, isMC, xcorr_flow=False):
    """ Basic filters.
    
    Args:
    	X    : Number of vectors N x Number of variables D
    	ids  : Variable name array D
        isMC : MC or not
    
    Returns:
        Passing indices mask (N)
    """

    if   isMC == 'mode_e1':
        cutlist = ['gen_e1_l1_dr  < 0.2',
                   'gen_e2_l1_dr  < 0.2',
                   'gen_e1_hlt_dr < 0.2']

    elif isMC == 'mode_e2':
        cutlist = ['gen_e1_l1_dr  < 0.2',
                   'gen_e2_l1_dr  < 0.2',
                   'gen_e1_hlt_dr < 0.2',
                   'gen_e2_hlt_dr < 0.2']
    
    elif isMC == 'data':
        cutlist = ['BOOL@isgjson == True']
    
    else:
        raise Exception(__name__ + '.filter_standard: Unknown isMC mode')

    # Construct and apply
    cuts, names = stx.construct_columnar_cuts(X=X, ids=ids, cutlist=cutlist)
    mask        = stx.apply_cutflow(cut=cuts, names=names, xcorr_flow=xcorr_flow)
    
    return mask

# Add alternative filters here ...
