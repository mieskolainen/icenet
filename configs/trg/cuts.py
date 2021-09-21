# Basic kinematic fiducial cuts, use only variables available in real data.
#
# m.mieskolainen@imperial.ac.uk, 2021

import numpy as np
import numba
import matplotlib.pyplot as plt

from icenet.tools import stx


def cut_nocut(X, ids, isMC, xcorr_flow=False):
    """ No cuts """
    return np.ones(X.shape[0], dtype=np.bool_) # # Note datatype np.bool_

def cut_fiducial(X, ids, isMC, xcorr_flow=False):
    """ Basic fiducial (kinematic) selections.
    
    Args:
        X:    Number of events N x Number of variables D
        ids:  Variable name array (D)
        isMC: is it MC or Data
    
    Returns:
        ind:   Passing indices (N)
    """
    
    if isMC == 'mode_e1' or isMC == 'mode_e2':
        cutlist = ['e1_l1_pt >= 5',
                   'e2_l1_pt >= 5']
    else:
        cutlist = ['l1_doubleE5 == 1']

    # Construct and apply
    cuts, names = stx.construct_columnar_cuts(X=X, ids=ids, cutlist=cutlist)
    ind         = stx.apply_cutflow(cut=cuts, names=names, xcorr_flow=xcorr_flow)

    return ind
