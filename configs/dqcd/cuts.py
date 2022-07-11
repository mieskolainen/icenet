# Basic kinematic fiducial cuts, use only variables available in real data.
#
# m.mieskolainen@imperial.ac.uk, 2022

import awkward as ak
import numpy as np
import numba
import matplotlib.pyplot as plt

from icenet.tools import stx


def cut_nocut(X, ids, isMC, xcorr_flow=False):
    """ No cuts """
    return ak.Array(np.ones(X.shape[0], dtype=np.bool_)) # # Note datatype np.bool_


def cut_fiducial(X, ids, isMC, xcorr_flow=False):
    """ Basic fiducial (kinematic) selections.
    
    Args:
        X:    Number of events N x Number of variables D
        ids:  Variable name array (D)
        isMC: is it MC or Data
    
    Returns:
        Passing indices mask (N)
    """
    
    # Awkward type
    mask    = X['nsv'] >= 1
    mask    = ak.to_numpy(mask)

    # Numpy type
    #cutlist = ['nsv >= 1']
    ## Construct and apply
    #cuts, names = stx.construct_columnar_cuts(X=X, ids=ids, cutlist=cutlist)
    #mask        = stx.apply_cutflow(cut=cuts, names=names, xcorr_flow=xcorr_flow)

    return mask
