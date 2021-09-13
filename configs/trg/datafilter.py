# Data filtering rules
#
# Note! In general observable cuts are defined in cuts.py, not here.
#
# Mikael Mieskolainen, 2021
# m.mieskolainen@imperial.ac.uk

import numpy as np
import numba

import icenet.tools.aux as aux


def filter_nofilter(X, VARS, xcorr_flow=False):
    """
    All pass
    """
    return np.ones(X.shape[0], dtype=np.bool_) # Note datatype np.bool_

def filter_data(X, VARS, xcorr_flow=False):
    """ Basic data filters.

    Args:
    	X    : # Number of vectors x # Number of variables
    	VARS : Variable name array

    Returns:
    	ind  : Passing indices
    """

    cutlist = ['isgjson     == 1',
               'l1_doubleE6 == 1']
    
    # Construct and apply
    cuts, names = aux.construct_columnar_cuts(X=X, VARS=VARS, cutlist=cutlist)
    ind = aux.apply_cutflow(cut=cuts, names=names, xcorr_flow=xcorr_flow)

    return ind

# Add alternative filters here ...
