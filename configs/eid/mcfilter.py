# MC sample filter (exclusion) rules, for treating mixed ROOT trees etc.
#
# Note! Real observable cuts are defined in cuts.py, not here.
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import numpy as np
import numba

import icenet.tools.aux as aux


def filter_nofilter(X, VARS):
    """ All pass """
    return np.ones(X.shape[0], dtype=np.bool_) # Note datatype np.bool_


def filter_no_egamma(X, VARS):
    """ Function implements MC filtering.
    Args:
    	X    : # Number of vectors x # Number of variables
    	VARS : Variable name array
    Returns:
    	ind  : Passing indices
    """

    # Construct passing filter
    cut = []
    cut.append( X[:, VARS.index('is_egamma')] == False )

    # Apply filters
    names = ['is_egamma == False']
    ind = aux.apply_cutflow(cut=cut, names=names)

    return ind

# Add alternative filters here ...
