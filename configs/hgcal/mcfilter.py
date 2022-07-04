# MC sample filter (exclusion) rules, for treating mixed ROOT trees etc.
#
# Note! Real observable cuts are defined in cuts.py, not here.
#
# m.mieskolainen@imperial.ac.uk, 2021

import numpy as np
import numba

from icenet.tools import stx


def filter_nofilter(X, ids, xcorr_flow=False):
    """ All pass """
    return np.ones(X.shape[0], dtype=np.bool_) # Note datatype np.bool_

# Add alternative filters here ...
