# Basic selection cuts, use only variables available in real data.
#
# m.mieskolainen@imperial.ac.uk, 2021

import numpy as np
import numba

from icenet.tools import stx


def cut_nocut(X, ids, xcorr_flow=False):
    """ No cuts """
    return np.ones(X.shape[0], dtype=np.bool_) # Note datatype np.bool_

# Add alternative cuts here ...
