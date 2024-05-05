# Data filtering rules
#
# Note! Physics observable (fiducial / kinematic) cuts are defined in cuts.py, not here.
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import numba

from icenet.tools import stx


def filter_nofilter(X, ids, isMC, xcorr_flow=False):
    """ All pass """
    return np.ones(X.shape[0], dtype=np.bool_) # Note datatype np.bool_
