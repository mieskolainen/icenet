# Basic kinematic fiducial cuts, use only variables available in real data.
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import numba
import matplotlib.pyplot as plt

from icenet.tools import stx


def cut_nocut(X, ids, isMC, xcorr_flow=False):
    """ No cuts """
    return np.ones(X.shape[0], dtype=np.bool_) # # Note datatype np.bool_

