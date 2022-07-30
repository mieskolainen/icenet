# Basic kinematic fiducial cuts, use only variables available in real data.
#
# m.mieskolainen@imperial.ac.uk, 2022

import awkward as ak
import numpy as np
import numba
import matplotlib.pyplot as plt

from icenet.tools import stx


def cut_nocut(X, xcorr_flow=False):
    """ No cuts """
    return ak.Array(np.ones(len(X), dtype=np.bool_)) # Note datatype np.bool_


def cut_fiducial(X, xcorr_flow=False):
    """ Basic fiducial (kinematic) selections.
    
    Args:
        X:          Awkward jagged array
        isMC:       is it MC or Data
        xcorr_flow: cut N-point cross-correlations

    Returns:
        Passing indices mask (N)
    """
    global O; O = X  # __technical__ due to eval() scope
    
    # Create cut strings
    names = ['O.nsv >= 1',
             'ak.sum(O.sv.dxysig >= 5.0, -1) > 0',
             'ak.sum(O.Jet.pt > 15.0,    -1) > 0',
             'ak.sum(np.abs(O.Jet.eta) < 2.5, -1) > 0']
    
    # Evaluate columnar cuts; Compute cutflow
    cuts  = [eval(names[i], globals()) for i in range(len(names))]
    mask  = stx.apply_cutflow(cut=cuts, names=names, xcorr_flow=xcorr_flow)

    return mask
