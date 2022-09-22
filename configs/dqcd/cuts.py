# Basic kinematic fiducial cuts, use only variables available in real data.
#
# m.mieskolainen@imperial.ac.uk, 2022

import awkward as ak
import numpy as np
import numba
import matplotlib.pyplot as plt

from icenet.tools import stx


def cut_nocut(X, xcorr_flow=False):
    """ No cuts
    """
    return ak.Array(np.ones(len(X), dtype=np.bool_)) # Note datatype np.bool_


def cut_fiducial(X, xcorr_flow=False):
    """ Basic fiducial (kinematic) selections.
    
    Args:
        X:          Awkward jagged array
        xcorr_flow: cut N-point cross-correlations
    
    Returns:
        Passing indices mask (N)
    """
    global O; O = X  # __technical__ recast due to eval() scope
    
    # Create cut strings
    names = ['ak.sum(np.logical_and(O.Muon.pt >  5.0, np.abs(O.Muon.eta) < 2.4), -1) > 0',
             'ak.sum(np.logical_and(O.Jet.pt  > 15.0, np.abs(O.Jet.eta)  < 2.4), -1) > 0']
             
             #'ak.sum(O.svAdapted.dxysig > 5.0, -1) > 0',
    
    # Evaluate columnar cuts; Compute cutflow
    cuts  = [eval(names[i], globals()) for i in range(len(names))]
    mask  = stx.apply_cutflow(cut=cuts, names=names, xcorr_flow=xcorr_flow)

    return mask
