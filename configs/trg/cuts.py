# Basic selection cuts, use only variables available in real data.
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import numpy as np
import numba
import matplotlib.pyplot as plt

import icenet.tools.aux as aux

def cut_nocut(X, VARS, xcorr_flow=False):
    """ No cuts """
    return np.ones(X.shape[0], dtype=np.bool_) # # Note datatype np.bool_


def cut_standard(X, VARS, xcorr_flow=False):
    """ Function implements basic selections (cuts).
    Args:
    	X    : # Number of vectors x # Number of variables
    	VARS : Variable name array
    Returns:
    	ind  : Passing indices
    """
    
    cutlist = ['e1_hlt_pms2         < 10000',
               'e1_hlt_invEInvP     < 0.2',
               'e1_hlt_trkDEtaSeed  < 0.01',
               'e1_hlt_trkDPhi      < 0.2',
               'e1_hlt_trkChi2      < 40',
               'e1_hlt_trkValidHits >= 5',
               'e1_hlt_trkNrLayerIT >= 2']
    
    # Construct and apply
    cuts, names = aux.construct_columnar_cuts(X=X, VARS=VARS, cutlist=cutlist)
    ind = aux.apply_cutflow(cut=cuts, names=names, xcorr_flow=xcorr_flow)
    
    return ind
