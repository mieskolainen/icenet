# Data filtering rules
#
# Note! Physics observable (fiducial / kinematic) cuts are defined in cuts.py, not here.
#
# m.mieskolainen@imperial.ac.uk, 2022

import awkward as ak
import numpy as np
import numba

from icenet.tools import stx


def filter_nofilter(X, ids, isMC, xcorr_flow=False):
    """ All pass """
    return ak.Array(np.ones(X.shape[0], dtype=np.bool_)) # Note datatype np.bool_


def filter_standard(X, ids, isMC, xcorr_flow=False):
    """ Basic filters.
    
    Args:
    	X    : Number of vectors N x Number of variables D
    	ids  : Variable name array D
        isMC : MC or not
    
    Returns:
        Passing indices mask (N)
    """

    # Awkward type
    mask    = (X['HLT_Mu9_IP6_part0'] == 1) | \
              (X['HLT_Mu9_IP6_part1'] == 1) | \
              (X['HLT_Mu9_IP6_part2'] == 1) | \
              (X['HLT_Mu9_IP6_part3'] == 1) | \
              (X['HLT_Mu9_IP6_part4'] == 1)

    mask    = ak.to_numpy(mask)

    # numpy type
    # cutlist = ['HLT_Mu9_IP6_part0 == 1 OR HLT_Mu9_IP6_part1 == 1 OR HLT_Mu9_IP6_part2 == 1 OR HLT_Mu9_IP6_part3 == 1 OR HLT_Mu9_IP6_part4 == 1']    
    # Construct and apply
    # cuts, names = stx.construct_columnar_cuts(X=X, ids=ids, cutlist=cutlist)
    # mask         = stx.apply_cutflow(cut=cuts, names=names, xcorr_flow=xcorr_flow)
    
    return mask

# Add alternative filters here ...
