# MC sample filter (exclusion) rules, for treating mixed ROOT trees etc.
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

def filter_mc(X, VARS, xcorr_flow=False):

    cutlist = ['gen_e1_l1_dr  < 0.2',
               'gen_e2_l1_dr  < 0.2',
               'gen_e1_hlt_dr < 0.2']

    cutlist += ['e1_l1_pt >= 6',
                'e2_l1_pt >= 6']
    
    # Construct and apply
    cuts, names = aux.construct_columnar_cuts(X=X, VARS=VARS, cutlist=cutlist)
    ind = aux.apply_cutflow(cut=cuts, names=names, xcorr_flow=xcorr_flow)
    
    return ind

# Add alternative filters here ...
