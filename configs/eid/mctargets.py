# MC only. Supervised training signal class target definitions.
#
# m.mieskolainen@imperial.ac.uk, 2022

import numpy as np
import numba

from icenet.tools import stx

def target_e(X, ids, xcorr_flow=False):
    """ Classification signal target """
    
    # Define cuts
    cutlist = [f'BOOL@is_e== True']
    
    # Construct and apply
    cuts, names = stx.construct_columnar_cuts(X=X, ids=ids, cutlist=cutlist)
    mask        = stx.apply_cutflow(cut=cuts, names=names, xcorr_flow=xcorr_flow)
    
    return mask

def target_egamma(X, ids, xcorr_flow=False):
    """ Classification signal target """
    
    # Define cuts
    cutlist = [f'BOOL@is_egamma == True']
    
    # Construct and apply
    cuts, names = stx.construct_columnar_cuts(X=X, ids=ids, cutlist=cutlist)
    mask        = stx.apply_cutflow(cut=cuts, names=names, xcorr_flow=xcorr_flow)
    
    return mask

# Add alternatives here
# ...
