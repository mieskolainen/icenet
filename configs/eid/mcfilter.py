# MC sample filter (exclusion) rules, for treating mixed ROOT trees etc.
#
# Note! Real observable cuts are defined in cuts.py, not here.
#
# Mikael Mieskolainen, 2021
# m.mieskolainen@imperial.ac.uk

import numpy as np
import numba

from icenet.tools import stx


def filter_nofilter(X, ids, xcorr_flow=False):
    """ All pass """
    return np.ones(X.shape[0], dtype=np.bool_) # Note datatype np.bool_


def filter_charged(X, ids, xcorr_flow=False):
    """ Only generator level charged """

    # Define cuts
    cutlist = [f'ABS__gen_charge == 1']

    # Construct and apply
    cuts, names = stx.construct_columnar_cuts(X=X, ids=ids, cutlist=cutlist)
    ind         = stx.apply_cutflow(cut=cuts, names=names, xcorr_flow=xcorr_flow)
    
    return ind


def filter_no_egamma(X, ids, xcorr_flow=False):
    """ Basic MC filters.
    Args:
    	X    : Data matrix (N events x D dimensions)
    	ids : Variable name list (D)
    Returns:
    	ind  : Passing indices list
    """
    
    # Fiducial cuts for the tag-side muon trigger object
    MINPT   = 5.0
    MAXETA  = 2.5
    
    # Define cuts (syntax accepts: logic ==, >=, <=, <, >, !=, ==, combinators AND and OR,
    #                              also ABS__, POW2__, SQRT__, INV__)
    cutlist = [f'is_egamma   == False' ,
               f'tag_pt       > {MINPT}',
               f'ABS__tag_eta < {MAXETA}']
    
    # Construct and apply
    cuts, names = stx.construct_columnar_cuts(X=X, ids=ids, cutlist=cutlist)
    ind         = stx.apply_cutflow(cut=cuts, names=names, xcorr_flow=xcorr_flow)

    return ind


# Add alternative filters here ...
