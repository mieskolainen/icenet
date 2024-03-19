# MC sample filter (exclusion) rules, for treating mixed ROOT trees etc.
#
# Note! Real observable cuts are defined in cuts.py, not here.
#
# m.mieskolainen@imperial.ac.uk, 2022

import numpy as np
import numba

from icenet.tools import stx


def filter_nofilter(X, ids, xcorr_flow=False):
    """ All pass """
    return np.ones(X.shape[0], dtype=np.bool_) # Note datatype np.bool_


def filter_charged(X, ids, xcorr_flow=False):
    """ Only generator level charged """
    
    # Define cuts
    cutlist = [f'ABS@gen_charge > 0.5']
    
    # Construct and apply
    cuts, names = stx.construct_columnar_cuts(X=X, ids=ids, cutlist=cutlist)
    mask        = stx.apply_cutflow(cut=cuts, names=names, xcorr_flow=xcorr_flow)
    
    return mask


def filter_no_egamma(X, ids, xcorr_flow=False):
    """ Basic MC filters.
    Args:
    	X    : Data matrix (N events x D dimensions)
    	ids : Variable name list (D)
    Returns:
        Passing indices mask (N)
    """
    
    # Fiducial cuts for the tag-side muon trigger object
    MINPT   = 5.0
    MAXETA  = 2.5
    
    # Define cuts (syntax accepts: logic ==, >=, <=, <, >, !=, ==, combinators AND and OR,
    #                              also ABS@, POW2@, SQRT@, INV@, BOOL@)
    cutlist = [f'BOOL@is_egamma == False' ,
               f'tag_pt       > {MINPT}',
               f'ABS@tag_eta < {MAXETA}']
    
    # Construct and apply
    cuts, names = stx.construct_columnar_cuts(X=X, ids=ids, cutlist=cutlist)
    mask        = stx.apply_cutflow(cut=cuts, names=names, xcorr_flow=xcorr_flow)

    return mask


# Add alternative filters here ...

# Needed???
#def filter_lptele(X, VARS):
#    """ Recommended filter for studying electron ID.
#    Args:
#       X    : # Number of vectors x # Number of variables
#       VARS : Variable name array
#    Returns:
#       ind  : Passing indices
#    """
#
#    cut = []
#    names = []
#
#    # Remove PF EGamma electrons
#    cut.append( X[:, VARS.index('is_egamma')] == False )
#    names.append('is_egamma == False')
#
#    # Ensure electron object is reconstructed
#    cut.append( X[:, VARS.index('has_ele')] == True )
#    names.append(f'has_ele == True')
#
#    # Fiducial cuts for the tag-side muon trigger object
#    MAXETA = 2.5
#    cut.append( np.abs(X[:, VARS.index('tag_eta')]) < MAXETA )
#    names.append(f'tag_eta < {MAXETA:0.2f}')
#    MINPT  = 5.0
#    cut.append( X[:, VARS.index('tag_pt')] > MINPT )
#    names.append(f'tag_pt > {MINPT:0.2f}')
#
#    # Apply filters
#    ind = aux.apply_cutflow(cut=cut, names=names)
#
#    return ind
