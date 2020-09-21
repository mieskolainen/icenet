# MC sample filter (exclusion) rules, for treating mixed ROOT trees etc.
#
# Note! Real observable cuts are defined in cuts.py, not here.
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import numpy as np
import numba

import icenet.tools.aux as aux


def filter_nofilter(X, VARS, xcorr_flow=False):
    """ All pass """
    return np.ones(X.shape[0], dtype=np.bool_) # Note datatype np.bool_


def filter_charged(X, VARS, xcorr_flow=False):
    """ Only generator level charged """

    # Construct passing filter
    cut = []
    cut.append( np.abs(X[:, VARS.index('gen_charge')]) == 1 )

    # Apply filters
    names = ['|gen_charge| == 1']
    ind = aux.apply_cutflow(cut=cut, names=names, xcorr_flow=xcorr_flow)

    return ind


def filter_no_egamma(X, VARS, xcorr_flow=False):
    """ No particle flow reconstructed electrons.
    Args:
    	X    : # Number of vectors x # Number of variables
    	VARS : Variable name array
    Returns:
    	ind  : Passing indices
    """

    # Fiducial cuts for the tag-side muon trigger object
    MINPT  = 5.0
    MAXETA = 2.5

    # Construct passing filter
    cut = []
    cut.append( X[:, VARS.index('is_egamma')] == False )
    cut.append( X[:, VARS.index('tag_pt')] > MINPT )
    cut.append( np.abs(X[:, VARS.index('tag_eta')]) < MAXETA )

    # Apply filters
    names = [f'is_egamma == False',
             f'tag_pt > {MINPT:0.2f}',
             f'|tag_eta| < {MAXETA:0.2f}']

    ind   = aux.apply_cutflow(cut=cut, names=names, xcorr_flow=xcorr_flow)

    return ind

# Add alternative filters here ...
