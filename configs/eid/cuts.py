# Basic selection cuts, use only variables available in real data.
#
# Mikael Mieskolainen, 2021
# m.mieskolainen@imperial.ac.uk

import numpy as np
import numba

import icenet.tools.aux as aux

def cut_nocut(X, VARS, xcorr_flow=False):
    """ No cuts """
    return np.ones(X.shape[0], dtype=np.bool_) # # Note datatype np.bool_


def cut_standard(X, VARS, xcorr_flow=False):
    """ Function implements basic selections (cuts)

    Args:
        X    : Data matrix (N events x D dimensions)
        VARS : Variable name list (D)
    Returns:
        ind  : Passing indices list
    """
    
    # Fiducial cuts
    MINPT   = 0.7
    MAXETA  = 1.5
    
    # Define cuts (syntax accepts: ==, >=, <=, <, >, !=, ==, AND and OR,
    #                                 also ABS__, POW2__, SQRT__, INV__)
    cutlist = [f'has_gsf     == True' ,
               f'gsf_pt       > {MINPT}',
               f'ABS__trk_eta < {MAXETA}']
    
    # Construct and apply
    cuts, names = aux.construct_columnar_cuts(X=X, VARS=VARS, cutlist=cutlist)
    ind = aux.apply_cutflow(cut=cuts, names=names, xcorr_flow=xcorr_flow)
    
    return ind

# Add alternative cuts here ...
