# Basic selection cuts, use only variables available in real data.
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import numpy as np
import numba

import icenet.tools.aux as aux

def cut_nocut(X,VARS):
    """ No cuts """
    return np.ones(X.shape[0], dtype=np.bool_) # # Note datatype np.bool_


def cut_standard(X, VARS):
    """ Function implements basic selections (cuts).
    Args:
    	X    : # Number of vectors x # Number of variables
    	VARS : Variable name array
    Returns:
    	ind  : Passing indices
    """

    # Fiducial cuts
    MINPT  = 0.5
    MAXETA = 2.4

    # Construct cuts
    cut = []
    cut.append( X[:, VARS.index('has_gsf')] == True )
    cut.append( X[:, VARS.index('gsf_pt')]  > MINPT )
    cut.append( np.abs(X[:, VARS.index('trk_eta')]) < MAXETA )

    # Apply cutflow
    names = ['has_gsf == True', f'gsf_pt > {MINPT:0.2f}', f'|gsf_eta| < {MAXETA:0.2f}']
    ind = aux.apply_cutflow(cut=cut, names=names)

    return ind

# Add alternative cuts here ...
