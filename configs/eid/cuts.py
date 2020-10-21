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
    
    # Fiducial cuts
    #MINPT  = 0.5
    #MAXETA = 2.4
    MINPT  = 0.7
    #MAXPT  = 10
    MAXETA = 1.5
    
    
    # Construct cuts
    cuts  = []
    names = []

    #
    cuts.append( X[:,VARS.index('has_gsf')] == True )
    names.append(f'has_gsf == True')
    #
    cuts.append( X[:,VARS.index('gsf_pt')] > MINPT )
    names.append(f'gsf_pt > {MINPT:0.2f}')
    #
    #cuts.append( X[:,VARS.index('gsf_pt')] < MAXPT )
    #names.append(f'gsf_pt < {MAXPT:0.2f}')
    #
    cuts.append( np.abs(X[:,VARS.index('trk_eta')]) < MAXETA )
    names.append(f'|gsf_eta| < {MAXETA:0.2f}')
    #
    #cuts.append( [(len(X[i,VARS.index('image_clu_eta')]) is not 0) for i in range(X.shape[0])] )
    #names.append(f'len(image_clu_eta) != 0')
    
    
    ind = aux.apply_cutflow(cut=cuts, names=names, xcorr_flow=xcorr_flow)
    return ind

# Add alternative cuts here ...
