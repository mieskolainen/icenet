# "Analytic" algorithms and metrics
#
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import numpy as np


def ktmetric(kt2_i, kt2_j, dR2_ij, p = -1, R = 1.0):
    """
    kt-algorithm type distance measure.
    
    Args:
        kt2_i      : Particle 1 pt squared
        kt2_j      : Particle 2 pt squared
        delta2_ij  : Angular seperation between particles squared (deta**2 + dphi**2)
        R          : Radius parameter
        
        p =  1     : (p=1) kt-like, (p=0) Cambridge/Aachen, (p=-1) anti-kt like

    Returns:
        distance measure
    """

    return np.min([kt2_i**(2*p), kt2_j**(2*p)]) * (dR2_ij/R**2)
