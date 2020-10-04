# "Analytic" algorithms and metrics
#
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import numpy as np
import numba


def gram_matrix(X, type='dot'):
    """
    Gram matrix for 4-vectors.

    Args:
        X    : Array (list of N) of 4-vectors
        type : Type of Lorentz scalars computed ('dot', 's', 't')
    
    Returns:
        G : Gram matrix (NxN)
    """
    
    N = len(X)
    G = np.zeros((N,N))
    for i in range(len(X)):
        for j in range(len(X)):
            if   type == 'dot':
                G[i,j] = X[i].dot(X[j])   ## 4-dot product
            elif type == 's':
                G[i,j] = (X[i] + X[j]).p2 ## s-type
            elif type == 't':
                G[i,j] = (X[i] - X[j]).p2 ## t-type
            else:
                raise Exception('gram_matrix: Unknown type!')
    return G


def ktmetric(kt2_i, kt2_j, dR2_ij, p = -1, R = 1.0):
    """
    kt-algorithm type distance measure.
    
    Args:
        kt2_i     : Particle 1 pt squared
        kt2_j     : Particle 2 pt squared
        delta2_ij : Angular seperation between particles squared (deta**2 + dphi**2)
        R         : Radius parameter
        
        p =  1    : (p=1) kt-like, (p=0) Cambridge/Aachen, (p=-1) anti-kt like
    
    Returns:
        distance measure
    """

    return np.min([kt2_i**(2*p), kt2_j**(2*p)]) * (dR2_ij/R**2)
