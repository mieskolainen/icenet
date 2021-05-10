# "Analytic" algorithms, observables, metrics etc.
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import numpy as np
import numba
from scipy import special as special


def fox_wolfram_boost_inv(p, L=10):
    """
    arxiv.org/pdf/1508.03144, (Formula 5.6)

    Args:
        p : list of 4-momentum vectors
        L : maximum angular moment order
    Returns:
        S : list of moments of order 0,1,...,L

    [untested function]
    """
    N  = len(p)
    S  = np.zeros(L+1)
    k  = special.jn_zeros(0, L+1)
    pt = [p[i].pt for i in range(N)]

    dR = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i > j:
                dR[i,j] = p[i].deltaR(p[j])
    
    # Compute moments
    for n in range(len(S)):
        for i in range(N):
            for j in range(N):
                if i >= j: # count also the case i==j
                    S[n] += pt[i] * pt[j] * special.j0(k[n]*dR[i,j])

    return S


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


@numba.njit
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
    a = kt2_i**(2*p)
    b = kt2_j**(2*p) 
    c = (dR2_ij/R**2)
    
    return (a * c) if (a < b) else (b * c)