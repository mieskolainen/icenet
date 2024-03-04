# Non-Negative matrix factorization
#
# m.mieskolainen@imperial.ac.uk, 2024

import numba
import numpy as np


@numba.njit
def ML_update_W(V, W, H):
    ''' Multiplicative (EM-type)
    non-negative matrix factorization update for basis components.
    
    Arguments:
        V: (d x n) (dimension x samples)
        W: (d x k) (dimension x dictionary size)
        H: (k x n) (expansion weights for each sample)
    Returns:
        W: (d x k) updated non-negative basis compoments
    '''

    W_new = W * np.dot(V / np.dot(W,H), H.T)
    return W_new / np.sum(W_new, axis=0)    #, keepdims=True)


@numba.njit
def ML_update_H(V, W, H):
    ''' Multiplicative (EM-type) 
    non-negative matrix factorization update for the expansion weights.
    
    Parameters:
        V: (d x n) (dimension x samples)
        W: (d x k) (dimension x dictionary size)
        H: (k x n) (expansion weights for each sample)
    Returns:
        H: (k x n) array of updated weights for each sample
    '''
    return H * np.dot((V / np.dot(W, H)).T, W).T


def ML_nmf(V, k, threshold=1e-8, maxiter=500):
    ''' Non-negative matrix factorization main function.
    
    Arguments:
        V:         (d x n) array (dimension x samples)
        k:         number of components
        threshold: relative error threshold (Frob norm)
        maxiter:   maximum number of iterations
    Returns:
        W: (d x k) array of basis elements
        H: (k x n) array of weights for each observations
    '''
    
    d,n = V.shape
    W   = np.random.rand(d,k) # Initialize with uniform noise
    H   = np.random.rand(k,n)
    prev_error = 1e9
    
    i = 0
    for i in range(maxiter):
        W_ = ML_update_W(V, W,  H)
        H_ = ML_update_H(V, W_, H)
        W, H  = W_, H_
        error = np.linalg.norm(np.dot(W,H) - V) / n

        if np.abs(error - prev_error) / error < threshold: break
        #print(__name__ + f'.ML_nmf: iter {i:3}, cost = {error:0.5E}')
        prev_error = error

    return W, H
