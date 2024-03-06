# Dim-by-Dim Factorized Histogram Likelihood Ratio Classifier    
#
# m.mieskolainen@imperial.ac.uk, 2024


import matplotlib.pyplot as plt
import numpy as np

from icenet.tools import aux


def train(X, y, weights, param):
    """ Factorized likelihood classifier training.
    
    Args:
        X:         input data [# vectors x # dimensions]
        y:         target data
        weights:   weighted events
        param:     dictionary for the parameters

    Returns:
        b_pdfs:    background pdfs
        s_pdfs:    signal pdfs
        bin_edges: histogram bin edges
    """

    bin_edges = []
    b_pdfs    = []
    s_pdfs    = []

    # Over dimensions
    for j in range(X.shape[1]):

        x = X[:,j]
        minval = np.percentile(x, param['qmin'])
        maxval = np.percentile(x, param['qmax'])

        bin_edges.append( np.linspace(minval, maxval, param['nbins']) )

        b_pdf, bins, patches = plt.hist(x[y == 0], bin_edges[j], weights = weights[y == 0],
            density = True, histtype = 'step')
        s_pdf, bins, patches = plt.hist(x[y == 1], bin_edges[j], weights = weights[y == 1],
            density = True, histtype = 'step')

        b_pdfs.append(b_pdf)
        s_pdfs.append(s_pdf)

    return b_pdfs, s_pdfs, bin_edges


def predict(X, b_pdfs, s_pdfs, bin_edges, return_prob=True, EPS=1e-12):
    """ Evaluate the likelihood ratio.
    
    Args:
        X:           input data [# vectors x # dimensions]
        b_pdfs:      background pdfs
        s_pdfs:      signal pdfs
        bin_edges:   bin edges
        return_prob: return probability if True, or likelihood ratio
    
    Returns:
        LR:        likelihood ratio, or probability
    """
    
    # Loop over events
    out = np.zeros((X.shape[0]))
    for k in range(X.shape[0]):

        # Log-likelihoods
        b_ll = 0
        s_ll = 0
        
        # Loop over dimensions
        for j in range(X.shape[1]):
            
            x   = X[k,j]

            # Evaluate likelihoods for this dimension
            ind = aux.x2ind([x], bin_edges[j])

            b = b_pdfs[j][ind]
            s = s_pdfs[j][ind]

            # Sum log-likelihoods (a factorized total product likelihood)
            if (b > EPS):
                b_ll += np.log(b)

            if (s > EPS):
                s_ll += np.log(s)

        # Probability
        if return_prob:
            out[k] = np.exp(s_ll) / (np.exp(s_ll) + np.exp(b_ll))

        # Likelihood ratio
        else:
            out[k] = np.exp(s_ll) / np.exp(b_ll)
    
    out[~np.isfinite(out)] = 0
    
    return out
