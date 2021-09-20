# Event sample re-weighting tools
#
# m.mieskolainen@imperial.ac.uk, 2021

import numpy as np
import numba
import matplotlib.pyplot as plt
from termcolor import colored, cprint

from icenet.tools import aux


def compute_ND_reweights(data, args, N_class=2, EPS=1e-12):
    """
    Compute N-dim reweighting coefficients (currently 2D or 1D supported)
    
    Args:
        data   : training data object
        args   : arguments object
    
    Returns:
        weights: array of re-weights
    """

    ### Construct parameter names
    ids = {}
    for var in ['A', 'B']: # Currently only 2 variables, A, B, supported
        try:
            varname  = args['reweight_param']['var_' + var]
            ids[var] = varname
        except:
            break
    print(__name__ + f'.compute_ND_reweights: Using the following variables {ids}')

    
    ### Re-weighting variables
    RV = {}
    for var in ids.keys():
        RV[var] = data.trn.x[:, data.ids.index(ids[var])].astype(np.float)

    ### Pre-transform
    for var in ids.keys():
        mode = args['reweight_param'][f'transform_{var}']

        if   mode == 'log10':

            if np.any(RV[var] <= 0):
                ind = (RV[var] <= 0)
                cprint(__name__ + f'.compute_ND_reweights: Variable {var} < 0 (in {np.sum(ind)} elements) in log10 -- truncating to zero', 'red')

            RV[var] = np.log10(np.maximum(RV[var], EPS))

            # Bins
            args['reweight_param'][f'bins_{var}'][0] = np.log10(args['reweight_param'][f'bins_{var}'][0] + EPS)
            args['reweight_param'][f'bins_{var}'][1] = np.log10(args['reweight_param'][f'bins_{var}'][1])

        elif mode == 'sqrt':
            RV[var] = np.sqrt(np.maximum(RV[var], EPS))

            # Bins
            args['reweight_param'][f'bins_{var}'][0] = np.sqrt(args['reweight_param'][f'bins_{var}'][0])
            args['reweight_param'][f'bins_{var}'][1] = np.sqrt(args['reweight_param'][f'bins_{var}'][1])

        elif mode == 'square':
            RV[var] = RV[var]**2

            # Bins
            args['reweight_param'][f'bins_{var}'][0] = (args['reweight_param'][f'bins_{var}'][0])**2
            args['reweight_param'][f'bins_{var}'][1] = (args['reweight_param'][f'bins_{var}'][1])**2

        elif mode == None:
            True
        else:
            raise Except(__name__ + '.compute_ND_reweights: Unknown pre-transform')

    # Binning setup
    binedges = {}
    for var in ids.keys():
        if   args['reweight_param'][f'binmode_{var}'] == 'linear':
            binedges[var] = np.linspace(
                                 args['reweight_param'][f'bins_{var}'][0],
                                 args['reweight_param'][f'bins_{var}'][1],
                                 args['reweight_param'][f'bins_{var}'][2])

        elif args['reweight_param'][f'binmode_{var}'] == 'log':
            binedges[var] = np.logspace(
                                 np.log10(np.max([args['reweight_param'][f'bins_{var}'][0], EPS])),
                                 np.log10(args['reweight_param'][f'bins_{var}'][1]),
                                 args['reweight_param'][f'bins_{var}'][2], base=10)
        else:
            raise Except(__name__ + ': Unknown re-weight binning mode')
    
    print(__name__ + f".compute_ND_reweights: reference class: <{args['reweight_param']['reference_class']}>")

    # Compute event-by-event weights
    if args['reweight_param']['reference_class'] != -1:
        
        rwparam = {
            'y':               data.trn.y,
            'N_class':         N_class,
            'equal_frac':      args['reweight_param']['equal_frac'],
            'reference_class': args['reweight_param']['reference_class'],
            'max_reg':         args['reweight_param']['max_reg']
        }

        if len(ids) == 2:
        
            ### Compute 2D-PDFs for each class
            pdf = {}
            for c in range(N_class):
                pdf[c] = pdf_2D_hist(X_A=RV['A'][data.trn.y==c], X_B=RV['B'][data.trn.y==c], \
                    binedges_A=binedges['A'], binedges_B=binedges['B'])

            pdf['binedges_A'] = binedges['A']
            pdf['binedges_B'] = binedges['B']

            trn_weights = reweightcoeff2D(X_A = RV['A'], X_B = RV['B'], pdf=pdf, **rwparam)
            
        elif len(ids) == 1:

            ### Compute 1D-PDFs for each class
            pdf = {}
            for c in range(N_class):
                pdf[c] = pdf_1D_hist(X=RV['A'][data.trn.y==c], binedges=binedges['A'])

            pdf['binedges'] = binedges['A']
            trn_weights = reweightcoeff1D(X = RV['A'], pdf=pdf, **rwparam)
        else:
            raise Exception(__name__ + f'.compute_ND_reweights: Unsupported dimensionality {len(ids)}')

    else:
        # No re-weighting
        weights_doublet = np.zeros((data.trn.x.shape[0], N_class))
        for c in range(N_class):    
            weights_doublet[data.trn.y == c, c] = 1
        trn_weights = np.sum(weights_doublet, axis=1)
    
    # Compute the sum of weights per class for the output print
    frac = np.zeros(N_class)
    sums = np.zeros(N_class)
    for c in range(N_class):
        frac[c] = np.sum(data.trn.y == c)
        sums[c] = np.sum(trn_weights[data.trn.y == c])
    
    print(__name__ + f'.compute_ND_reweights: sum[trn.y==c]: {frac}')
    print(__name__ + f'.compute_ND_reweights: sum[trn_weights[trn.y==c]]: {sums}')
    print(__name__ + f'.compute_ND_reweights: [done] \n')
    
    return trn_weights


def reweight_1D(X, pdf, y, N_class=2, reference_class = 0, max_reg = 1E3, EPS=1E-12) :
    """ Compute N-class density reweighting coefficients.
    Args:
        X   :             Input data (# samples)
        pdf :             Dictionary of pdfs for each class
        y   :             Class target data (# samples)
        N_class :         Number of classes
        reference_class : Target class of re-weighting
    
    Returns:
        weights for each event
    """

    # Re-weighting weights
    weights_doublet = np.zeros((X.shape[0], N_class)) # Init with zeros!!

    # Weight each class against the reference class
    for c in range(N_class):
        inds = aux.x2ind(X[y == c], pdf['binedges'])
        if c is not reference_class:
            weights_doublet[y == c, c] = pdf[reference_class][inds] / (pdf[c][inds] + EPS)
        else:
            weights_doublet[y == c, c] = 1 # Reference class stays intact

    # Maximum weight cut-off regularization
    weights_doublet[weights_doublet > max_reg] = max_reg

    # Save weights
    weights_doublet[y == 0, 0] = C0
    weights_doublet[y == 1, 1] = C1

    return weights_doublet


def reweightcoeff1D(X, y, pdf, N_class=2, reference_class = 0, equal_frac = True, max_reg = 1e3) :
    """ Compute N-class density reweighting coefficients.
    
    Args:
        X  :   Observable of interest (N x 1)
        y  :   Class labels (0,1,...) (N x 1)
        pdf:   PDF for each class
        N_class : Number of classes
        equal_frac:  equalize class fractions
        reference_class : e.g. 0 (background) or 1 (signal)
    
    Returns:
        weights for each event
    """
    weights_doublet = reweight_1D(X=X, pdf=pdf, y=y, N_class=N_class, reference_class=reference_class, max_reg=max_reg)

    # Apply class balance equalizing weight
    if (equal_frac == True):
        weights_doublet = balanceweights(weights_doublet=weights_doublet, y=y)

    # Get 1D array
    weights = np.sum(weights_doublet, axis=1)

    return weights


def reweightcoeff2DFP(X_A, X_B, y, pdf_A, pdf_B, N_class=2, reference_class = 0,
    equal_frac = True, max_reg = 1e3) :
    """ Compute N-class density reweighting coefficients.
    
    Operates in 2D with FACTORIZED PRODUCT marginal 1D distributions.
    
    Args:
        X_A   :  Observable of interest (N x 1)
        X_B   :  Observable of interest (N x 1)
        y     :  Signal (1) and background (0) targets
        pdf_A :  Density of observable A
        pdf_B :  Density of observable B
        N_class: Number of classes
        reference_class: e.g. 0 (background) or 1 (signal)
        equal_frac:      Equalize integrated class fractions
        max_reg:         Maximum weight regularization
    
    Returns:
        weights for each event
    """

    weights_doublet_A = reweight_1D(X=X_A, pdf=pdf_A, N_class=N_class, y=y, reference_class=reference_class, max_reg=max_reg)
    weights_doublet_B = reweight_1D(X=X_B, pdf=pdf_B, N_class=N_class, y=y, reference_class=reference_class, max_reg=max_reg)

    # Factorized product
    weights_doublet   = weights_doublet_A * weights_doublet_B

    # Apply class balance equalizing weight
    if (equal_frac == True):
        weights_doublet = balanceweights(weights_doublet=weights_doublet, reference_class=reference_class, y=y)

    # Get 1D array
    weights = np.sum(weights_doublet, axis=1)

    return weights


def reweightcoeff2D(X_A, X_B, y, pdf, N_class=2, reference_class = 0, equal_frac = True, max_reg = 1e3, EPS=1E-12) :
    """ Compute N-class density reweighting coefficients.
    
    Operates in full 2D without factorization.

    Args:
        X_A : Observable A of interest (N x 1)
        X_B : Observable B of interest (N x 1)
        y   : Signal (1) and background (0) labels (N x 1)
        pdf : Density histograms for each class
        N_class :         Number of classes
        reference_class : e.g. Background (0) or signal (1)
        equal_frac :      Equalize class fractions
        max_reg :         Regularize the maximum reweight coefficient
    
    Returns:
        weights for each event
    """
    
    # Re-weighting weights
    weights_doublet = np.zeros((X_A.shape[0], N_class)) # Init with zeros!!

    # Weight each class against the reference class
    for c in range(N_class):
        inds_A = aux.x2ind(X_A[y == c], pdf['binedges_A'])
        inds_B = aux.x2ind(X_B[y == c], pdf['binedges_B'])
        if c is not reference_class:
            weights_doublet[y == c, c] = pdf[reference_class][inds_A, inds_B] / (pdf[c][inds_A, inds_B] + EPS)
        else:
            weights_doublet[y == c, c] = 1 # Reference class stays intact

    # Maximum weight cut-off regularization
    weights_doublet[weights_doublet > max_reg] = max_reg

    # Apply class balance equalizing weight
    if (equal_frac == True):
        weights_doublet = balanceweights(weights_doublet=weights_doublet, reference_class=reference_class, y=y)

    # Get 1D array
    weights = np.sum(weights_doublet, axis=1)
    return weights


def pdf_1D_hist(X, binedges):
    """ 
    Compute re-weighting 1D pdfs.
    """

    # Take re-weighting variables
    pdf,_,_ = plt.hist(x = X, bins = binedges)

    # Make them densities
    pdf  /= np.sum(pdf.flatten())
    return pdf


def pdf_2D_hist(X_A, X_B, binedges_A, binedges_B):
    """
    Compute re-weighting 2D pdfs.
    """

    # Take re-weighting variables
    pdf,_,_,_ = plt.hist2d(x = X_A, y = X_B, bins = [binedges_A, binedges_B])

    # Make them densities
    pdf  /= np.sum(pdf.flatten())
    return pdf


@numba.njit
def balanceweights(weights_doublet, reference_class, y, EPS=1e-12):
    """ Balance N-class weights to sum to equal counts.
    
    Args:
        weights_doublet: N-class event weights (events x classes)
        reference_class: which class gives the reference (integer)
        y : class targets
    Returns:
        weights doublet with new weights per event
    """
    N = weights_doublet.shape[1]
    ref_sum = np.sum(weights_doublet[(y == reference_class), reference_class])

    for i in range(N):
        if i is not reference_class:
            EQ = ref_sum / (np.sum(weights_doublet[y == i, i]) + EPS)
            weights_doublet[y == i, i] *= EQ

    return weights_doublet

