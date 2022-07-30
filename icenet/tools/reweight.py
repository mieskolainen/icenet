# Event sample re-weighting tools
#
# m.mieskolainen@imperial.ac.uk, 2022

import numpy as np
import awkward as ak

import numba
import matplotlib.pyplot as plt
from termcolor import colored, cprint
import copy

from icenet.tools import aux


def compute_ND_reweights(x, y, w, ids, args, pdf=None, EPS=1e-12):
    """
    Compute N-dim reweighting coefficients (currently 2D or 1D supported)
    
    Args:
        x      : training sample input
        y      : training sample (class) labels
        w      : training sample weights
        ids    : variable names of columns of x
        pdf    : pre-computed pdfs
        args   : reweighting parameters in a dictionary
    
    Returns:
        weights: 1D-array of re-weights
    """
    use_ak = True if isinstance(x, ak.Array) else False

    num_classes = len(np.unique(y))

    if use_ak:
        y = copy.deepcopy(ak.to_numpy(y).astype(int))
        w = copy.deepcopy(ak.to_numpy(w).astype(float))

    args = copy.deepcopy(args) # Make sure we make a copy, because we modify args here

    ### Construct parameter names
    paramdict = {}
    for var in ['A', 'B']: # Currently only 2 variables, A, B, supported
        try:
            varname  = args['var_' + var]
            paramdict[var] = varname
        except:
            break
    
    # Compute event-by-event weights
    if args['differential']:
        
        print(__name__ + f".compute_ND_reweights: Reference class: <{args['reference_class']}> (Found {num_classes} classes: {np.unique(y)} from y) | Differential re-weighting using variables: {paramdict}")
        
        ### Collect re-weighting variables
        RV = {}
        for var in paramdict.keys():
            if use_ak:
                RV[var] = ak.to_numpy(x[var]).astype(np.float)
            else:
                RV[var] = x[:, ids.index(paramdict[var])].astype(np.float)

        ### Pre-transform
        for var in paramdict.keys():
            mode = args[f'transform_{var}']

            if   mode == 'log10':
                if np.any(RV[var] <= 0):
                    ind = (RV[var] <= 0)
                    cprint(__name__ + f'.compute_ND_reweights: Variable {var} < 0 (in {np.sum(ind)} elements) in log10 -- truncating to zero', 'red')

                # Transform values
                RV[var] = np.log10(np.maximum(RV[var], EPS))

                # Transform bins
                args[f'bins_{var}'][0] = np.log10(args[f'bins_{var}'][0] + EPS)
                args[f'bins_{var}'][1] = np.log10(args[f'bins_{var}'][1])

            elif mode == 'sqrt':

                # Values
                RV[var] = np.sqrt(np.maximum(RV[var], EPS))

                # Bins
                args[f'bins_{var}'][0] = np.sqrt(args[f'bins_{var}'][0] + EPS)
                args[f'bins_{var}'][1] = np.sqrt(args[f'bins_{var}'][1])

            elif mode == 'square':

                # Values
                RV[var] = RV[var]**2

                # Bins
                args[f'bins_{var}'][0] = (args[f'bins_{var}'][0])**2
                args[f'bins_{var}'][1] = (args[f'bins_{var}'][1])**2

            elif mode == None:
                True
            else:
                raise Except(__name__ + '.compute_ND_reweights: Unknown pre-transform')

        # Binning setup
        binedges = {}
        for var in paramdict.keys():
            if   args[f'binmode_{var}'] == 'linear':
                binedges[var] = np.linspace(
                                     args[f'bins_{var}'][0],
                                     args[f'bins_{var}'][1],
                                     args[f'bins_{var}'][2])

            elif args[f'binmode_{var}'] == 'log10':
                binedges[var] = np.logspace(
                                     np.log10(np.max([args[f'bins_{var}'][0], EPS])),
                                     np.log10(args[f'bins_{var}'][1]),
                                     args[f'bins_{var}'][2], base=10)
            else:
                raise Except(__name__ + ': Unknown re-weight binning mode')
        

        rwparam = {
            'y':               y,
            'reference_class': args['reference_class'],
            'max_reg':         args['max_reg']
        }

        ### Compute 2D-PDFs for each class
        if args['dimension'] == '2D':

            if pdf is None: # Not given by user
                pdf = {}
                for c in range(num_classes):

                    sample_weights = w[y==c] if w is not None else None # Feed in the input weights

                    pdf[c] = pdf_2D_hist(X_A=RV['A'][y==c], X_B=RV['B'][y==c], w=sample_weights, \
                        binedges_A=binedges['A'], binedges_B=binedges['B'])

                pdf['binedges_A']  = binedges['A']
                pdf['binedges_B']  = binedges['B']
                pdf['num_classes'] = num_classes

            weights_doublet = reweightcoeff2D(X_A = RV['A'], X_B = RV['B'], pdf=pdf, **rwparam)
            
        ### Compute geometric mean factorized 1D x 1D product
        elif args['dimension'] == 'pseudo-2D':

            if len(binedges['A']) != len(binedges['B']):
                raise Exception(__name__ + f'.compute_ND_reweights: Error: <pseudo-2D> requires same number of bins for A and B')

            if pdf is None: # Not given by user
                pdf = {}
                for c in range(num_classes):

                    sample_weights = w[y==c] if w is not None else None # Feed in the input weights

                    pdf_A  = pdf_1D_hist(X=RV['A'][y==c], w=sample_weights, binedges=binedges['A'])
                    pdf_B  = pdf_1D_hist(X=RV['B'][y==c], w=sample_weights, binedges=binedges['B'])
                    
                    if   args['pseudo_type'] == 'geometric_mean':
                        pdf[c] = np.sqrt(np.outer(pdf_A, pdf_B)) # (A,B) order gives normal matrix indexing
                    elif args['pseudo_type'] == 'product':
                        pdf[c] = np.outer(pdf_A, pdf_B)
                    else:
                        raise Exception(__name__ + f'.compute_ND_reweights: Unknown <pseudo_type>')

                    # Normalize to discrete density
                    pdf[c] /= np.sum(pdf[c].flatten())

                pdf['binedges_A']  = binedges['A']
                pdf['binedges_B']  = binedges['B']
                pdf['num_classes'] = num_classes

            weights_doublet = reweightcoeff2D(X_A = RV['A'], X_B = RV['B'], pdf=pdf, **rwparam)

        ### Compute 1D-PDFs for each class
        elif args['dimension'] == '1D':
            
            if pdf is None: # Not given by user
                pdf = {}
                for c in range(num_classes):
                    sample_weights = w[y==c] if w is not None else None # Feed in the input weights
                    pdf[c] = pdf_1D_hist(X=RV['A'][y==c], w=sample_weights, binedges=binedges['A'])

                pdf['binedges']    = binedges['A']
                pdf['num_classes'] = num_classes

            weights_doublet = reweightcoeff1D(X = RV['A'], pdf=pdf, **rwparam)
        else:
            raise Exception(__name__ + f'.compute_ND_reweights: Unsupported dimensionality mode <{args["dimension"]}>')
    
    # No differential re-weighting    
    else:
        print(__name__ + f".compute_ND_reweights: Reference class: <{args['reference_class']}> (Found {num_classes} classes {np.unique(y)} from y)")
        weights_doublet = np.zeros((len(x), num_classes))

        for c in range(num_classes):
            sample_weights = w[y==c] if w is not None else 1.0 # Feed in the input weights
            weights_doublet[y == c, c] = sample_weights

    ### Apply class balance equalizing weight
    if (args['equal_frac'] == True):
        cprint(__name__ + f'.Compute_ND_reweights: Equalizing class fractions', 'green')
        weights_doublet = balanceweights(weights_doublet=weights_doublet, reference_class=args['reference_class'], y=y)

    ### Finally map back to 1D-array
    weights = np.sum(weights_doublet, axis=1)

    ### Compute the sum of weights per class for the output print
    frac = np.zeros(num_classes)
    sums = np.zeros(num_classes)
    for c in range(num_classes):
        frac[c], sums[c] = np.sum(y == c), np.sum(weights[y == c])
    
    print(__name__ + f'.compute_ND_reweights: Output sum[y == c] = {frac} || sum[weights[y == c]] = {sums}')
    
    if use_ak:
        return ak.Array(weights), pdf
    else:
        return weights, pdf


def reweightcoeff1D(X, y, pdf, reference_class, equal_frac, max_reg = 1e3, EPS=1e-12) :
    """ Compute N-class density reweighting coefficients.
    
    Args:
        X  :              Observable of interest (N x 1)
        y  :              Class labels (0,1,...) (N x 1)
        pdf:              PDF for each class
        reference_class : e.g. 0 (background) or 1 (signal)
    
    Returns:
        weights for each event
    """
    num_classes = pdf['num_classes']

    # Re-weighting weights
    weights_doublet = np.zeros((X.shape[0], num_classes)) # Init with zeros!!

    # Weight each class against the reference class
    for c in range(num_classes):
        inds = aux.x2ind(X[y == c], pdf['binedges'])
        if c is not reference_class:
            weights_doublet[y == c, c] = pdf[reference_class][inds] / (pdf[c][inds] + EPS)
        else:
            weights_doublet[y == c, c] = 1.0 # Reference class stays intact

    # Maximum weight cut-off regularization
    weights_doublet[weights_doublet > max_reg] = max_reg

    return weights_doublet


def reweightcoeff2D(X_A, X_B, y, pdf, reference_class, max_reg = 1e3, EPS=1E-12):
    """ Compute N-class density reweighting coefficients.
    
    Operates in full 2D without any factorization.
    
    Args:
        X_A :             Observable A of interest (N x 1)
        X_B :             Observable B of interest (N x 1)
        y   :             Signal (1) and background (0) labels (N x 1)
        pdf :             Density histograms for each class
        reference_class : e.g. Background (0) or signal (1)
        max_reg :         Regularize the maximum reweight coefficient
    
    Returns:
        weights for each event
    """
    num_classes = pdf['num_classes']

    # Re-weighting weights
    weights_doublet = np.zeros((X_A.shape[0], num_classes)) # Init with zeros!!

    # Weight each class against the reference class
    for c in range(num_classes):
        inds_A = aux.x2ind(X_A[y == c], pdf['binedges_A'])
        inds_B = aux.x2ind(X_B[y == c], pdf['binedges_B'])
        if c is not reference_class:
            weights_doublet[y == c, c] = pdf[reference_class][inds_A, inds_B] / (pdf[c][inds_A, inds_B] + EPS)
        else:
            weights_doublet[y == c, c] = 1.0 # Reference class stays intact

    # Maximum weight cut-off regularization
    weights_doublet[weights_doublet > max_reg] = max_reg

    return weights_doublet


def pdf_1D_hist(X, w, binedges):
    """ 
    Compute re-weighting 1D pdfs.
    """

    # Take re-weighting variables
    pdf,_,_ = plt.hist(x = X, weights=w, bins = binedges)

    # Make them densities
    pdf  /= np.sum(pdf.flatten())
    return pdf


def pdf_2D_hist(X_A, X_B, w, binedges_A, binedges_B):
    """
    Compute re-weighting 2D pdfs.
    """

    # Take re-weighting variables
    pdf,_,_,_ = plt.hist2d(x = X_A, y = X_B, weights=w, bins = [binedges_A, binedges_B])

    # Make them densities
    pdf  /= np.sum(pdf.flatten())
    return pdf


def balanceweights(weights_doublet, reference_class, y, EPS=1e-12):
    """ Balance N-class weights to sum to equal counts.
    
    Args:
        weights_doublet: N-class event weights (events x classes)
        reference_class: which class gives the reference (integer)
        y : class targets
    Returns:
        weights doublet with new weights per event
    """
    num_classes = weights_doublet.shape[1]
    ref_sum = np.sum(weights_doublet[(y == reference_class), reference_class])

    for i in range(num_classes):
        if i is not reference_class:
            EQ = ref_sum / (np.sum(weights_doublet[y == i, i]) + EPS)
            weights_doublet[y == i, i] *= EQ

    return weights_doublet

