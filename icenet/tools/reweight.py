# Event sample re-weighting tools
#
# m.mieskolainen@imperial.ac.uk, 2023

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
    for i in range(len(args['var'])):
        try:
            varname  = args['var'][i]
            paramdict[str(i)] = varname
        except:
            break
    
    # Compute event-by-event weights
    if args['differential']:
        
        print(__name__ + f".compute_ND_reweights: Reference class: <{args['reference_class']}> (Found {num_classes} classes: {np.unique(y)} from y)")
        
        ### Collect re-weighting variables
        RV = {}
        for var in paramdict.keys():
            if use_ak:
                RV[var] = ak.to_numpy(x[var]).astype(np.float)
            else:
                RV[var] = x[:, ids.index(paramdict[var])].astype(np.float)

        ### Pre-transform
        for var in paramdict.keys():
            mode = args[f'transform'][int(var)]
            
            if args[f'binmode'][int(var)] == 'edges':
                raise Exception(__name__ + '.compute_ND_reweights: Cannot transform "edges" type')
            
            d = args[f'bins'][int(var)]
            
            if   mode == 'log10':
                if np.any(RV[var] <= 0):
                    ind = (RV[var] <= 0)
                    cprint(__name__ + f'.compute_ND_reweights: Variable {var} < 0 (in {np.sum(ind)} elements) in log10 -- truncating to zero', 'red')

                # Transform values
                RV[var] = np.log10(np.maximum(RV[var], EPS))

                # Transform bins
                args[f'bins'][int(var)][0] = np.log10(d[0] + EPS)
                args[f'bins'][int(var)][1] = np.log10(d[1])

            elif mode == 'sqrt':

                # Values
                RV[var] = np.sqrt(np.maximum(RV[var], EPS))

                # Bins
                args[f'bins'][int(var)][0] = np.sqrt(d[0] + EPS)
                args[f'bins'][int(var)][1] = np.sqrt(d[1])
                
            elif mode == 'square':

                # Values
                RV[var] = RV[var]**2

                # Bins
                args[f'bins'][int(var)][0] = d[0]**2
                args[f'bins'][int(var)][1] = d[1]**2

            elif mode == None:
                True
            else:
                raise Exception(__name__ + '.compute_ND_reweights: Unknown pre-transform')

        # Binning setup
        binedges = {}
        for var in paramdict.keys():
            
            d = args[f'bins'][int(var)]
            
            if   args[f'binmode'][int(var)] == 'linear':
                binedges[var] = np.linspace(d[0], d[1], d[2])

            elif args[f'binmode'][int(var)] == 'log10':
                binedges[var] = np.logspace(np.log10(np.maximum(d[0], EPS)), np.log10(d[1]), d[2], base=10)
                
            elif args[f'binmode'][int(var)] == 'edges':
                binedges[var] = np.array(d)
            else:
                raise Exception(__name__ + ': Unknown re-weight binning mode')
        

        rwparam = {
            'y':               y,
            'reference_class': args['reference_class'],
            'max_reg':         args['max_reg']
        }

        ### Compute 2D-PDFs for each class
        if args['dimension'] == '2D':
            
            print(__name__ + f'.compute_ND_reweights: 2D re-weighting using variables: {paramdict} ...')
            
            if pdf is None: # Not given by user
                pdf = {}
                for c in range(num_classes):

                    sample_weights = w[y==c] if w is not None else None # Feed in the input weights

                    pdf[c] = pdf_2D_hist(X_A=RV['0'][y==c], X_B=RV['1'][y==c], w=sample_weights, \
                        binedges_A=binedges['0'], binedges_B=binedges['1'])

                pdf['binedges_0']  = binedges['0']
                pdf['binedges_1']  = binedges['1']
                pdf['num_classes'] = num_classes
            
            weights_doublet = reweightcoeff2D(X_A = RV['0'], X_B = RV['1'], pdf=pdf, **rwparam)
            
        ### Compute geometric mean factorized 1D x 1D product
        elif args['dimension'] == 'pseudo-2D':
            
            print(__name__ + f'.compute_ND_reweights: pseudo-2D re-weighting using variables: {paramdict} ...')
            
            #if len(binedges['0']) != len(binedges['1']):
            #    raise Exception(__name__ + f'.compute_ND_reweights: Error: <pseudo-2D> requires same number of bins for both variables')

            if pdf is None: # Not given by user
                pdf = {}
                for c in range(num_classes):

                    sample_weights = w[y==c] if w is not None else None # Feed in the input weights

                    pdf_0  = pdf_1D_hist(X=RV['0'][y==c], w=sample_weights, binedges=binedges['0'])
                    pdf_1  = pdf_1D_hist(X=RV['1'][y==c], w=sample_weights, binedges=binedges['1'])
                    
                    if   args['pseudo_type'] == 'geometric_mean':
                        pdf[c] = np.sqrt(np.outer(pdf_0, pdf_1)) # (A,B) order gives normal matrix indexing
                    elif args['pseudo_type'] == 'product':
                        pdf[c] = np.outer(pdf_0, pdf_1)
                    else:
                        raise Exception(__name__ + f'.compute_ND_reweights: Unknown <pseudo_type>')

                    # Normalize to discrete density
                    pdf[c] /= np.sum(pdf[c].flatten())

                pdf['binedges_0']  = binedges['0']
                pdf['binedges_1']  = binedges['1']
                pdf['num_classes'] = num_classes

            weights_doublet = reweightcoeff2D(X_A = RV['0'], X_B = RV['1'], pdf=pdf, **rwparam)

        ### Compute 1D-PDFs for each class
        elif args['dimension'] == '1D':
            
            print(__name__ + f'.compute_ND_reweights: 1D re-weighting using variable {paramdict[list(paramdict.keys())[0]]} ...')
            
            if pdf is None: # Not given by user
                pdf = {}
                for c in range(num_classes):
                    sample_weights = w[y==c] if w is not None else None # Feed in the input weights
                    pdf[c] = pdf_1D_hist(X=RV['0'][y==c], w=sample_weights, binedges=binedges['0'])

                pdf['binedges']    = binedges['0']
                pdf['num_classes'] = num_classes

            weights_doublet = reweightcoeff1D(X = RV['0'], pdf=pdf, **rwparam)
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


def reweightcoeff1D(X, y, pdf, reference_class, max_reg = 1e3, EPS=1e-12):
    """ Compute N-class density reweighting coefficients.
    
    Args:
        X:               Observable of interest (N x 1)
        y:               Class labels (0,1,...) (N x 1)
        pdf:             PDF for each class
        reference_class: e.g. 0 (background) or 1 (signal)
    
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
        X_A :             First observable of interest  (N x 1)
        X_B :             Second observable of interest (N x 1)
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
        inds_0 = aux.x2ind(X_A[y == c], pdf['binedges_0']) # variable 0
        inds_1 = aux.x2ind(X_B[y == c], pdf['binedges_1']) # variable 1
        if c is not reference_class:
            weights_doublet[y == c, c] = pdf[reference_class][inds_0, inds_1] / (pdf[c][inds_0, inds_1] + EPS)
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

