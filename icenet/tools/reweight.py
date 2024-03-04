# Event sample re-weighting tools
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import awkward as ak

import matplotlib.pyplot as plt
from termcolor import colored, cprint
import copy

from icenet.tools import aux


def compute_ND_reweights(x, y, w, ids, args, pdf=None, EPS=1e-12):
    """
    Compute N-dim reweighting coefficients (currently 2D or 1D supported)
    
    Args:
        x    : training sample input
        y    : training sample (class) labels
        w    : training sample weights
        ids  : variable names of columns of x
        pdf  : pre-computed pdfs
        args : reweighting parameters in a dictionary
    
    Returns:
        weights : 1D-array of re-weights
    """
    use_ak = True if isinstance(x, ak.Array) else False
    
    if use_ak:
        y = copy.deepcopy(ak.to_numpy(y).astype(int))
        w = copy.deepcopy(ak.to_numpy(w).astype(float))
    
    class_ids = np.unique(y.astype(int))
    
    args = copy.deepcopy(args) # Make sure we make a copy, because we modify args here

    # Compute event-by-event weights
    if args['differential']:
        
        ### Construct parameter names
        paramdict = {}
        for i in range(len(args['var'])):
            try:
                varname  = args['var'][i]
                paramdict[str(i)] = varname
            except:
                break    
        
        print(__name__ + f".compute_ND_reweights: Reference class: <{args['reference_class']}> | Found classes: {np.unique(y)} from y")
        
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
                args[f'bins'][int(var)][0] = np.log10(max(d[0], EPS))
                args[f'bins'][int(var)][1] = np.log10(d[1])

            elif mode == 'sqrt':

                # Values
                RV[var] = np.sqrt(np.maximum(RV[var], EPS))

                # Bins
                args[f'bins'][int(var)][0] = np.sqrt(max(d[0], EPS))
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
                binedges[var] = np.logspace(np.log10(max(d[0], EPS)), np.log10(d[1]), d[2], base=10)
                
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
                for c in class_ids:
                    
                    sample_weights = w[y==c] if w is not None else None # Feed in the input weights

                    pdf[c] = pdf_2D_hist(X_A=RV['0'][y==c], X_B=RV['1'][y==c], w=sample_weights, \
                        binedges_A=binedges['0'], binedges_B=binedges['1'])

                pdf['binedges_0'] = binedges['0']
                pdf['binedges_1'] = binedges['1']
                pdf['class_ids']  = class_ids
            
            weights_doublet = reweightcoeff2D(X_A = RV['0'], X_B = RV['1'], pdf=pdf, **rwparam)
            
        ### Compute geometric mean factorized 1D x 1D product
        elif args['dimension'] == 'pseudo-2D':
            
            print(__name__ + f'.compute_ND_reweights: pseudo-2D re-weighting using variables: {paramdict} ...')
            
            #if len(binedges['0']) != len(binedges['1']):
            #    raise Exception(__name__ + f'.compute_ND_reweights: Error: <pseudo-2D> requires same number of bins for both variables')

            if pdf is None: # Not given by user
                pdf = {}
                for c in class_ids:
                    
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
                pdf['class_ids']   = class_ids

            weights_doublet = reweightcoeff2D(X_A = RV['0'], X_B = RV['1'], pdf=pdf, **rwparam)

        ### Compute 1D-PDFs for each class
        elif args['dimension'] == '1D':
            
            print(__name__ + f'.compute_ND_reweights: 1D re-weighting using variable {paramdict[list(paramdict.keys())[0]]} ...')
            
            if pdf is None: # Not given by user
                pdf = {}
                for c in class_ids:
                    sample_weights = w[y==c] if w is not None else None # Feed in the input weights
                    pdf[c] = pdf_1D_hist(X=RV['0'][y==c], w=sample_weights, binedges=binedges['0'])

                pdf['binedges']  = binedges['0']
                pdf['class_ids'] = class_ids
            
            weights_doublet = reweightcoeff1D(X = RV['0'], pdf=pdf, **rwparam)
        else:
            raise Exception(__name__ + f'.compute_ND_reweights: Unsupported dimensionality mode <{args["dimension"]}>')
    
    # No differential re-weighting    
    else:
        print(__name__ + f".compute_ND_reweights: Reference class: <{args['reference_class']}> | Found classes {class_ids} from y")
        weights_doublet = {}
        
        for c in class_ids:
            weights_doublet[c] = np.zeros(len(x))
            sample_weights = w[y==c] if w is not None else 1.0 # Feed in the input weights
            weights_doublet[c][y == c] = sample_weights

    ### Apply class balance equalizing weight
    if (args['equal_frac'] == True):
        cprint(__name__ + f'.Compute_ND_reweights: Equalizing class fractions', 'green')
        weights_doublet = balanceweights(weights_doublet=weights_doublet, reference_class=args['reference_class'], y=y)

    ### Finally map back to 1D-array
    weights = np.zeros(len(w))
    for c in class_ids:
        weights = weights + weights_doublet[c]

    ### Compute the sum of weights per class for the output print
    frac = {}
    sums = {}
    for c in class_ids:
        frac[c], sums[c] = np.sum(y == c), np.round(np.sum(weights[y == c]), 2)
    
    print(__name__ + f'.compute_ND_reweights: Output sum[y == c] = {frac} || sum[weights[y == c]] = {sums} ({class_ids})')
    
    if use_ak:
        return ak.Array(weights), pdf
    else:
        return weights, pdf


def reweightcoeff1D(X, y, pdf, reference_class, max_reg = 1e3, EPS=1e-12):
    """ Compute N-class density reweighting coefficients.
    
    Args:
        X:               Observable of interest (N x 1)
        y:               Class labels (N x 1)
        pdf:             PDF for each class
        reference_class: e.g. 0 (background) or 1 (signal)
    
    Returns:
        weights for each event
    """
    class_ids = pdf['class_ids']

    # Re-weighting weights
    weights_doublet = {} # Init with zeros!!
    for c in class_ids:
        weights_doublet[c] = np.zeros(X.shape[0])
    
    # Weight each class against the reference class
    for c in class_ids:
        inds = aux.x2ind(X[y == c], pdf['binedges'])
        if c is not reference_class:
            weights_doublet[c][y == c] = pdf[reference_class][inds] / np.clip(pdf[c][inds], a_min=EPS, a_max=None)
        else:
            weights_doublet[c][y == c] = 1.0 # Reference class stays intact

        # Maximum weight cut-off regularization
        weights_doublet[c][weights_doublet[c] > max_reg] = max_reg

    return weights_doublet


def reweightcoeff2D(X_A, X_B, y, pdf, reference_class, max_reg = 1e3, EPS=1E-12):
    """ Compute N-class density reweighting coefficients.
    
    Operates in full 2D without any factorization.
    
    Args:
        X_A :             First observable of interest  (N x 1)
        X_B :             Second observable of interest (N x 1)
        y   :             Class labels (N x 1)
        pdf :             Density histograms for each class
        reference_class : e.g. Background (0) or signal (1)
        max_reg :         Regularize the maximum reweight coefficient
    
    Returns:
        weights for each event
    """
    class_ids = pdf['class_ids']

    # Re-weighting weights
    weights_doublet = {} # Init with zeros!!
    for c in class_ids:
        weights_doublet[c] = np.zeros(X_A.shape[0])
    
    # Weight each class against the reference class
    for c in class_ids:
        inds_0 = aux.x2ind(X_A[y == c], pdf['binedges_0']) # variable 0
        inds_1 = aux.x2ind(X_B[y == c], pdf['binedges_1']) # variable 1
        if c is not reference_class:
            weights_doublet[c][y == c] = pdf[reference_class][inds_0, inds_1] / np.clip(pdf[c][inds_0, inds_1], a_min=EPS, a_max=None)
        else:
            weights_doublet[c][y == c] = 1.0 # Reference class stays intact

        # Maximum weight cut-off regularization
        weights_doublet[c][weights_doublet[c] > max_reg] = max_reg

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
        weights_doublet:  N-class event weights (events x classes)
        reference_class:  which class gives the reference (integer)
        y:                class targets
    Returns:
        weights doublet with new weights per event
    """
    class_ids = np.unique(y).astype(int)
    ref_sum   = np.sum(weights_doublet[reference_class][y == reference_class])
    
    for c in class_ids:
        if c is not reference_class:
            EQ = ref_sum / np.clip(np.sum(weights_doublet[c][y == c]), a_min=EPS, a_max=None)
            weights_doublet[c][y == c] *= EQ
    
    return weights_doublet

