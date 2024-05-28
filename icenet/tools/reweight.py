# Event sample re-weighting tools
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import awkward as ak
import torch
import matplotlib.pyplot as plt
from termcolor import colored, cprint
import copy
from prettytable import PrettyTable

from icenet.tools import aux
from icenet.tools import io
from icenet.deep  import iceboost


def rw_transform_with_logits(logits, mode, absMax=30):
    """
    AI/Deep reweighting transform with logits input
    
    Args:
        prob: probabilities
        mode: operation mode
    
    Returns:
        transformed values
    """
    
    if type(logits) is torch.Tensor:
        exp_     = torch.exp
        sigmoid_ = torch.sigmoid
        logits   = torch.clip(logits, -absMax, absMax) # Protect to not overflow exp
    else:
        exp_     = np.exp
        sigmoid_ = aux.sigmoid
        logits   = np.clip(logits, -absMax, absMax)
    
    if   mode  == 'LR':                # LR trick
        return exp_(logits)
    elif mode == 'inverse-LR':         # Inverse LR trick
        return exp_(-logits)
    elif mode == 'DeepEfficiency':     # https://arxiv.org/abs/1809.06101
        return exp_(-logits) + 1.0
    elif mode == 'direct':             # Direct
        return sigmoid_(logits)
    elif mode == 'identity' or mode == None:
        if type(logits) is torch.Tensor:
            return torch.ones_like(logits)
        else:
            return np.ones_like(logits)
    else:
        raise Exception(__name__ + f'.reweight_transform_with_logits: Unknown operation mode {mode}')


def rw_transform(phat, mode, EPS=1E-12):
    """
    AI/Deep reweighting transform
    
    Args:
        phat: estimated probabilities
        mode: operation mode
    
    Returns:
        transformed values
    """
    
    if type(phat) is torch.Tensor:
        phat = torch.clip(phat, EPS, 1-EPS)
    else:
        phat = np.clip(phat, EPS, 1-EPS)
    
    if   mode  == 'LR':                   # LR trick
        return phat / (1.0 - phat)
    elif mode == 'inverse-LR':            # Inverse LR trick
        return (1.0 - phat) / phat
    elif mode == 'DeepEfficiency':        # https://arxiv.org/abs/1809.06101
        return 1.0 / phat
    elif mode == 'direct':                # Direct
        return phat
    elif mode == 'identity' or mode == None:
        if type(phat) is torch.Tensor:
            return torch.ones_like(phat)
        else:
            return np.ones_like(phat)
    else:
        raise Exception(__name__ + f'.reweight_transform: Unknown operation mode {mode}')


def histogram_helper(x, y, w, ids, pdf, args, EPS):
    """
    Helper function for histogram based reweighting
    """
    
    class_ids       = np.unique(y.astype(int))
    reference_class = args['reweight_param']['reference_class']
    
    diff_args = args['reweight_param']['diff_param']
    h_args    = args['reweight_param']['diff_param']['hist_param']
    variables = diff_args['var']
    
    print(__name__ + f".compute_ND_reweights: Reference (target) class: <{reference_class}> | Found classes: {class_ids} from y")
    
    ### Collect re-weighting variables
    RV = {}
    for i, var in enumerate(variables):
        if isinstance(x, ak.Array):
            RV[i] = ak.to_numpy(x[var]).astype(np.float32)
        else:
            RV[i] = x[:, ids.index(var)].astype(np.float32)
    
    ### Pre-transform
    for i, var in enumerate(variables):
        
        mode = h_args[f'transform'][i]
        
        if h_args[f'binmode'][i] == 'edges':
            raise Exception(__name__ + '.compute_ND_reweights: Cannot transform "edges" type')
        
        d = h_args[f'bins'][i]
        
        if   mode == 'log10':
            if np.any(RV[i] <= 0):
                ind = (RV[i] <= 0)
                cprint(__name__ + f'.compute_ND_reweights: Variable {var} < 0 (in {np.sum(ind)} elements) in log10 -- truncating to zero', 'red')

            # Transform values
            RV[i] = np.log10(np.maximum(RV[i], EPS))

            # Transform bins
            h_args[f'bins'][i][0] = np.log10(max(d[0], EPS))
            h_args[f'bins'][i][1] = np.log10(d[1])

        elif mode == 'sqrt':
            
            # Values
            RV[i] = np.sqrt(np.maximum(RV[i], EPS))

            # Bins
            h_args[f'bins'][i][0] = np.sqrt(max(d[0], EPS))
            h_args[f'bins'][i][1] = np.sqrt(d[1])
            
        elif mode == 'square':
            
            # Values
            RV[i] = RV[i]**2

            # Bins
            h_args[f'bins'][i][0] = d[0]**2
            h_args[f'bins'][i][1] = d[1]**2

        elif mode == None:
            True
        else:
            raise Exception(__name__ + '.compute_ND_reweights: Unknown pre-transform')

    # Binning setup
    binedges = {}
    for i, var in enumerate(variables):
        
        d = h_args[f'bins'][i]
        
        if   h_args[f'binmode'][i] == 'linear':
            binedges[i] = np.linspace(d[0], d[1], d[2])

        elif h_args[f'binmode'][i] == 'log10':
            binedges[i] = np.logspace(np.log10(max(d[0], EPS)), np.log10(d[1]), d[2], base=10)
            
        elif h_args[f'binmode'][i] == 'edges':
            binedges[i] = np.array(d)
            
        else:
            raise Exception(__name__ + ': Unknown re-weight binning mode')
    
    rwparam = {
        'y':               y,
        'reference_class': reference_class,
        'max_reg':         h_args['max_reg']
    }

    ### Compute 2D-PDFs for each class
    if diff_args['type'] == '2D':
        
        print(__name__ + f'.compute_ND_reweights: 2D re-weighting with: [{variables[0]}, {variables[1]}] ...')
        
        if pdf is None: # Not given by user
            pdf = {}
            for c in class_ids:
                
                ind = (y == c)
                sample_weights = w[ind] if w is not None else None # Feed in the input weights !

                pdf[c] = pdf_2D_hist(X_A=RV[0][ind], X_B=RV[1][ind], w=sample_weights, \
                    binedges_A=binedges[0], binedges_B=binedges[1])

            pdf['binedges']  = binedges
            pdf['class_ids'] = class_ids
        
        weights_doublet = reweightcoeff2D(X_A = RV[0], X_B = RV[1], pdf=pdf, **rwparam)
        
    ### Compute factorized 1D x 1D product
    elif diff_args['type'] == 'pseudo-ND':
        
        print(__name__ + f'.compute_ND_reweights: pseudo-ND (2D for now) re-weighting with: [{variables[0]}, {variables[1]}] ...')
        
        if pdf is None: # Not given by user
            pdf = {}
            for c in class_ids:
                
                ind = (y == c)
                sample_weights = w[ind] if w is not None else None # Feed in the input weights !
                
                pdf_0 = pdf_1D_hist(X=RV[0][ind], w=sample_weights, binedges=binedges[0])
                pdf_1 = pdf_1D_hist(X=RV[1][ind], w=sample_weights, binedges=binedges[1])
                
                if   h_args['pseudo_type'] == 'geometric_mean':
                    pdf[c] = np.sqrt(np.outer(pdf_0, pdf_1)) # (A,B) order gives normal matrix indexing
                elif h_args['pseudo_type'] == 'product':
                    pdf[c] = np.outer(pdf_0, pdf_1)
                else:
                    raise Exception(__name__ + f'.compute_ND_reweights: Unknown <pseudo_type>')

                # Normalize to discrete density
                pdf[c] /= np.sum(pdf[c].flatten())
            
            pdf['binedges']  = binedges
            pdf['class_ids'] = class_ids
        
        weights_doublet = reweightcoeff2D(X_A = RV[0], X_B = RV[1], pdf=pdf, **rwparam)

    ### Compute 1D-PDFs for each class
    elif diff_args['type'] == '1D':
        
        print(__name__ + f'.compute_ND_reweights: 1D re-weighting with: {variables[0]} ...')
        
        if pdf is None: # Not given by user
            pdf = {}
            for c in class_ids:
                ind = (y == c)
                sample_weights = w[ind] if w is not None else None # Feed in the input weights !
                pdf[c] = pdf_1D_hist(X=RV[0][ind], w=sample_weights, binedges=binedges[0])

            pdf['binedges']  = binedges[0]
            pdf['class_ids'] = class_ids
        
        weights_doublet = reweightcoeff1D(X = RV[0], pdf=pdf, **rwparam)
    else:
        raise Exception(__name__ + f".compute_ND_reweights: Unsupported dimensionality mode <{diff_args['type']}>")

    pdf['vars'] = variables
    
    return weights_doublet, pdf


def map_xyw(x, y, w, vars, c, reference_class):
    """
    For AIRW
    """
    
    # Source and Target collected
    ind = (y == c) | (y == reference_class)
    
    new_data = io.IceXYW(x=x[ind], y=y[ind], w=w[ind], ids=vars)
    
    # ----------------------------
    # Change labels to (0,1)
    y_new = np.zeros(len(new_data.y)).astype(np.int32)
    y_new[new_data.y == c] = 0
    y_new[new_data.y == reference_class] = 1                
    
    new_data.y = y_new # !
    # ----------------------------
    
    # Equalize class balance for the training
    for k in [0,1]:
        ind = (y_new == k)
        new_data.w[ind] /= np.sum(new_data.w[ind])

    return new_data


def AIRW_helper(x, y, w, ids, pdf, args, x_val, y_val, w_val, EPS=1e-12):
    """
    Helper function for ML based reweighting
    """
    
    class_ids       = np.unique(y.astype(int))
    reference_class = args['reweight_param']['reference_class']
    
    ID        = args['reweight_param']['diff_param']['ML_param']['active_model']
    param     = args['models'][ID]
    diff_args = args['reweight_param']['diff_param']
    variables = diff_args['var']
    
    # ----------------------------------------
    # Conversions and pick variables of interest
    
    if isinstance(x, ak.Array):
        x     = ak.to_numpy(x[variables]).astype(np.float32)
        if x_val is not None:
            x_val = ak.to_numpy(x_val[variables]).astype(np.float32)
    else:
        x = x[:, io.index_list(ids, variables)].astype(np.float32)
        if x_val is not None:
            x_val = x_val[:, io.index_list(ids, variables)].astype(np.float32)
    # ----------------------------------------
    
    # Train model per class pair
    if pdf is None:
        
        pdf = {'ID': ID, 'param': param, 'model': {}, 'vars': variables}
        
        for c in class_ids:
            if c != reference_class:
                
                data_trn = map_xyw(x=x, y=y, w=w, vars=variables, c=c, reference_class=reference_class) 
                data_val = map_xyw(x=x_val, y=y_val, w=w_val, vars=variables, c=c, reference_class=reference_class)
                
                inputs = {'data_trn':    data_trn,
                          'data_val':    data_val,
                          'args':        args,
                          'data_trn_MI': None,
                          'data_val_MI': None,
                          'param':       param}

                pdf['model'][c] = iceboost.train_xgb(**inputs)
    
    # ------------------------------------------------
    ## Now apply the model
    from icenet.deep import predict
    
    param = pdf['param']
    wnew  = copy.deepcopy(w)
    
    for c in pdf['model'].keys():
        
        print(__name__ + f'.AIRW_helper: Applying AIRW reweighter to class {c}')
        
        if   param['predict'] == 'xgb':
            func_predict = predict.pred_xgb(args=args, param=param, feature_names=variables)
        elif param['predict'] == 'xgb_logistic':
            func_predict = predict.pred_xgb_logistic(args=args, param=param, feature_names=variables)
        else:
            raise Exception('Unsupported model -- "predict" should be "xgb" or "xgb_logistic"')
        
        # -----------------------------------------------
        # Predict for events of this class
        ind = (y == c)
        p   = func_predict(x[ind])

        # Turn into a likelihood ratio to obtain the reweight factor
        p   = np.clip(p, a_min = EPS, a_max = 1 - EPS)
        LR  = p / (1 - p)
        
        # Apply maximum weight regularization
        LR = np.clip(LR, a_min=0.0, a_max=diff_args['ML_param']['max_reg'])
        
        # Apply to weights
        wnew[ind] = wnew[ind] * LR if w is not None else np.ones_like(y) * LR
        # -----------------------------------------------
    
    # Transform weights into weights doublet
    weights_doublet = {}
    for c in class_ids:
        ind = (y == c)
        weights_doublet[c]      = np.zeros(x.shape[0])
        weights_doublet[c][ind] = wnew[ind]
    
    return weights_doublet, pdf

    
def compute_ND_reweights(x, y, w, ids, args, pdf=None, EPS=1e-12, x_val=None, y_val=None, w_val=None):
    """
    Compute N-dim reweighting coefficients
    
    Supports 'ML' (ND), 'pseudo-ND' (1D x 1D ... x 1D), '2D', '1D'
    
    For 'args' dictionary structure, see steering cards.
    
    Args:
        x    : training sample input
        y    : training sample (class) labels
        w    : training sample weights
        ids  : variable names of columns of x
        pdf  : pre-computed pdfs (default None)
        args : reweighting parameters in a dictionary
       
    Returns:
        weights : 1D-array of re-weights
        pdf     : computed pdfs
    """
    
    use_ak = True if isinstance(x, ak.Array) else False
    
    if use_ak:
        y = copy.deepcopy(ak.to_numpy(y).astype(int))
        w = copy.deepcopy(ak.to_numpy(w).astype(float))
    
    class_ids = np.unique(y.astype(int))
    
    # Make sure we make a copy, because we modify args here
    args = copy.deepcopy(args)
    
    ## Differential reweighting
    if args['reweight_param']['differential']:
        
        print(__name__ + f'.compute_ND_reweights: Differential reweighting')
        
        # Histogram based
        if args['reweight_param']['diff_param']['type'] != 'ML':
            
            weights_doublet, pdf = histogram_helper(x=x, y=y, w=w, ids=ids,
                                                    pdf=pdf, args=args, EPS=EPS)      

        # ML-model based
        else:
            weights_doublet, pdf = AIRW_helper(x=x, y=y, w=w, ids=ids, pdf=pdf, args=args,
                                    x_val=x_val, y_val=y_val, w_val=w_val, EPS=EPS)
        
        # Renormalize integral (sum) to the event counts per class
        for c in class_ids:
            ind = (y == c)
            weights_doublet[c][ind] /= np.sum(weights_doublet[c][ind])
            weights_doublet[c][ind] *= np.sum(ind)
    
    # No differential re-weighting    
    else:
        print(__name__ + f'.compute_ND_reweights: No differential reweighting')
        print(__name__ + f".compute_ND_reweights: Reference (target) class: <{args['reweight_param']['reference_class']}> | Found classes {class_ids} from y")
        weights_doublet = {}
        
        for c in class_ids:
            weights_doublet[c] = np.zeros(len(x))                   # init with zeros
            sample_weights     = w[y==c] if w is not None else 1.0  # Feed in the input weights
            weights_doublet[c][y == c] = sample_weights

    # -------------------------------------------------
    
    ### Apply class balance equalizing weight
    if (args['reweight_param']['equal_frac'] == True):
        cprint(__name__ + f'.Compute_ND_reweights: Equalizing class fractions', 'green')
        weights_doublet = balanceweights(weights_doublet=weights_doublet,
                                         reference_class=args['reweight_param']['reference_class'], y=y)

    ### Finally map back to 1D-array
    weights = np.zeros(len(w))
    for c in class_ids:
        weights = weights + weights_doublet[c]
    
    ### Compute diagnostics
    table = PrettyTable(["class", "events", "sum(w)", "mean(w)", "std(w)", "min(w)", "Q5(w)", "Q95(w)", "max(w)"]) 
    
    for c in class_ids:
        ind = (y == c)
        table.add_row([f'{c}',
                       f'{np.sum(ind)}',
                       f'{np.round(np.sum(weights[ind]), 2)}',
                       f'{np.mean(weights[ind]):0.3E}',
                       f'{np.std(weights[ind]):0.3E}',
                       f'{np.min(weights[ind]):0.3E}',
                       f'{np.percentile(weights[ind],  5):0.3E}',
                       f'{np.percentile(weights[ind], 95):0.3E}',
                       f'{np.max(weights[ind]):0.3E}'])
    
    print(table)
    print('')
    
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
        inds_0 = aux.x2ind(X_A[y == c], pdf['binedges'][0]) # variable 0
        inds_1 = aux.x2ind(X_B[y == c], pdf['binedges'][1]) # variable 1
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

