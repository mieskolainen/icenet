# Event sample re-weighting tools
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import awkward as ak
import torch
import matplotlib.pyplot as plt
import copy

from icenet.tools import aux, io, prints
from icenet.deep  import iceboost, predict

# ------------------------------------------
from icenet import print
# ------------------------------------------


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
    
    print(f'Reweight transform with mode: {mode}')
    
    # Simplified functions based on ratios of sigmoids
    
    if   mode == 'LR':                 # LR trick
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
    
    print(f'Reweight transform with mode: {mode}')
    
    if   mode == 'LR':                    # LR trick
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
    
    print(f"Reference (target) class [{reference_class}] | Found classes: {class_ids} from y")
    
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
            raise Exception(__name__ + '.histogram_helper: Cannot transform "edges" type')
        
        d = h_args[f'bins'][i]
        
        if   mode == 'log10':
            if np.any(RV[i] <= 0):
                ind = (RV[i] <= 0)
                print(f'Variable {var} < 0 (in {np.sum(ind)} elements) in log10 -- truncating to zero', 'red')

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
            raise Exception(__name__ + '.histogram_helper: Unknown pre-transform')

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
            raise Exception(__name__ + ': Unknown reweight binning mode')
    
    rwparam = {
        'y':               y,
        'reference_class': reference_class,
        'max_reg':         h_args['max_reg']
    }

    ### Compute 2D-PDFs for each class
    if diff_args['type'] == '2D':
        
        print(f'2D re-weighting with: [{variables[0]}, {variables[1]}] ...')
        
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
        
        print(f'pseudo-ND (2D for now) reweighting with: [{variables[0]}, {variables[1]}] ...')
        
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
                    raise Exception(__name__ + f'.histogram_helper: Unknown "pseudo_type"')

                # Normalize to discrete density
                pdf[c] /= np.sum(pdf[c].flatten())
            
            pdf['binedges']  = binedges
            pdf['class_ids'] = class_ids
        
        weights_doublet = reweightcoeff2D(X_A = RV[0], X_B = RV[1], pdf=pdf, **rwparam)

    ### Compute 1D-PDFs for each class
    elif diff_args['type'] == '1D':
        
        print(f'1D reweighting with: {variables[0]} ...')
        
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
        raise Exception(__name__ + f'.histogram_helper: Unsupported dimensionality mode "{diff_args["type"]}"')

    pdf['vars'] = variables
    
    return weights_doublet, pdf


def map_xyw(x, y, w, vars, c, reference_class):
    """
    For AIRW helper
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
    
    diff_args = args['reweight_param']['diff_param']
    variables = diff_args['var']
    ID        = diff_args['AIRW_param']['active_model']
    RW_mode   = diff_args['AIRW_param']['mode']
    MAX_REG   = diff_args['AIRW_param']['max_reg']
    
    param     = args['models'][ID]
    
    
    # ----------------------------------------
    # Conversions and pick variables of interest
    
    if isinstance(x, ak.Array):
        x     = aux.ak2numpy(x=x, fields=variables).astype(np.float32)
        if x_val is not None:
            x_val = aux.ak2numpy(x=x_val, fields=variables).astype(np.float32)
    else:
        x = x[:, io.index_list(ids, variables)].astype(np.float32)
        if x_val is not None:
            x_val = x_val[:, io.index_list(ids, variables)].astype(np.float32)
    # ----------------------------------------
    
    print(f'Training N-dim reweighting', 'magenta')
    print(f'x.shape = {x.shape}', 'magenta')
    print(f'variables = {variables}', 'magenta')
    
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
    
    param = pdf['param']
    
    if w is not None:
        wnew = copy.deepcopy(w)
    else:
        wnew = np.ones(len(y))
    
    for c in pdf['model'].keys():
        
        print(f'Applying AIRW reweighter to class [{c}]')
        
        if   param['predict'] == 'xgb':
            func_predict = predict.pred_xgb(args=args, param=param, feature_names=variables)
        elif param['predict'] == 'xgb_logistic':
            func_predict = predict.pred_xgb_logistic(args=args, param=param, feature_names=variables)
        else:
            raise Exception(__name__ + f'.AIRW_helper: Unsupported model -- "predict" field should be "xgb" or "xgb_logistic"')
        
        # -----------------------------------------------
        # Predict for events of this class
        ind  = (y == c)
        pred = func_predict(x[ind])

        # Handle logits vs probabilities
        min_pred, max_pred = np.min(pred), np.max(pred)
        
        THRESH = 1E-5
        if min_pred < (-THRESH) or max_pred > (1.0 + THRESH):
            print(f'Detected raw logit output [{min_pred:0.4f}, {max_pred:0.4f}] from the model')
            logits = pred
            p      = aux.sigmoid(logits)
            print(f'Corresponding probability output [{np.min(p):0.4f}, {np.max(p):0.4f}]')
        else:
            print(f'Detected probability output [{min_pred:0.4f}, {max_pred:0.4f}] from the model')
            logits = aux.inverse_sigmoid(pred)
            print(f'Corresponding logit output [{np.min(logits):0.4f}, {np.max(logits):0.4f}]')
        
        # Get weights after the re-weighting transform
        AI_w = rw_transform_with_logits(logits=logits, mode=RW_mode)
        
        # Apply maximum weight regularization
        AI_w = np.clip(AI_w, 0.0, MAX_REG)
        
        # Apply multiplicatively to event weights
        wnew[ind] = wnew[ind] * AI_w
        # -----------------------------------------------
    
    # Transform weights into weights doublet
    weights_doublet = {}
    for c in class_ids:
        ind = (y == c)
        weights_doublet[c]      = np.zeros(x.shape[0])
        weights_doublet[c][ind] = wnew[ind]
    
    return weights_doublet, pdf


def doublet_helper(x, y, w, class_ids):

    weights_doublet = {}
    
    for c in class_ids:
        weights_doublet[c] = np.zeros(len(x))                   # init with zeros
        sample_weights     = w[y==c] if w is not None else 1.0  # Feed in the input weights
        weights_doublet[c][y == c] = sample_weights

    return weights_doublet    


def compute_ND_reweights(x, y, w, ids, args, pdf=None, EPS=1e-12,
                         x_val=None, y_val=None, w_val=None, skip_reweights=False):
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
        
        print(f'Differential reweighting')
        
        # Histogram based
        if args['reweight_param']['diff_param']['type'] != 'AIRW':
            
            weights_doublet, pdf = histogram_helper(x=x, y=y, w=w, ids=ids,
                                                    pdf=pdf, args=args, EPS=EPS)      

        # AIRW based
        else:
            weights_doublet, pdf = AIRW_helper(x=x, y=y, w=w, ids=ids, pdf=pdf, args=args,
                                    x_val=x_val, y_val=y_val, w_val=w_val, EPS=EPS)
        
        # Renormalize integral (sum) to the event counts per class
        if args['reweight_param']['diff_param']['renorm_weight_to_count']:
            print(f'Renormalizing sum(weights) == sum(count) per class')    
            for c in class_ids:
                ind = (y == c)
                weights_doublet[c][ind] /= np.sum(weights_doublet[c][ind])
                weights_doublet[c][ind] *= np.sum(ind)

        # ---------------------------------------------------------------
        # Special mode -- allows to only train pdf but then do not apply
        if skip_reweights:
            print(f'Special mode [skip_reweights] active: differential reweight model not applied', 'red')    
            weights_doublet = doublet_helper(x=x, y=y, w=w, class_ids=class_ids)
        
        # ---------------------------------------------------------------
        
    # No differential re-weighting    
    else:
        print(f"No differential reweighting")
        print(f"Reference [target] class: [{args['reweight_param']['reference_class']}] | Found classes {class_ids} from y")

        weights_doublet = doublet_helper(x=x, y=y, w=w, class_ids=class_ids)
    
    # --------------------------------------------------------
    
    ### Apply class balance equalizing weight
    if (args['reweight_param']['equal_frac'] == True):
        print(f"Equalizing class fractions", "green")
        weights_doublet = balanceweights(weights_doublet=weights_doublet,
                                         reference_class=args['reweight_param']['reference_class'], y=y)

    ### Finally map back to 1D-array
    weights = np.zeros(len(w))
    for c in class_ids:
        weights = weights + weights_doublet[c]
    
    ### Print weights
    prints.print_weights(weights=weights, y=y)
    
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
            weights_doublet[c][y == c] = pdf[reference_class][inds] / np.clip(pdf[c][inds], EPS, None)
        else:
            weights_doublet[c][y == c] = 1.0 # Reference class stays intact

        # Maximum weight cut-off regularization
        weights_doublet[c][weights_doublet[c] > max_reg] = max_reg

    return weights_doublet


def reweightcoeff2D(X_A, X_B, y, pdf, reference_class, max_reg = 1e3, EPS=1E-12):
    """
    Compute N-class density reweighting coefficients.
    
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
            weights_doublet[c][y == c] = pdf[reference_class][inds_0, inds_1] / np.clip(pdf[c][inds_0, inds_1], EPS, None)
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
            EQ = ref_sum / np.clip(np.sum(weights_doublet[c][y == c]), EPS, None)
            weights_doublet[c][y == c] *= EQ
    
    return weights_doublet

