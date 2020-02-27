# Auxialary functions
# 
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import stats



# Data in format (# samples x # dimensions)
#
#
def print_variables(X : np.array, VARS) :

    print('\n')
    print('[i] variable_name : [min, med, max]   mean +- std   [[isinf, isnan]]')
    for j in range( X.shape[1]):

        minval = np.min(X[:,j])
        maxval = np.max(X[:,j])
        mean   = np.mean(X[:,j])
        med    = np.median(X[:,j])
        std    = np.std(X[:,j])

        isinf  = np.any(np.isinf(X[:,j]))
        isnan  = np.any(np.isnan(X[:,j]))

        print('[{: >3}]{: >35} : [{: >10.2f}, {: >10.2f}, {: >10.2f}] \t {: >10.2f} +- {: >10.2f}   [[{}, {}]]'
            .format(j, VARS[j], minval, med, maxval, mean, std, isinf, isnan))

# Load pytorch checkpoint
#
#
def load_checkpoint(filepath) :
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model


# PyTorch model saver
#
#
def save_torch_model(model, optimizer, epoch, path):
    def f():
        print('Saving model..')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, (path))
    
    return f


# PyTorch model loader
#
#
def load_torch_model(model, optimizer, param, path, load_start_epoch = False):
    def f():
        print('Loading model..')
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        if load_start_epoch:
            param.start_epoch = checkpoint['epoch']

    return f


# Compute reweighting coefficients
#
#
def reweight_aux(X, y, binedges, shape_reference = 'signal', max_reg = 1e3) :
    
    EPS = 1e-12

    # Re-weighting weights
    weights_doublet = np.zeros((X.shape[0], 2)) # Init with zeros!!

    # Take re-weighting variables
    pdf0, bins, patches = plt.hist(x = X[y == 0], bins = binedges)
    pdf1, bins, patches = plt.hist(x = X[y == 1], bins = binedges)

    # Make them densities
    pdf0 = pdf0 / np.sum(pdf0)
    pdf1 = pdf1 / np.sum(pdf1)

    # Indexing
    inds = x2ind(X[y == 0], binedges)
    C0 = np.ones((inds.shape[0]))
    if (shape_reference == 'signal'):
        C0 = pdf1[inds] / (pdf0[inds] + EPS)

    inds = x2ind(X[y == 1], binedges)
    C1 = np.ones((inds.shape[0]))
    if (shape_reference == 'background'):
        C1 = pdf0[inds] / (pdf1[inds] + EPS)

    if (shape_reference == 'none'):
        C0 = C1 = 1

    # Maximum threshold regularization
    weights_doublet[weights_doublet > max_reg] = max_reg

    # Save weights
    weights_doublet[y == 0, 0] = C0
    weights_doublet[y == 1, 1] = C1

    return weights_doublet


# Balance class weights to sum to equal counts
#
#
def balanceweights(weights_doublet, y):
    EQ = np.sum(weights_doublet[y == 0, 0]) / np.sum(weights_doublet[y == 1, 1])
    weights_doublet[y == 1,1] *= EQ

    return weights_doublet


# Compute TRAINING re-weighting coefficients for each vector
#
# Input: X = Observable of interest (N x 1)
#        y = signal (1) and background (0) labels
#        shape_reference = 'signal' or 'background' or 'none'
#
def reweightcoeff1D(X, y, binedges, shape_reference = 'signal', equalize_classes = True, max_reg = 1e3) :

    weights_doublet = reweight_aux(X, y, binedges, shape_reference, max_reg)

    # Apply class balance equalizing weight
    if (equalize_classes == True):
        weights_doublet = balanceweights(weights_doublet, y)

    # Get 1D array
    weights = np.sum(weights_doublet, axis=1)

    return weights


# Compute TRAINING re-weighting coefficients for each vector
#
# 2D with factorized marginal 1D distributions
# 
# Input: XA, XB = Observables of interest (N x 1)
#        y = signal (1) and background (0) labels
#        shape_reference = 'signal' or 'background' or 'none'
# 
def reweightcoeff2DFP(X_A, X_B, y, binedges_A, binedges_B, shape_reference = 'signal', equalize_classes = True, max_reg = 1e3) :

    weights_doublet_A = reweight_aux(X_A, y, binedges_A, shape_reference, max_reg)
    weights_doublet_B = reweight_aux(X_B, y, binedges_B, shape_reference, max_reg)

    weights_doublet   = weights_doublet_A * weights_doublet_B

    # Apply class balance equalizing weight
    if (equalize_classes == True):
        weights_doublet = balanceweights(weights_doublet, y)

    # Get 1D array
    weights = np.sum(weights_doublet, axis=1)

    return weights


# Compute TRAINING re-weighting coefficients for each vector
# 
# Full 2D without factorization
#
# Input: XA, XB = Observables of interest (N x 1)
#        y = signal (1) and background (0) labels
#        shape_reference = 'signal' or 'background' or 'none'
# 
def reweightcoeff2D(X_A, X_B, y, binedges_A, binedges_B, shape_reference = 'signal', equalize_classes = True, max_reg = 1e3) :

    EPS = 1e-12
    
    # Re-weighting weights
    weights_doublet = np.zeros((X_A.shape[0], 2)) # Init with zeros!!

    # Take re-weighting variables
    pdf0, foo, bar, zoo = plt.hist2d(x = X_A[y == 0], y = X_B[y == 0], bins = [binedges_A, binedges_B])
    pdf1, foo, bar, zoo = plt.hist2d(x = X_A[y == 1], y = X_B[y == 1], bins = [binedges_A, binedges_B])

    # Make them densities
    pdf0 = pdf0 / np.sum(pdf0)
    pdf1 = pdf1 / np.sum(pdf1)

    # Indexing
    inds_A = x2ind(X_A[y == 0], binedges_A)
    inds_B = x2ind(X_B[y == 0], binedges_B)

    C0 = np.ones((inds_A.shape[0]))
    if (shape_reference == 'signal'):
        C0 = pdf1[inds_A, inds_B] / (pdf0[inds_A, inds_B] + EPS)

    inds_A = x2ind(X_A[y == 1], binedges_A)
    inds_B = x2ind(X_B[y == 1], binedges_B)

    C1 = np.ones((inds_A.shape[0]))
    if (shape_reference == 'background'):
        C1 = pdf0[inds_A, inds_B] / (pdf1[inds_A, inds_B] + EPS)

    if (shape_reference == 'none'):
        C0 = C1 = 1

    # Save weights
    weights_doublet[y == 0, 0] = C0
    weights_doublet[y == 1, 1] = C1

    # Maximum threshold regularization
    weights_doublet[weights_doublet > max_reg] = max_reg

    # Apply class balance equalizing weight
    if (equalize_classes == True):
        weights_doublet = balanceweights(weights_doublet, y)

    # Get 1D array
    weights = np.sum(weights_doublet, axis=1)

    return weights


# Return indices
#
def pick_ind(x, minmax):
    return (x >= minmax[0]) & (x <= minmax[1])


# Return histogram bin indices for data in x, which needs to be array []
# 
def x2ind(x, binedges) :
    NBINS = len(binedges) - 1
    inds = np.digitize(x, binedges, right=True) - 1

    if len(x) > 1:
        inds[inds >= NBINS] = NBINS-1
        inds[inds < 0] = 0
    else:
        if inds < 0:
            inds = 0
        if inds >= NBINS:
            inds = NBINS - 1

    return inds


# Soft decision to hard decision at point 0.5
# Input y_soft needs to be understood as probabilities for two classes!
#
def hardclass(y_soft, valrange):
    y_out = y_soft[:]

    boundary = (valrange[1] - valrange[0]) / 2
    y_out[y_out  > boundary] = 1
    y_out[y_out <= boundary] = 0

    return y_out


# Classifier performance evaluation
# Input in y_pred needs to be understood as probabilities for two classes!
#
class Metric:

    def __init__(self, y_true, y_soft, valrange = [0,1]) :

        ok = np.isfinite(y_true) & np.isfinite(y_soft)

        lhs = len(y_true) 
        rhs = (ok == True).sum()
        if (lhs != rhs) :
            print('Metric: input length = {} with not-finite values = {}'.format(lhs, lhs-rhs))
            print(y_soft)

        # invalid input
        if (np.sum(y_true == 0) == 0) | (np.sum(y_true == 1) == 0):
            print('Metric: only one class present in y_true, cannot evaluate metrics')
            self.fpr = -1
            self.tpr = -1
            self.thresholds = -1
            self.auc = -1
            self.acc = -1
            return

        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(y_true = y_true[ok], y_score = y_soft[ok])
        self.auc = metrics.roc_auc_score(y_true  = y_true[ok], y_score = y_soft[ok])
        self.acc = metrics.accuracy_score(y_true = y_true[ok], y_pred = hardclass(y_soft = y_soft[ok], valrange = valrange))


