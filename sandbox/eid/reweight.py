# Test re-weighting v.0.0
#
# m.mieskolainen@imperial.ac.uk, 2024

import os
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn import metrics

def x2ind(x, binedges):
    """ Return histogram bin indices for data in x, which needs to be an array [].
    Args:
        x:        data to be classified between bin edges
        binedges: histogram bin edges
    Returns:
        inds:     histogram bin indices
    """
    nbin_edges = len(binedges) - 1
    inds = np.digitize(x, binedges, right=True) - 1

    if len(x) > 1:
        inds[inds >= nbin_edges] = nbin_edges-1
        inds[inds < 0] = 0
    else:
        if inds < 0:
            inds = 0
        if inds >= nbin_edges:
            inds = nbin_edges - 1

    return inds

def generate_events(N, mu, sigma, bin_edges, reference_class, pdf=None):

    x = {}
    for i in range(2):
        # Generate only positive values
        x[i] = np.abs(np.random.normal(loc=mu[i], scale=sigma[i], size=N))

    if pdf is None:
        
        pdf = {}
        for i in range(2):
            pdf[i], _,_ = plt.hist(x[i], bin_edges)
            pdf[i] = pdf[i] / np.sum(pdf[i])

        plt.close()

    # Compute event-by-event re-weights
    w = {}
    EPS = 1e-12
    for c in range(2):
        
        inds = x2ind(x[c], bin_edges)

        # Weight each class against the reference class
        if c is not reference_class:
            w[c] = pdf[reference_class][inds] / np.clip(pdf[c][inds], a_min=EPS, a_max=None)
        else:
            w[c] = np.ones(len(x[c])) # Reference class stays intact

        # Maximum weight cut-off regularization
        w[c][w[c] > MAX_REG] = MAX_REG
        
    Y       = np.hstack((np.zeros(N), np.ones(N)))
    X       = np.hstack((x[0], x[1]))
    
    # Xgboost wants event weights to be at event scale, so N* (due to optimization)
    weights = N*np.hstack((w[0] / np.sum(w[0]), w[1] / np.sum(w[1])))
    
    return X,Y,weights,pdf

# ----------------------------------------
# Config

base = './outputs'
if not os.path.isdir(base) : os.makedirs(base)

# ----------------------------------------
# Parameters

N = int(1e5)        # Number of events
reference_class = 0 # Re-weighting reference
MAX_REG = 1000      # Maximum re-weight cutoff

# Class density parameters
mu    = {}
sigma = {}

mu[0]    = 0.0
sigma[0] = 0.5

mu[1]    = 1.0
sigma[1] = 1.5

# Re-weighting histogram
bin_edges = np.linspace(0,6,200)

# ----------------------------------------
# Generate events

X_train, Y_train, W_train, pdf = generate_events(N, mu, sigma, bin_edges, reference_class)
X_test,  Y_test,  W_test, _    = generate_events(N, mu, sigma, bin_edges, reference_class, pdf)


# ------------------------------------------------
## Train 1D xgboost model

bst = XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1,
                    objective='binary:logistic')

# Fit model
bst.fit(X_train[:,None], Y_train, sample_weight=W_train) # add outer [] with [:,None]

# ------------------------------------------------
# Make predictions

scores = bst.predict(X_test)

# ------------------------------------------------
# Histogram (train) sample

fig,ax = plt.subplots(1,2, figsize=(8,4))

plt.sca(ax[0])
_,bin_edges,_ = plt.hist(X_train[Y_train == 1], bin_edges, label='signal (1)', histtype='step')
_,bin_edges,_ = plt.hist(X_train[Y_train == 0], bin_edges, label='background (0)', histtype='step')
plt.legend()
plt.xlabel('x')

plt.sca(ax[1])
plt.title(f'reference class = {reference_class}')
_,bin_edges,_ = plt.hist(X_train[Y_train == 1], bin_edges, weights=W_train[Y_train == 1], label='weighted signal (1)', histtype='step')
_,bin_edges,_ = plt.hist(X_train[Y_train == 0], bin_edges, weights=W_train[Y_train == 0], label='weighted background (0)', histtype='step')
plt.legend()
plt.xlabel('x')
plt.savefig(f'{base}/fig_histograms.pdf')
plt.close()

# ------------------------------------------------
# Check the decision as a function of x

fig,ax = plt.subplots(1,2,figsize=(8,4))

ind    = np.argsort(X_test.squeeze())

plt.sca(ax[0])
plt.plot(X_test[ind], scores[ind], label='BDT score')
plt.xlabel('x')
plt.ylabel('BDT score')
plt.title('Decision for each x')

# Check BDT score distribution

plt.sca(ax[1])
plt.hist(scores, bins=np.linspace(0,1,100))
plt.xlabel('BDT score')
plt.ylabel('counts')
plt.title('Decision distribution')
plt.savefig(f'{base}/fig_BDT_scores.pdf')
plt.close()

# ------------------------------------------------
# ROCs

fig,ax = plt.subplots()

# Cut on BDT score with weights included
fpr, tpr, thresholds = metrics.roc_curve(Y_test, scores, sample_weight=W_test)
auc = metrics.roc_auc_score(y_true=Y_test, y_score=scores, sample_weight=W_test)
plt.plot(fpr, tpr, label=f'BDT (weight in eval) {auc:0.2f}')

# Cut on BDT score
fpr, tpr, thresholds = metrics.roc_curve(Y_test, scores)
auc = metrics.roc_auc_score(y_true=Y_test, y_score=scores)
plt.plot(fpr, tpr, label=f'BDT (no weights in eval) {auc:0.2f}')

# Cut on x with weights included
fpr, tpr, thresholds = metrics.roc_curve(Y_test, X_test, sample_weight=W_test)
auc = metrics.roc_auc_score(y_true=Y_test, y_score=X_test, sample_weight=W_test)
plt.plot(fpr, tpr, label=f'cut on $x$ (weights in eval) {auc:0.2f}')

# Cut on x
fpr, tpr, thresholds = metrics.roc_curve(Y_test, X_test)
auc = metrics.roc_auc_score(y_true=Y_test, y_score=X_test)
plt.plot(fpr, tpr, label=f'cut on $x$ (no weights in eval) {auc:0.2f}')

plt.legend()
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig(f'{base}/fig_ROC.pdf')
plt.close()
