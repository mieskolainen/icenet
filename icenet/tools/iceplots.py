# Plotting functions
# 
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import matplotlib.pyplot as plt
import numpy as np
import torch
import xgboost

from . import aux


# Training evolution
def plot_train_evolution(losses, trn_aucs, val_aucs, label):

    fig,ax = plt.subplots(1, 2, figsize = (8,6))
    
    ax[0].plot(losses)
    ax[0].set_xlabel('k (epoch)')
    ax[0].set_ylabel('train loss')
    ax[0].set_title(label)

    ax[1].plot(trn_aucs)
    ax[1].plot(val_aucs)
    ax[1].legend(['train','validation'])
    ax[1].set_xlabel('k (epoch)')
    ax[1].set_ylabel('AUC')
    ax[1].grid(True)
    
    ratio = 1.0
    ax[0].set_aspect(1.0/ax[0].get_data_ratio()*ratio)

    for i in [1]:
        ax[i].set_ylim([0.5, 1.0])
        ax[i].set_aspect(1.0/ax[i].get_data_ratio()*ratio)

    return fig,ax


# Evaluate AUC per bin
#
def binned_AUC(func_predict, X, y, X_kin, VARS_kin, pt_edges, eta_edges, label):

    y_tot      = np.array([])
    y_pred_tot = np.array([])

    AUC = np.zeros((len(pt_edges)-1, len(eta_edges)-1))

    for i in range(len(pt_edges) - 1):
        for j in range(len(eta_edges) - 1):

            pt_range  = [ pt_edges[i],  pt_edges[i+1]]
            eta_range = [eta_edges[j], eta_edges[j+1]]

            # Indices
            ind = np.logical_and(aux.pick_ind(X_kin[:, VARS_kin.index('trk_pt')],   pt_range),
                                 aux.pick_ind(X_kin[:, VARS_kin.index('trk_eta')], eta_range))

            print('\nEvaluate classifier ...')
            print('*** PT = [{:.3f},{:.3f}], ETA = [{:.3f},{:.3f}] ***'.format(pt_range[0], pt_range[1], eta_range[0], eta_range[1]))
            y_pred = func_predict(X[ind, :])
            
            # Evaluate metric
            met = aux.Metric(y_true = y[ind], y_soft = y_pred)
            print('AUC = {:.5f}'.format(met.auc))

            # Accumulate
            y_tot      = np.concatenate((y_tot, y[ind]))
            y_pred_tot = np.concatenate((y_pred_tot, y_pred))

            AUC[i,j]   = met.auc

    # Evaluate total performance
    met = aux.Metric(y_true = y_tot, y_soft = y_pred_tot)
    fig,ax = plot_auc_matrix(AUC, pt_edges, eta_edges)
    ax.set_title('{}: Integrated AUC = {:.3f}'.format(label, met.auc))

    return fig,ax,met

# Add text annotations to a matrix heatmap plot
#
def annotate_heatmap(X, ax, xlabels, ylabels, x_rot = 90, y_rot = 0, decimals = 1, color = "w"):

    ax.set_xticks(np.arange(0, len(xlabels), 1));
    ax.set_yticks(np.arange(0, len(ylabels), 1));

    ax.set_xticklabels(labels = xlabels, rotation = x_rot, fontsize = 'xx-small')
    ax.set_yticklabels(labels = ylabels, rotation = y_rot, fontsize = 'xx-small')

    # Loop over data dimensions and create text annotations.
    for i in range(len(ylabels)):
        for j in range(len(xlabels)):
            
            if (decimals == 0):
                text = ax.text(j, i, '{:.0f}'.format(X[i,j]), ha="center", va="center", color=color)
            if (decimals == 1):
                text = ax.text(j, i, '{:.1f}'.format(X[i,j]), ha="center", va="center", color=color)
            if (decimals == 2):
                text = ax.text(j, i, '{:.2f}'.format(X[i,j]), ha="center", va="center", color=color)

    return ax


# Plot AUC matrix
#
#
def plot_auc_matrix(AUC, pt_edges, eta_edges):

    fig, ax = plt.subplots()

    xlabels = []
    for k in range(len(eta_edges) - 1):
        xlabels.append('[{},{}]'.format(eta_edges[k], eta_edges[k+1]))
    ylabels = []
    for k in range(len(pt_edges) - 1):
        ylabels.append('[{},{}]'.format(pt_edges[k], pt_edges[k+1]))

    ax.imshow(AUC, origin = 'lower')
    ax.set_xlabel('$\eta$')
    ax.set_ylabel('$p_t$ (GeV)')
    ax.set_title('AUC x 100')
    
    ax = annotate_heatmap(X = AUC * 100, ax = ax, xlabels = xlabels,
        ylabels = ylabels, decimals = 0, x_rot = 0, y_rot = 0, color = "w")

    return fig, ax


# Plot all variables
#
#
def plotvars(X, y, VARS, weights, NBINS = 70, title = '', targetdir = '.'):

    for i in range(X.shape[1]):
        x = X[:,i]
        var = VARS[i]
        plotvar(x, y, var, weights, NBINS, title, targetdir)


# Plot single variable
#
#
def plotvar(x, y, var, weights, NBINS = 70, title = '', targetdir = '.'):

    bins = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99), NBINS)
    plot_reweight_result(x, y, bins, weights, title = title, xlabel = var)
    plt.savefig('{}/{}.pdf'.format(targetdir, var)); plt.close()


# N.B. Here plot with pure event counts
# so we see that also integrated class fractions are equalized (or not)!
#
def plot_reweight_result(X, y, bins, trn_weights, title = '', xlabel = 'x') :

  # Plot re-weighting results
  fig,(ax1,ax2) = plt.subplots(1, 2, figsize = (10,5))

  for i in range(2):

    if i == 0:
      ax = ax1
    if i == 1:
      ax = ax2

    # Loop over classes
    for c in range(2) :
        w = trn_weights[y == c]
        ax.hist(X[y == c], bins, density = False,
          histtype = 'step', fill = False, linewidth = 1.5)

    # Loop over classes
    for c in range(2) :
        w = trn_weights[y == c]
        ax.hist(X[y == c], bins, weights = w, density = False,
          histtype = 'step', fill = False, linestyle = '-', linewidth = 1.5)

    ax.set_ylabel('weighted counts')
    ax.set_xlabel(xlabel)

  ax1.set_title(title)
  ax1.legend(['background','signal', 'background (w)','signal (w)'])
  ax2.set_yscale('log')
  plt.tight_layout()


# Plot cross-correlations
# Data in format (# samples x # dimensions)
#
def plot_correlations(X, netvars, colorbar = False):

    import pandas as pd
    from sklearn import neighbors
    import seaborn as sns

    C = np.corrcoef(X, rowvar = False) * 100
    C[np.abs(C) < 0.5] = 0 # round near zero to 0

    N = np.ceil(C.shape[0]/3)
    fig,ax = plt.subplots(1,1,figsize=(N,N))

    ax.imshow(C)
    ax = annotate_heatmap(X = C, ax = ax, xlabels = netvars,
        ylabels = netvars, decimals = 0, x_rot = 90, y_rot = 0, color = "w")
    ax.set_title('linear correlation $\in$ [-100,100]')
    
    if colorbar:
        cb = plt.colorbar()

    return fig,ax

# 'Receiver Operating Characteristics'
#  False positive (x) vs True positive (y)
#
def ROC_plot(metrics, labels, title = '', filename = 'ROC') :

    fig,ax = plt.subplots()
    xx = np.logspace(-5, 0, 100)
    plt.plot(xx, xx, linestyle='--', color='black', linewidth=1) # ROC diagonal

    for i in range(len(metrics)) :
        plt.plot(metrics[i].fpr, metrics[i].tpr, label = '{}: AUC = {:.3f}'.format(labels[i], metrics[i].auc))

    plt.legend()

    ax.set_xlabel('False Positive (background) rate $\\alpha$')
    ax.set_ylabel('True Positive (signal) rate $1-\\beta$')
    ax.set_title(title)
    ax.set_aspect(1.0/ax.get_data_ratio() * 1.0)


    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, 1.0)
    plt.savefig(filename + '.pdf')

    plt.gca().set_xscale('log')
    plt.ylim(0.0, 1.0)
    plt.xlim(1e-4, 1.0)
    plt.savefig(filename + '_log.pdf')

    plt.close()


# Histogram observables 1D
#
def plothist1d(X, y, labels) :

    # Number of classes
    C = int(np.max(y) - np.min(y) + 1)

    classlegs = []
    for k in range(C) :
        classlegs.append('class ' + str(k))

    # Over observables
    for j in range(X.shape[1]) :

        fig,ax = plt.subplots()
        
        # Over classes
        for k in range(C):

            ind = (y == k)
            x = X[ind,j]
            edges = np.linspace(0, np.max(x), 100);

            hI, bins, patches = plt.hist(x, edges,
                density = True, histtype = 'step', fill = False, linewidth = 2, label = 'inverse')

        plt.legend(classlegs)
        plt.xlabel('x')
        plt.ylabel('density')
        plt.title(labels[j])

        plt.gca().set_yscale('log')


### Decision contour pairwise for each dimension,
# other dimensions evaluated at zero=(0,0,...0)
# 
def plot_decision_contour(pred_func, X, y, labels, targetdir = '.', matrix = 'numpy'):

    print(__name__ + '.plot_decision_contour')
    
    NPOINTS = 3000
    MAXP = min(NPOINTS, X.shape[0])

    D = X.shape[1]

    N = 100 # resolution

    for dim1 in range(D) :
        for dim2 in range(D) :

            if dim2 <= dim1 :
                continue

            # (x,y)-plane limits
            pad = 0.5
            x_min, x_max = X[:, dim1].min() - pad, X[:, dim1].max() + pad
            y_min, y_max = X[:, dim2].min() - pad, X[:, dim2].max() + pad

            # Grid points
            PX,PY = np.meshgrid(np.linspace(x_min, x_max, N),
                                np.linspace(y_min, y_max, N))
            
            # Function values through 'pred_func' lambda            
            Z = np.zeros((N*N, D))
            Z[:, dim1] = PX.ravel()
            Z[:, dim2] = PY.ravel()

            signalclass = 1
            if (matrix == 'torch'):
                Z = pred_func(torch.tensor(Z, dtype=torch.float32))
                Z = Z[:, signalclass].detach().numpy() # 2 output units
            if (matrix == 'numpy'):
                Z = pred_func(Z)
            if (matrix == 'xgboost'):
                Z = pred_func(xgboost.DMatrix(data = Z))

            Z = Z.reshape(PX.shape)

            fig, axs = plt.subplots()

            # Contour
            cs = plt.contourf(PX, PY, Z, cmap = plt.cm.Spectral)

            # Samples as dots
            plt.scatter(X[0:MAXP, dim1], X[0:MAXP, dim2], c = y[0:MAXP], cmap = plt.cm.binary)

            plt.xlabel('X[%d]' % dim1 + ' (%s)' % labels[dim1])
            plt.ylabel('X[%d]' % dim2 + ' (%s)' % labels[dim2])
            plt.colorbar(cs, ticks = np.linspace(0.0, 1.0, 11))
            
            plt.savefig(targetdir + str(dim1) + "_" + str(dim2) + ".pdf")
            plt.close()
