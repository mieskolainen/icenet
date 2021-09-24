# Plotting functions
# 
# Mikael Mieskolainen, 2021
# m.mieskolainen@imperial.ac.uk

import matplotlib.pyplot as plt
import numpy as np
import torch
import xgboost
import os

from tqdm import tqdm


from iceplot import iceplot

from icenet.tools import aux
from icenet.tools import process


def plot_selection(X, ind, ids, args, label, varlist, density=True):
    """
    Plot selection before / after type histograms against all chosen variables

    Args:
        X      : data array (N events x D dimensions)
        ind    : boolean selection indices (N)
        ids    : variable string array (D)
        args   : plotting arguments
        label  : a string label
        varlist: a list of variables to be plotted (from ids)
        density: normalize all histograms to unit density
    """
    
    for var in tqdm(varlist):

        if var not in ids:
            continue

        # Histogram (autobinning)
        counts1, errs1, bins, cbins = iceplot.hist(np.asarray(X[:, ids.index(var)]),   bins=100,  density=density)
        counts2, errs2, bins, cbins = iceplot.hist(np.asarray(X[ind, ids.index(var)]), bins=bins, density=density)
        
        # Plot
        obs_x = {
            'xlim'      : (np.min(bins), np.max(bins)),
            'ylim'      : None,
            'ylim_ratio': (0.7, 1.3),
            
            'xlabel'    : var,
            'ylabel'    : r'Counts',
            'units'     : {'x': None, 'y' : r'counts'},
            'density'   : density,
            'figsize'   : (4, 3.75)
        }
        fig, ax = iceplot.create_axes(**obs_x, ratio_plot=True)

        label1  = f'before cuts'
        label2  = f'after cuts'

        ax[0].hist(x=cbins, bins=bins, weights=counts1, color=(0,0,0), label=label1,            **iceplot.hist_style_step)
        ax[0].hist(x=cbins, bins=bins, weights=counts2, color=(1,0,0), alpha=0.5, label=label2, **iceplot.hist_style_step)
        
        iceplot.ordered_legend(ax=ax[0], order=[label1, label2])
        iceplot.plot_horizontal_line(ax[1])

        ax[1].hist(x=cbins, bins=bins, weights=counts2 / (counts1 + 1E-30), color=(1,0,0), alpha=0.5, label='ratio', **iceplot.hist_style_step)

        # Save it
        targetdir = f'./figs/{args["rootname"]}/{args["config"]}/cuts/{label}'

        os.makedirs(targetdir, exist_ok = True)
        fig.savefig(f'{targetdir}/{var}.pdf', bbox_inches='tight')

        ax[0].set_yscale('log')
        fig.savefig(f'{targetdir}/{var}__log.pdf', bbox_inches='tight')
        plt.close()


def plot_matrix(XY, x_bins, y_bins, vmin=0, vmax=None, cmap='RdBu', figsize=(4,3), grid_on=False):
    """
    Visualize matrix.
    """
    fig,ax = plt.subplots(figsize=figsize)
    
    ax.grid(grid_on)

    x,y    = np.meshgrid(x_bins, y_bins)
    # Note transpose
    c      = ax.pcolormesh(x, y, XY.T, cmap=cmap, vmin=vmin, vmax=vmax, antialiased=True, linewidth=0.0)
    
    ax.axis([x.min(), x.max(), y.min(), y.max()])

    return fig,ax,c


def plot_train_evolution(losses, trn_aucs, val_aucs, label):
    """ Training evolution plots.

    Args:
        losses:   loss values
        trn_aucs: training matrices
        val_aucs: validation metrics
    
    Returns:
        fig: figure handle
        ax:  figure axis
    """
    
    fig,ax = plt.subplots(1,2,figsize=(8,6))
    
    ax[0].plot(losses)
    ax[0].set_xlabel('k (epoch)')
    ax[0].set_ylabel('train loss')
    ax[0].set_title(label, fontsize=10)

    ax[1].plot(trn_aucs)
    ax[1].plot(val_aucs)
    ax[1].legend(['train','validation'])
    ax[1].set_xlabel('k (epoch)')
    ax[1].set_ylabel('AUC')
    ax[1].grid(True)
    
    ratio = 1.0
    ax[0].set_aspect(1.0/ax[0].get_data_ratio()*ratio)

    for i in [1]:
        ax[1].set_ylim([0.5, 1.0])
        ax[1].set_aspect(1.0/ax[i].get_data_ratio()*ratio)

    return fig,ax


def binned_2D_AUC(func_predict, X, y, X_kin, VARS_kin, edges_A, edges_B, label, ids=['trk_pt', 'trk_eta']):
    """
    Evaluate AUC per 2D-bin.
    
    Args:
        func_predict:  Function handle of the classifier
        X           :  Input data
        y           :  Output (truth level target) data
        X_kin       :  Kinematic (A,B) data
        VARS_kin    :  Kinematic variables (strings)
        edges_A     :  Edges of the A-space cells
        edges_B     :  Edges of the B-space cells
        label       :  Label of the classifier (string)
        ids         :  Variable identifiers
    
    Returns:
        fig,ax      :  Figure handle and axis
        met         :  Metrics object
    """

    AUC = np.zeros((len(edges_A)-1, len(edges_B)-1))
    y_pred = process.compute_predictions(func_predict=func_predict, X=X)

    if len(y_pred) != len(y):
        raise Exception(__name__ + f'.binned_2D_AUC: len(y_pred) = {len(y_pred)} != len(y) = {len(y)}')

    # Loop over bins
    for i in range(len(edges_A)-1):
        for j in range(len(edges_B)-1):

            range_A = [edges_A[i], edges_A[i+1]]
            range_B = [edges_B[j], edges_B[j+1]]

            # Indices
            ind = np.logical_and(aux.pick_ind(X_kin[:, VARS_kin.index(ids[0])], range_A),
                                 aux.pick_ind(X_kin[:, VARS_kin.index(ids[1])], range_B))

            string = f'{ids[0]} = [{range_A[0]:.3f},{range_A[1]:.3f}), {ids[1]} = [{range_B[0]:.3f},{range_B[1]:.3f})'
            
            if np.sum(ind) > 0: # Do we have any events in this cell
                
                # Evaluate metric
                met      = aux.Metric(y_true=y[ind], y_soft=y_pred[ind])
                print(f'{string} | AUC = {met.auc:.5f}')
                AUC[i,j] = met.auc

            else:
                print(f'{string} | No events found in this cell!')


    # Evaluate total performance
    met = aux.Metric(y_true=y, y_soft=y_pred)

    # Finally plot it
    fig,ax = plot_AUC_matrix(AUC=AUC, edges_A=edges_A, edges_B=edges_B)
    ax.set_title(f'{label}: AUC = {met.auc:.3f} (integrated)', fontsize=9)
    ax.set_xlabel(f"{ids[0]}")
    ax.set_ylabel(f"{ids[1]}")

    return fig, ax, met


def binned_1D_AUC(func_predict, X, y, X_kin, VARS_kin, edges, label, ids='trk_pt'):
    """
    Evaluate AUC & ROC per 1D-bin.
    
    Args:
        func_predict:  Function handle of the classifier
        X           :  Input data
        y           :  Output (truth level target) data
        X_kin       :  Kinematic (A,B) data
        VARS_kin    :  Kinematic variables (strings)
        edges       :  Edges of the space cells
        label       :  Label of the classifier (string)
        ids         :  Variable identifier
    
    Returns:
        fig,ax      :  Figure handle and axis
        met         :  Metrics object
    """

    AUC = np.zeros((len(edges)-1, 1))
    y_pred = process.compute_predictions(func_predict=func_predict, X=X)

    if len(y_pred) != len(y):
        raise Exception(__name__ + f'.binned_1D_AUC: len(y_pred) = {len(y_pred)} != len(y) = {len(y)}')

    METS   = []
    LABELS = []

    # Loop over bins
    for i in range(len(edges)-1):

            range_ = [edges[i], edges[i+1]]

            # Indices
            ind = aux.pick_ind(X_kin[:, VARS_kin.index(ids)], range_)
            
            string = f'{ids} = [{range_[0]:5.3f},{range_[1]:5.3f}]'

            if np.sum(ind) > 0: # Do we have any events in this cell
                
                # Evaluate metric
                met    = aux.Metric(y_true=y[ind], y_soft=y_pred[ind])
                AUC[i] = met.auc
                METS.append(met)
                print(f'{string} | AUC = {AUC[i]}')
            else:
                METS.append(None)
                print(f'{string} | No events found in this cell!')

            LABELS.append(f'{ids} = [{range_[0]:.1f},{range_[1]:.1f})')

    return METS, LABELS


def density_MVA_output(func_predict, X, y, label, hist_edges=80):
    """
    Evaluate MVA output density per class.
    
    Args:
        func_predict:  Function handle of the classifier
        X           :  Input data
        y           :  Output (truth level target) data
        label       :  Label of the classifier (string)
        hist_edges  :  Histogram edges list (or number of bins, as an alternative)
    
    Returns:
        fig,ax      :  Figure handle and axis
        met         :  Metrics object
    """

    y_pred    = process.compute_predictions(func_predict=func_predict, X=X)

    # --------------------------------------------------------------------

    # Number of classes
    C         = int(np.max(y) - np.min(y) + 1)

    classlegs = [f'class {k}, $N={np.sum(y == k)}$' for k in range(C)]
    fig,ax    = plt.subplots()

    # Over classes
    for k in range(C):
        ind = (y == k)
        hI, bins, patches = plt.hist(y_pred[ind], hist_edges,
            density = True, histtype = 'step', fill = False, linewidth = 2, label = 'inverse')
    
    plt.legend(classlegs, loc='upper center')
    plt.xlabel('MVA output $f(\\mathbf{{x}})$')
    plt.ylabel('density')
    plt.title(label, fontsize=10)
    
    ax.set_yscale('log')
    
    return fig, ax


def annotate_heatmap(X, ax, xlabels, ylabels, x_rot = 90, y_rot = 0, decimals = 1, color = "w"):
    """ Add text annotations to a matrix heatmap plot
    """

    ax.set_xticks(np.arange(0, len(xlabels), 1));
    ax.set_yticks(np.arange(0, len(ylabels), 1));

    ax.set_xticklabels(labels=xlabels, rotation=x_rot, fontsize='xx-small')
    ax.set_yticklabels(labels=ylabels, rotation=y_rot, fontsize='xx-small')

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


def plot_AUC_matrix(AUC, edges_A, edges_B):
    """ Plot AUC matrix.

    Args:
        AUC:      AUC-ROC matrix
        edges_A:  histogram edges of variable A
        edges_B:  histogram edges of variable B
    
    Returns:
        fig:      figure handle
        ax:       figure axis
    """

    fig, ax = plt.subplots()

    xlabels = [f'[{edges_A[k]},{edges_A[k+1]})' for k in range(len(edges_A) - 1)]
    ylabels = [f'[{edges_B[k]},{edges_B[k+1]})' for k in range(len(edges_B) - 1)]

    ax.imshow(AUC.transpose(), origin = 'lower')
    ax.set_title('AUC x 100', fontsize=10)
    
    ax = annotate_heatmap(X = AUC.transpose() * 100, ax = ax, xlabels = xlabels,
        ylabels = ylabels, decimals = 0, x_rot = 0, y_rot = 0, color = "w")

    return fig, ax


def plotvars(X, y, ids, weights, NBINS = 70, title = '', targetdir = '.'):
    """ Plot all variables.
    """
    for i in tqdm(range(X.shape[1])):
        x = X[:,i]
        var = ids[i]
        plotvar(x, y, var, weights, NBINS, title, targetdir)


def plotvar(x, y, var, weights, NBINS = 70, title = '', targetdir = '.'):
    """ Plot a single variable.
    """
    bins = np.linspace(np.percentile(x, 0.5), np.percentile(x, 99), NBINS)
    plot_reweight_result(x, y, bins, weights, title = title, xlabel = var)
    plt.savefig(f'{targetdir}/{var}.pdf', bbox_inches='tight')
    plt.close()


def plot_reweight_result(X, y, bins, trn_weights, title = '', xlabel = 'x'):
    """ Here plot pure event counts
        so we see that also integrated class fractions are equalized (or not)!
    """
    # Plot re-weighting results
    fig,(ax1,ax2) = plt.subplots(1, 2, figsize = (10,5))

    for i in range(2):
        ax = ax1 if i == 0 else ax2

        # Loop over classes
        for c in range(2) :
            w = trn_weights[y == c]
            ax.hist(X[y == c], bins, density = False,
                histtype = 'step', fill = False, linewidth = 1.5)

        # Loop over classes
        for c in range(2) :
            w = trn_weights[y == c]
            ax.hist(X[y == c], bins, weights = w, density = False,
                histtype = 'step', fill = False, linestyle = '--', linewidth = 2.0)
        
        ax.set_ylabel('weighted counts')
        ax.set_xlabel(xlabel)

    ax1.set_title(title, fontsize=10)
    ax1.legend(['class 0','class 1', 'class 0 (w)','class 1 (w)'])
    ax2.set_yscale('log')
    plt.tight_layout()


def plot_correlations(X, netvars, colorbar = False):
    """ Plot cross-correlations
    Data in format (# samples x # dimensions)
    """
    
    C = np.corrcoef(X, rowvar = False) * 100
    C[np.abs(C) < 0.5] = 0 # round near zero to 0

    N = np.ceil(C.shape[0]/3)
    fig,ax = plt.subplots(1,1,figsize=(N,N))

    ax.imshow(C)
    ax = annotate_heatmap(X = C, ax = ax, xlabels = netvars,
        ylabels = netvars, decimals = 0, x_rot = 90, y_rot = 0, color = "w")
    ax.set_title('linear correlation $\\in$ [-100,100]', fontsize=10)
    
    if colorbar:
        cb = plt.colorbar()

    print(__name__ + f'.plot_correlations: [done]')

    return fig,ax


def ROC_plot(metrics, labels, title = '', filename = 'ROC', legend_fontsize=7) :
    """ Receiver Operating Characteristics i.e. False positive (x) vs True positive (y)
    """

    for k in [0,1]: # linear & log
        
        fig,ax = plt.subplots()
        xx = np.logspace(-5, 0, 100)
        plt.plot(xx, xx, linestyle='--', color='black', linewidth=1) # ROC diagonal

        for i in range(len(metrics)):

            linestyle = '-'
            marker    = 'None'

            fpr = metrics[i].fpr
            tpr = metrics[i].tpr

            # Autodetect a ROC point triangle (instead of a curve)
            if len(np.unique(fpr)) == 3:
                tpr = tpr[1:-1] # Remove first and last
                fpr = fpr[1:-1] # Remove first and last
                
                linestyle = "None"
                marker    = 'o'
            
            # A ROC-curve
            else:
                fpr = fpr[1:] # Remove always the first element for log-plot reasons
                tpr = tpr[1:]

            plt.plot(fpr, tpr, linestyle=linestyle, marker=marker, \
                label = '{}: AUC = {:.3f}'.format(labels[i], metrics[i].auc))

        plt.legend(fontsize=legend_fontsize)
        ax.set_xlabel('False Positive (background) rate $\\alpha$')
        ax.set_ylabel('True Positive (signal) rate $1-\\beta$')
        ax.set_title(title, fontsize=10)

        if k == 0:
            plt.ylim(0.0, 1.0)
            plt.xlim(0.0, 1.0)
            ax.set_aspect(1.0/ax.get_data_ratio() * 1.0)
            plt.savefig(filename + '.pdf', bbox_inches='tight')

        if k == 1:
            plt.ylim(0.0, 1.0)
            plt.xlim(1e-4, 1.0)
            plt.gca().set_xscale('log')
            ax.set_aspect(1.0/ax.get_data_ratio() * 0.75)
            plt.savefig(filename + '_log.pdf', bbox_inches='tight')

        plt.close()


def plothist1d(X, y, labels) :
    """ Histogram observables in 1D
    """

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
        plt.title(labels[j], fontsize=10)

        plt.gca().set_yscale('log')


def plot_decision_contour(pred_func, X, y, labels, targetdir = '.', matrix = 'numpy', reso=50, npoints=400):
    """ Decision contour pairwise for each dimension,
    other dimensions evaluated at zero=(0,0,...0)
    """
    
    print(__name__ + '.plot_decision_contour ...')
    MAXP = min(npoints, X.shape[0])
    D = X.shape[1]
    pad = 0.5

    for dim1 in tqdm(range(D)) :
        x_min, x_max = X[:, dim1].min() - pad, X[:, dim1].max() + pad
        for dim2 in range(D) :
            if dim2 <= dim1 :
                continue

            # (x,y)-plane limits
            y_min, y_max = X[:, dim2].min() - pad, X[:, dim2].max() + pad

            # Grid points
            PX,PY = np.meshgrid(np.linspace(x_min, x_max, reso), np.linspace(y_min, y_max, reso))
            
            # Function values through 'pred_func' lambda            
            Z = np.zeros((reso*reso, D))
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
            
            plt.savefig(targetdir + str(dim1) + "_" + str(dim2) + ".pdf", bbox_inches='tight')
            plt.close()
