# Plotting functions
#
# m.mieskolainen@imperial.ac.uk, 2024

import matplotlib.pyplot as plt
import numpy as np
import awkward as ak
import torch
import gc
from pprint import pprint
import copy
from prettytable import PrettyTable
import multiprocessing
import time
from tqdm import tqdm
import pandas as pd

from iceplot import iceplot
from icefit import statstools
from icefit import cortools

from icenet.tools import aux
from icenet.tools import reweight
from icenet.tools import prints

# ------------------------------------------
from icenet import print
# ------------------------------------------


def binengine(bindef, x):
    """
    Binning processor function
    
    Args:
        bindef:  binning definition
        x:       data input array

    Examples:
        50                                                    (number of bins, integer)
        [1.0, 40.0, 50.0]                                     (list of explicit edges)
        {'nbin': 30, 'q': [0.0, 0.95], 'space': 'linear'}     (automatic with quantiles)
        {'nbin': 30, 'minmax': [2.0, 50.0], 'space': 'log10'} (automatic with boundaries)
    
    Returns:
        edges:   binning edges
    """
    
    if len(x) == 0:
        print(f'Input is zero-array, returning [0,0]')
        return np.array([0,0])

    def mylinspace(minval, maxval, nbin):

        if 'space' in bindef and bindef['space'] == 'log10':
            if minval <= 0:
                raise Exception(__name__ + f'.bin_processor: Trying log10 binning with minimum = {minval} <= 0')
            return np.logspace(np.log10(minval), np.log10(maxval), nbin + 1)
        else:
            return np.linspace(minval, maxval, nbin + 1)

    # Integer
    if   type(bindef) is int or type(bindef) is float:
        return np.linspace(np.min(x), np.max(x), int(bindef) + 1)

    # Bin edges given directly
    elif type(bindef) is list:
        return np.array(bindef)

    elif type(bindef) is np.ndarray:
        return bindef

    # Automatic with quantiles
    elif type(bindef) is dict and 'q' in bindef:
        return mylinspace(minval=np.percentile(x, 100*bindef['q'][0]), maxval=np.percentile(x, 100*bindef['q'][1]), nbin=bindef['nbin'])

    # Automatic with boundaries
    elif type(bindef) is dict and 'minmax' in bindef:
        return mylinspace(minval=bindef['minmax'][0], maxval=bindef['minmax'][1], nbin=bindef['nbin'])
    
    else:
        raise Exception(__name__ + f'.bin_processor: Unknown binning description in {bindef}')


def plot_selection(X, mask, ids, plotdir, label, varlist, density=True, library='np'):
    """
    Plot selection before / after type histograms against all chosen variables

    Args:
        X       : data array (N events x D dimensions)
        mask    : boolean selection indices (N)
        ids     : variable string array (D)
        plotdir : plotting directory
        label   : a string label
        varlist : a list of variables to be plotted (from ids)
        density : normalize all histograms to unit density
        library : 'np' or 'ak'
    """

    for var in tqdm(varlist):

        if var not in ids:
            continue

        # Histogram (autobinning)
        if library == 'np':
            before = np.asarray(X[:, ids.index(var)])
            after  = np.asarray(X[mask, ids.index(var)])
        elif library == 'ak':
            before = ak.to_numpy(X[:][var])
            after  = ak.to_numpy(X[mask][var])
        else:
            raise Exception(__name__ + f'.Unknown library: {library}')

        counts1, errs1, bins, cbins = iceplot.hist(before, bins=100,  density=density)
        counts2, errs2, bins, cbins = iceplot.hist(after,  bins=bins, density=density)
        
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
        targetdir = aux.makedir(f'{plotdir}/cuts/{label}')
        fig.savefig(f'{targetdir}/{var}.pdf', bbox_inches='tight')

        ax[0].set_yscale('log')
        fig.savefig(f'{targetdir}/{var}--log.pdf', bbox_inches='tight')
        
        # --------        
        fig.clf()
        plt.close()
        gc.collect()


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


def plot_train_evolution_multi(losses, trn_aucs, val_aucs, label, aspect=0.85,
                               yscale='linear', xscale='linear'):
    """ Training evolution plots.

    Args:
        losses:   loss values in a dictionary
        trn_aucs: training metrics
        val_aucs: validation metrics
    
    Returns:
        fig: figure handle
        ax:  figure axis
    """
    
    fig,ax = plt.subplots(1,2, figsize=(10, 7.5))
    
    # How many loss terms
    N_terms = 0
    for key in losses.keys():
        if 'eval' in key:
            N_terms += 1
    
    for key in losses.keys():
        if ('sum' in key) and (N_terms == 2):
            continue # Do not plot 'sum' if only sum made of one term
        
        ax[0].plot(losses[key], label=key)
    
    ax[0].set_xlabel('k (epoch)')
    ax[0].set_ylabel('loss')
    ax[0].legend(fontsize=6)
    ax[0].set_title(f'{label}', fontsize=10)
    
    plt.sca(ax[0])
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.yscale(yscale)
    plt.xscale(xscale)
    
    ax[1].plot(trn_aucs)
    ax[1].plot(val_aucs)
    ax[1].legend(['train','validation'], fontsize=8)
    ax[1].set_xlabel('k (epoch)')
    ax[1].set_ylabel('AUC')
    ax[1].grid(True)
    
    plt.sca(ax[1])
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.yscale(yscale)
    plt.xscale(xscale)
    
    ax[0].set_aspect(1.0/ax[0].get_data_ratio()*aspect)
    
    for i in [1]:
        ax[1].set_ylim([np.min([np.min(trn_aucs), np.min(val_aucs)]), 1.0])
        ax[1].set_aspect(1.0/ax[i].get_data_ratio()*aspect)

    return fig,ax


def binned_2D_AUC(y_pred, y, X_kin, ids_kin, X, ids, edges, label, weights=None, VAR:list=['trk_pt', 'trk_eta']):
    """
    Evaluate AUC per 2D-bin.
    
    Args:
        y_pred      :  MVA algorithm output
        y           :  Output (truth level target) data
        X_kin       :  Data
        ids_kin     :  Variables
        X           :  Data
        ids         :  Variables  
        edges       :  Edges of the A,B-space cells (2D array)
        label       :  Label of the classifier (string)
        weights     :  Sample weights
        VAR         :  Variable identifiers (two)
    
    Returns:
        fig,ax      :  Figure handle and axis
        met         :  Metrics object
    """

    edges_A = edges[0]
    edges_B = edges[1]

    AUC = np.zeros((len(edges_A)-1, len(edges_B)-1))

    if len(y_pred) != len(y):
        raise Exception(__name__ + f'.binned_2D_AUC: len(y_pred) = {len(y_pred)} != len(y) = {len(y)}')

    # Loop over bins
    for i in range(len(edges_A)-1):
        for j in range(len(edges_B)-1):

            range_A = [edges_A[i], edges_A[i+1]]
            range_B = [edges_B[j], edges_B[j+1]]

            # Indices
            try:
                ind = np.logical_and(aux.pick_ind(X_kin[:, ids_kin.index(VAR[0])], range_A),
                                 aux.pick_ind(X_kin[:, ids_kin.index(VAR[1])], range_B))
            except:
                try:
                    ind = np.logical_and(aux.pick_ind(X[:, ids.index(VAR[0])], range_A),
                                    aux.pick_ind(X[:, ids.index(VAR[1])], range_B))
                except:
                    raise Exception(__name__ + f'.binned_2D_AUC: Cannot find the variables {VAR}')
            
            string = f'{VAR[0]} = [{range_A[0]:10.3f},{range_A[1]:10.3f}), {VAR[1]} = [{range_B[0]:10.3f},{range_B[1]:10.3f})'
            
            if np.sum(ind) > 0: # Do we have any events in this cell
                
                # Evaluate metric
                if weights is not None:
                    met = aux.Metric(y_true=y[ind], y_pred=y_pred[ind], weights=weights[ind])
                else:
                    met = aux.Metric(y_true=y[ind], y_pred=y_pred[ind])

                print(f'{string} | AUC = {met.auc:.5f}')
                AUC[i,j] = met.auc

            else:
                print(f'{string} | No events found in this cell!')
    
    # Evaluate total performance
    met = aux.Metric(y_true=y, y_pred=y_pred, weights=weights)

    # Finally plot it
    fig,ax = plot_AUC_matrix(AUC=AUC, edges_A=edges_A, edges_B=edges_B)
    ax.set_title(f'{label} | AUC = {met.auc:.3f} (integrated)', fontsize=9)
    ax.set_xlabel(f"{VAR[0]}")
    ax.set_ylabel(f"{VAR[1]}")
    
    return fig, ax, met


def binned_1D_AUC(y_pred, y, X_kin, ids_kin, X, ids, edges, weights=None,
                  VAR:str='trk_pt', num_bootstrap=0):
    """
    Evaluate AUC & ROC per 1D-bin.
    
    Args:
        y_pred      :  MVA algorithm output
        y           :  Output (truth level target) data
        X_kin       :  Data
        ids_X_kin   :  Variables (strings)
        X           :  Data
        ids         :  Variables (strings)
        edges       :  Edges of the space cells
        weights     :  Sample weights
        VAR         :  Variable identifier to pick (one)
    
    Returns:
        fig,ax      :  Figure handle and axis
        met         :  Metrics object
    """

    AUC = np.zeros((len(edges)-1, 1))

    if len(y_pred) != len(y):
        raise Exception(__name__ + f'.binned_1D_AUC: len(y_pred) = {len(y_pred)} != len(y) = {len(y)}')

    METS   = []
    LABELS = []

    # Loop over bins
    for i in range(len(edges)-1):

            range_ = [edges[i], edges[i+1]]

            # Indices (either from X_kin or X)
            try:
                ind = aux.pick_ind(X_kin[:, ids_kin.index(VAR)], range_)
            except:
                try:
                    ind = aux.pick_ind(X[:, ids.index(VAR)], range_)
                except:
                    raise Exception(__name__ + f'.binned_1D_AUC: Cannot find the variable {VAR}')
            
            string = f'{VAR} = [{range_[0]:10.3f},{range_[1]:10.3f})'

            if np.sum(ind) > 0: # Do we have any events in this cell
                    
                # Evaluate metric
                if weights is not None:
                    met = aux.Metric(y_true=y[ind], y_pred=y_pred[ind], weights=weights[ind], num_bootstrap=num_bootstrap)
                else:
                    met = aux.Metric(y_true=y[ind], y_pred=y_pred[ind], num_bootstrap=num_bootstrap)

                AUC[i] = met.auc
                METS.append(met)
                print(f'{string} | AUC = {AUC[i]}')
            else:
                METS.append(None)
                print(f'{string} | No events found in this cell!')

            LABELS.append(f'{VAR}$ \\in [{range_[0]:.1f},{range_[1]:.1f})$')

    return METS, LABELS


def density_MVA_wclass(y_pred, y, label, weights=None, class_ids=None, edges=80, path='', **kwargs):
    """
    Evaluate MVA output (1D) density per class.
    
    Args:
        y_pred     :  MVA algorithm output
        y          :  Output (truth level target) data
        label      :  Label of the MVA model (string)
        weights    :  Sample weights
        class_ids  :  Class IDs to plot
        hist_edges :  Histogram edges list (or number of bins, as an alternative)
    
    Returns:
        Plot pdf saved directly
    """
    
    if class_ids is None:
        class_ids = np.unique(y.astype(int))
    
    # Make sure it is 1-dim array of length N (not N x num classes)
    if (weights is not None) and len(weights.shape) > 1:
        weights = np.sum(weights, axis=1)
    
    if weights is not None:
        classlegs = [f'$\\mathcal{{C}} = {k}$, $N={np.sum(y == k)}$ ($\\Sigma_w = {np.sum(weights[y == k]):0.1f}$)' for k in class_ids]
    else:
        classlegs = [f'$\\mathcal{{C}} = {k}$, $N={np.sum(y == k)}$ (no weights)' for k in class_ids]
    
    # Handle logits vs probabilities
    THRESH = 1E-5
    if np.min(y_pred) < (-THRESH) or np.max(y_pred) > (1.0 + THRESH):
        logit = copy.deepcopy(y_pred)
        prob  = aux.sigmoid(logit)
    else:
        prob  = copy.deepcopy(y_pred)
        logit = aux.inverse_sigmoid(prob)    
    
    # Plot both
    for mode in ['logit', 'prob']:
        
        # Over classes
        fig,ax = plt.subplots()
        
        for k in class_ids:
            ind = (y == k)

            if np.sum(ind) == 0:
                print(f'No samples for class {k} -- continue')
                continue
            
            w = weights[ind] if weights is not None else None
            
            if mode == 'logit':
                x = logit[ind]
            else:
                x = prob[ind]
            
            hI, bins, patches = plt.hist(x=x, bins=binengine(bindef=edges, x=x), weights=w,
                density = True, histtype = 'step', fill = False, linewidth = 2, label = 'inverse')
        
        plt.legend(classlegs, loc='upper center', fontsize=7)
        plt.xlabel(f'MVA output $f(\\mathbf{{x}})$ [{mode}]')
        plt.ylabel('density')
        plt.title(f'{label}', fontsize=9)
        
        for scale in ['linear', 'log']:
            try:
                ax.set_yscale(scale)
                outputdir = aux.makedir(f'{path}')
                savepath  = f'{outputdir}/MVA-output__{mode}__{scale}.pdf'
                plt.savefig(savepath, bbox_inches='tight')
                print(f'Saved: "{savepath}"', 'green')
            
            except Exception as e:
                print(e, 'red')
                print(f'Plotting failed for {label} -- continue', 'red')
                continue
        
        # --------        
        fig.clf()
        plt.close(fig)
        gc.collect()


def plot_correlation_comparison(corr_mstats, targetdir, xlim=None):
    """
    Plot collected correlation metrics from density_COR_wclass()
    
    Args:
        corr_mstats: statistics dictionary
        targetdir:   output directory
        xlim:        plot limits dictionary per class
    Returns:
        plots saved to a directory
    """
    
    print(f'Plotting correlation comparisons ...')

    class_ids = {}
    
    ## Find all variables
    all_var   = []
    for model in corr_mstats.keys():
        for category in corr_mstats[model].keys():
            for c in corr_mstats[model][category].keys():
                class_ids = corr_mstats[model][category].keys() # This is overwritten (all same)
                for var in corr_mstats[model][category][c].keys():
                    all_var.append(var)

    all_var = set(all_var) # Reduce to a set (values appear only once)

    ## Over all variables
    for var in all_var:

        # Over classes
        for c in class_ids:

            # Over different statistical metrics
            for stats in ['pearson', 'abs_pearson', 'disco', 'MI']:

                fig,ax = plt.subplots()
                
                # Per model
                for model in corr_mstats.keys():

                    categories = list(corr_mstats[model].keys())

                    values = np.nan * np.ones(len(categories))
                    lower  = np.nan * np.ones(len(categories))
                    upper  = np.nan * np.ones(len(categories))

                    ## Over each powerset category
                    for i in range(len(categories)):
                        x = corr_mstats[model][categories[i]][c]

                        if x is not {}: # We have some stats
                            values[i] = x[var][f'{stats}']
                            lower[i]  = x[var][f'{stats}_CI'][0]
                            upper[i]  = x[var][f'{stats}_CI'][1]

                    error = (upper - lower)/2 # Simple symmetric errors
                    #lower = np.abs(values - np.array(lower))
                    #upper = np.abs(values - np.array(upper))
                    #asymmetric_error = np.array(list(zip(lower, upper))).T

                    # Vertical line at zero
                    plt.plot(np.zeros(len(values)), np.arange(len(values)), color=np.ones(3)*0.5, label=None)

                    ## Plot horizontal plot i.e. values +- (lower, upper) on x-axis, category on y-axis
                    legend_label = f'{model} [{np.median(values):0.3f}]'
                    plt.errorbar(values, np.arange(len(values)), xerr=error,
                        fmt='s', capsize=5.0, label=legend_label)

                    title = f'$\\mathcal{{C}} = {c}$'
                    plt.title(title)
                    plt.xlabel(f'{stats}$_{{XY}}$ (MVA score, {var}) (68CL)')
                    
                    if xlim is not None:
                        ax.set_xlim(xlim[stats][c])
                    ax.set_yticks(np.arange(len(values)))
                    ax.set_yticklabels(categories)
                
                ax.invert_yaxis()    
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                filename = aux.makedir(targetdir + f'/COR/')
                plt.savefig(filename + f'var-{var}--stats-{stats}--class-{c}.pdf',
                    bbox_inches='tight')
                plt.close()


def density_COR_wclass(y_pred, y, X, ids, label, \
    weights=None, class_ids=None, edges=[[50], [50]], density=True,
    path='', cmap='Oranges', **kwargs):
    
    """
    Evaluate the 2D-density of the MVA algorithm output vs other variables per class.
    
    Args:
        y_pred      :  MVA algorithm output
        y           :  Output (truth level target) data
        X           :  Variables to be plotted
        ids         :  Identifiers of the variables in X
        label       :  Label of the MVA model (string)
        weights     :  Sample weights
        class__ids  :  Class ids to plot
        hist_edges  :  Histogram edges list (or number of bins, as an alternative) (2D)
        density     :  Normalize to density
        path        :  Save path
        cmap        :  Color map
        
    Returns:
        correlation values in a dictionary (per variable, per class)
        plots are saved directly
    """

    if class_ids is None:
        class_ids = np.unique(y.astype(int))
    
    # Make sure it is 1-dim array of length N (not N x num classes)
    if (weights is not None) and len(weights.shape) > 1:
        weights = np.sum(weights, axis=1)
    
    if weights is not None:
        classlegs = [f'$\\mathcal{{C}} = {k}$, $N={np.sum(y == k)}$ (weighted {np.sum(weights[y == k]):0.1f})' for k in class_ids]
    else:
        classlegs = [f'$\\mathcal{{C}} = {k}$, $N={np.sum(y == k)}$ (no weights)' for k in class_ids]
    
    output = {}

    # Over classes
    for k in class_ids:
        
        output[k] = {}
        ind = (y == k)
        w = weights[ind] if weights is not None else None
        
        if np.sum(ind) == 0:
            print(f'No samples for class {k} -- continue')
            continue

        # Loop over variables
        for var in ids:

            xx   = y_pred[ind]
            yy   = X[ind, ids.index(var)]
            
            bins = [binengine(bindef=edges[0], x=xx), binengine(bindef=edges[1], x=yy)]

            # Apply histogram bounds
            box = np.logical_and(np.logical_and(xx > bins[0][0], xx < bins[0][-1]),
                                 np.logical_and(yy > bins[1][0], yy < bins[1][-1]))
            xx  = xx[box]
            yy  = yy[box]
            ww  = w[box]
            
            # Pearson correlation coefficient
            cc,cc_CI,_         = cortools.pearson_corr(x=xx, y=yy, weights=ww)
            
            # Absolute Pearson correlation coefficient
            cc_abs,cc_abs_CI,_ = cortools.pearson_corr(x=xx, y=yy, weights=ww, return_abs=True)
            
            # Distance correlation
            disco,disco_CI     = cortools.distance_corr(x=xx, y=yy, weights=ww)
            
            # Histogram based MI
            MI,MI_CI = cortools.mutual_information(x=xx, y=yy, weights=ww, automethod='Scott2D')
            
            # Neural Mutual Information [cannot use, unreliable to compute for small samples]
            #from icefit import mine
            #MI,MI_err = mine.estimate(X=xx, Z=yy, weights=ww)
            #MI_CI = np.array([MI-MI_err, MI+MI_err])
            
            # Save output
            output[k][var] = {}

            output[k][var]['pearson']    = cc
            output[k][var]['pearson_CI'] = cc_CI
            
            output[k][var]['abs_pearson']    = cc_abs
            output[k][var]['abs_pearson_CI'] = cc_abs_CI

            output[k][var]['disco']    = disco
            output[k][var]['disco_CI'] = disco_CI
            
            output[k][var]['MI']       = MI
            output[k][var]['MI_CI']    = MI_CI

            for scale in ['linear', 'log']: 

                fig,ax    = plt.subplots()
                outputdir = aux.makedir(f'{path}')
                savepath  = f'{outputdir}/var-{var}--class-{k}--{scale}.pdf'
                
                try:
                    if scale == 'log':
                        import matplotlib as mpl
                        h2,xedges,yedges,im = plt.hist2d(x=xx, y=yy, bins=bins, weights=ww, density=density, norm=mpl.colors.LogNorm(), cmap=plt.get_cmap(cmap))
                    else:
                        h2,xedges,yedges,im = plt.hist2d(x=xx, y=yy, bins=bins, weights=ww, density=density, cmap=plt.get_cmap(cmap))
                    
                    fig.colorbar(im)
                    plt.xlabel(f'MVA output $f(\\mathbf{{x}})$')
                    plt.ylabel(f'{var}')
                    rho_value = f'$\\rho_{{XY}} = {cc:0.2f}_{{-{cc-cc_CI[0]:0.2f}}}^{{+{cc_CI[1]-cc:0.2f}}}$'
                    MI_value  = f'$\\mathcal{{I}}_{{XY}} = ({MI_CI[0]:0.3f}, {MI_CI[1]:0.3f})$'
                    
                    plt.title(f'{label} | $\\mathcal{{C}} = {k}$ | {rho_value} | {MI_value}', fontsize=9)
                    # -----

                    plt.savefig(savepath, bbox_inches='tight')
                    print(f'Saved: "{savepath}"', 'green')
                    
                    # -----                    
                    fig.clf()
                    plt.close()
                    gc.collect()
                
                except: # Matplotlib LogNorm() can be buggy
                    print(f'Failed to produce plot {savepath}')
    pprint(output)

    return output


def density_COR(y_pred, X, ids, label, weights=None, hist_edges=[[50], [50]], path='', cmap='Oranges'):
    """
    Evaluate the 2D-density of the MVA algorithm output vs other variables.
    
    Args:
        y_pred      :  MVA algorithm output
        X           :  Variables to be plotted
        ids         :  Identifiers of the variables in X
        label       :  Label of the MVA model (string)
        weights     :  Sample weights
        hist_edges  :  Histogram edges list (or number of bins, as an alternative) (2D)
        path        :  Save path
        cmap        :  Color map
    
    Returns:
        Plot pdf saved directly
    """

    # Make sure it is 1-dim array of length N (not N x num classes)
    if (weights is not None) and len(weights.shape) > 1:
        weights = np.sum(weights, axis=1)
    
    # Loop over variables
    for var in ids:
        
        fig,ax = plt.subplots()

        # Plot 2D
        xx   = y_pred
        yy   = X[:, ids.index(var)]
        
        bins = [binengine(bindef=hist_edges[0], x=xx), binengine(bindef=hist_edges[1], x=yy)]
        h2,xedges,yedges,im = plt.hist2d(x=xx, y=yy, bins=bins, weights=weights, cmap=plt.get_cmap(cmap))
        
        fig.colorbar(im)
        plt.xlabel(f'MVA output $f(\\mathbf{{x}})$')
        plt.ylabel(f'{var}')
        plt.title(f'{label}', fontsize=10)
        
        # -----

        outputdir = aux.makedir(f'{path}/{label}')
        savepath = f'{outputdir}/var-{var}.pdf'
        plt.savefig(savepath, bbox_inches='tight')
        print(f'Saved: "{savepath}"', 'green')

        # -----                    
        fig.clf()
        plt.close()
        gc.collect()
        

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
                text = ax.text(j, i, f'{X[i,j]:.0f}', ha="center", va="center", color=color)
            if (decimals == 1):
                text = ax.text(j, i, f'{X[i,j]:.1f}', ha="center", va="center", color=color)
            if (decimals == 2):
                text = ax.text(j, i, f'{X[i,j]:.2f}', ha="center", va="center", color=color)
    return ax


def plot_AUC_matrix(AUC, edges_A, edges_B):
    """ Plot AUC matrix.

    Args:
        AUC:      AUC-ROC matrix
        edges_A:  Histogram edges of variable A
        edges_B:  Histogram edges of variable B
    
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


def multiprocess_plot_wrapper(p):
    """
    Multiprocessing wrapper
    """
    i = p['i']
    
    # Exclude special values
    ind = np.ones(len(p['X']), dtype=bool)
    for k in range(len(p['exclude_vals'])):
        ind = np.logical_and(ind, (p['X'][:,i] != p['exclude_vals'][k]))

    plotvar(x                = p['X'][ind,i],
            y                = p['y'][ind],
            weights          = p['weights'][ind],
            var              = p['ids'][i],
            nbins            = p['nbins'],
            percentile_range = p['percentile_range'],
            title            = p['title'],
            targetdir        = p['targetdir'],
            plot_unweighted  = p['plot_unweighted'])


def plotvars(X, y, ids, weights, nbins=70, percentile_range=[0.5, 99.5],
             exclude_vals=[None], plot_unweighted=True, title = '', targetdir = '.',
             num_cpus: int=0):
    """
    Plot all variables.
    """
    
    print(f'Creating plots | exclude_vals = {exclude_vals} ...')
    
    # Chunked loop [parallel processing]
    param = {
        'i':                None,
        'X':                None,
        'y':                None,
        'weights':          None,
        'ids':              ids,
        'nbins':            nbins,
        'percentile_range': percentile_range,
        'exclude_vals':     exclude_vals,
        'plot_unweighted':  plot_unweighted,
        'title':            title,
        'targetdir':        targetdir
    }
    
    # Create parameter dictionaries for each process
    paramlist = []
    for i in range(X.shape[1]):
        paramlist.append(copy.deepcopy(param)) 
        paramlist[-1]['i'] = i 
        paramlist[-1]['X'] = X # big numpy arrays should share the memory (i.e. no copy of X)
        paramlist[-1]['y'] = y
        paramlist[-1]['weights'] = weights
    
    # Start multiprocessing
    print(f'Multiprocessing {X.shape[1]} plots', 'yellow')
    
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() // 2 if num_cpus == 0 else num_cpus)
    tic  = time.time()
    pool.map(multiprocess_plot_wrapper, paramlist)
    pool.close() # no more tasks
    pool.join()  # wrap up current tasks
    
    toc  = time.time()
    print(f'Took {toc-tic:0.2f} sec')


def plotvar(x, y, var, weights, nbins=70, percentile_range=[0.5, 99.5],
            plot_unweighted=True, title='', targetdir='.'):
    """
    Plot a single variable.
    """
    try:
        binrange = (np.percentile(x, percentile_range[0]), np.percentile(x, percentile_range[1]))
        
        fig, axs = plot_reweight_result(X=x, y=y, nbins=nbins, binrange=binrange, weights=weights,
                                        title=title, xlabel=var, plot_unweighted=plot_unweighted)
        plt.savefig(f'{targetdir}/var-{var}.pdf', bbox_inches='tight')    
        fig.clf()
        plt.close(fig)
        gc.collect()
    
    except Exception as e:
        print(e, 'red')
        print(f'Problem plotting variable "{var}"', 'red')


def plot_reweight_result(X, y, nbins, binrange, weights, title = '', xlabel = 'x', linewidth=1.5,
                         plot_unweighted=True):
    """
    Here plot pure event counts
    so we see that also integrated class fractions are equalized (or not) after weighting!
    """

    fig,ax    = plt.subplots(1, 2, figsize = (10, 4.25))
    class_ids = np.unique(y.astype(int))
    legends   = []
    
    min_x =  1e30
    max_x = -1e30
    
    # Loop over classes
    for c in class_ids:

        ind = (y == c)
        
        if np.sum(ind) == 0:
            print(f'No samples for class {c} -- continue')
            continue
        
        # Compute histograms with numpy (we use nbins and range() for speed)
        if plot_unweighted:
            counts, edges = np.histogram(X[ind], bins=nbins, range=binrange, weights=None)
        
        counts_w, edges = np.histogram(X[ind], bins=nbins, range=binrange, weights=weights[ind])
        mu, std = aux.weighted_avg_and_std(values=X[ind], weights=weights[ind])
        
        min_x = edges[0]  if edges[0]  < min_x else min_x
        max_x = edges[-1] if edges[-1] > max_x else max_x
        
        # Linear and log scale scale (left and right plots)
        for i in range(2):
            
            plt.sca(ax[i])
            if plot_unweighted:
                plt.stairs(counts, edges, fill=False, linewidth = linewidth+0.75, linestyle='--') # bigger linewidth first
            plt.stairs(counts_w, edges, fill=False, linewidth = linewidth, linestyle='-')
            
            if i == 0:
                if plot_unweighted:
                    legends.append(f'$\\mathcal{{C}} = {c}$ (unweighted)')
                legends.append(f'$\\mathcal{{C}} = {c}$ [$\\mu={mu:0.2f}, \\sigma={std:0.2f}$]')
    
    ax[0].set_ylabel('[weighted] counts')
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylim([0,None])
    ax[0].set_xlim([min_x, max_x])
    
    ax[1].set_xlabel(xlabel)
    ax[1].set_title(title, fontsize=10)
    ax[1].legend(legends, fontsize=8)
    ax[1].set_yscale('log')
    plt.tight_layout()
    
    return fig,ax


def plot_correlations(X, ids, weights=None, y=None, round_threshold=0.0, targetdir=None, colorbar = False):
    """
    Plot a cross-correlation matrix of vector data
    
    Args:
        X:                Data matrix (N x D)
        ids:              Variable names (list of length D)
        weights:          Event weights
        y:                Class labels per event (list of length N)
        round_threshold:  Correlation matrix |C| < threshold to set matrix elements to zero
        targetdir:        Output plot directory
        colorbar:         Colorbar on the plot
    
    Returns:
        figs, axs:        Figures, axes (per class)
    """
    N = X.shape[0]
    
    if y is None:
        y = np.zeros(N)
        class_ids = [0]
    else:
        class_ids = np.unique(y.astype(int))

    figs = {}
    axs  = {}
    
    for k in class_ids:

        label = f'all' if (len(class_ids) == 1) else f'class_{k}'

        # Compute correlation matrix
        tic = time.time()
        w = weights[y==k] if weights is not None else None
        C = statstools.correlation_matrix(X=X[y==k,:], weights=w)
        C[np.abs(C) < round_threshold] = np.nan
        C *= 100
        toc = time.time()
        print(f'Computing C-matrix (took {toc-tic:0.2f} sec)')

        # Compute suitable figsize
        size = np.ceil(C.shape[0] / 3)
        
        # Plot it
        figs[label], axs[label] = plt.subplots(1,1, figsize=(size,size))

        axs[label].imshow(C)
        axs[label] = annotate_heatmap(X = C, ax = axs[label], xlabels = ids,
            ylabels = ids, decimals = 0, x_rot = 90, y_rot = 0, color = "w")
        axs[label].set_title(f'{label}: linear correlation $\\in$ [-100,100]', fontsize=10)
        
        if colorbar:
            cb = plt.colorbar()

        if targetdir is not None:
            tic = time.time()
            if X.shape[1] <= 100: # Not too many dimensions --> pdf OK
                fname = targetdir + f'{label}-correlation-matrix.pdf'
                plt.savefig(fname=fname, pad_inches=0.2, bbox_inches='tight')
            else:
                fname = targetdir + f'{label}-correlation-matrix.png'
                plt.savefig(fname=fname, pad_inches=0.2, dpi=300, bbox_inches='tight')
            toc = time.time()
            print(f'Saved: "{fname} (took {toc-tic:0.2f} sec)"')

    return figs, axs


def draw_error_band(ax, x, y, x_err, y_err, **kwargs):
    
    """
    Calculate normals via centered finite differences (except the first point
    which uses a forward difference and the last point which uses a backward
    difference).

    https://matplotlib.org/stable/gallery/lines_bars_and_markers/curve_error_band.html
    """
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch

    dx = np.concatenate([[x[1] - x[0]], x[2:] - x[:-2], [x[-1] - x[-2]]])
    dy = np.concatenate([[y[1] - y[0]], y[2:] - y[:-2], [y[-1] - y[-2]]])
    l = np.hypot(dx, dy)
    nx = dy / l
    ny = -dx / l

    # End points of errors
    xp = x + nx * x_err
    yp = y + ny * y_err
    xn = x - nx * x_err
    yn = y - ny * y_err

    vertices = np.block([[xp, xn[::-1]],
                         [yp, yn[::-1]]]).T
    codes    = np.full(len(vertices), Path.LINETO)
    codes[0] = codes[len(xp)] = Path.MOVETO
    path     = Path(vertices, codes)
    ax.add_patch(PathPatch(path, **kwargs))


def ROC_plot(metrics, labels, title = '', plot_thresholds=True, \
    thr_points_signal = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95], \
    filename = 'ROC', legend_fontsize=7, xmin=1.0e-4, alpha=0.32):
    """
    Receiver Operating Characteristics i.e. False positive (x) vs True positive (y)
    Args:
        metrics:
        labels:
        title:
        plot_thresholds:
        thr_points_signal: 
        filename:
        legend_fontsize:
        xmin:
        alpha:
    """

    for k in [0,1]: # linear & log
        
        fig,ax = plt.subplots()
        xx     = np.logspace(-5, 0, 100)
        plt.plot(xx, xx, linestyle='--', color='black', linewidth=1) # ROC diagonal

        for i in range(len(metrics)):

            if         i < 10:
                linestyle = '-'
            elif 10 <= i < 20:
                linestyle = '--'
            else:
                linestyle = ':'
            marker = 'None'
            
            if metrics[i] is None:
                print(f'metrics[{i}] ({labels[i]}) is None, continue without')
                continue
            
            fpr        = metrics[i].fpr
            tpr        = metrics[i].tpr
            thresholds = metrics[i].thresholds
            
            if metrics[i].tpr_bootstrap is not None:

                q = [100*(alpha/2), 100*(1-alpha/2)]
                
                # Percentile bootstrap based uncertainties
                tpr_CI = cortools.percentile_per_dim(x=metrics[i].tpr_bootstrap, q=q)
                tpr_lo = tpr_CI[0]
                tpr_hi = tpr_CI[1]
                
                fpr_CI = cortools.percentile_per_dim(x=metrics[i].fpr_bootstrap, q=q)
                fpr_lo = fpr_CI[0]
                fpr_hi = fpr_CI[1]
            
            # Autodetect a ROC point triangle (instead of a curve)
            if len(np.unique(fpr)) == 3:
                tpr = tpr[1:-1] # Remove first and last
                fpr = fpr[1:-1] # Remove first and last
                
                linestyle = "None"
                marker    = 'o'
            
            """
            # ROC-curve
            elif not (isinstance(fpr, int) or isinstance(fpr, float)):
                fpr    = fpr[1:] # Remove always the first element for log-plot reasons
                tpr    = tpr[1:]
                if metrics[i].tpr_bootstrap is not None:
                    tpr_lo = tpr_lo[1:]
                    tpr_hi = tpr_hi[1:]
                    fpr_lo = fpr_lo[1:]
                    fpr_hi = fpr_hi[1:]
            """
            
            ## Plot it
            plt.plot(fpr, tpr, drawstyle='steps-mid', color=f'C{i}', linestyle=linestyle, marker=marker, label = f'{labels[i]}: AUC = {metrics[i].auc:.4f}')
            
            # Uncertainty band
            if marker == 'None' and (metrics[i].tpr_bootstrap is not None):
                
                plt.fill_between(fpr,  tpr_lo, tpr_hi, step='mid', alpha=0.2, color=f'C{i}', edgecolor='none') # vertical
                plt.fill_betweenx(tpr, fpr_lo, fpr_hi, step='mid', alpha=0.2, color=f'C{i}', edgecolor='none') # horizontal

                # draw_error_band(ax=ax, x=fpr, y=tpr, \
                #   x_err=np.std(metrics[i].fpr_bootstrap, axis=0)[1:], \
                #   y_err=np.std(metrics[i].tpr_bootstrap, axis=0)[1:], \
                #   facecolor=f'C{i}', edgecolor="none", alpha=0.2)

            # Plot corresponding threshold points
            if plot_thresholds:
                
                try:
                    for eff in thr_points_signal:
                        index = np.argmin(np.abs(tpr - eff))
                        if fpr[index] >= xmin and fpr[index] <= 1.0:
                            plt.plot(fpr[index], tpr[index], '.', color=f'C{i}')
                            t = plt.text(x=fpr[index], y=tpr[index], s=f'{thresholds[index]:0.4g}', fontsize=5, color=f'C{i}')
                            t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                except: # If failed
                    True
        
        ax.set_xlabel('False Positive Rate $\\alpha$ (background efficiency)')
        ax.set_ylabel('True Positive Rate $1-\\beta$ (signal efficiency)')
        ax.set_title(title, fontsize=10)
        
        if k == 0: # Linear-Linear
            
            # Legend
            if len(metrics) > 12: # Put outside the figure
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=legend_fontsize)
            else:
                plt.legend(loc='lower right', fontsize=legend_fontsize)
            
            plt.ylim(0.0, 1.0)
            plt.xlim(0.0, 1.0)
            plt.locator_params(axis="x", nbins=11)
            plt.locator_params(axis="y", nbins=11)

            ax.set_aspect(1.0 / ax.get_data_ratio() * 1.0)
            plt.savefig(filename + '.pdf', bbox_inches='tight')
            print(f'Saved: ' + filename + '.pdf','green')
        
        if k == 1: # Log-Linear
            
            # Legend
            if len(metrics) > 12: # Put outside the figure
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=legend_fontsize)
            else:
                plt.legend(fontsize=legend_fontsize) # Automatic positioning
            
            plt.ylim(0.0, 1.0)
            plt.xlim(xmin, 1.0)
            plt.locator_params(axis="x", nbins=int(-np.log10(xmin) + 1))
            plt.locator_params(axis="y", nbins=11)

            plt.gca().set_xscale('log')
            ax.set_aspect(1.0 / ax.get_data_ratio() * 0.75)
            plt.savefig(filename + '--log.pdf', bbox_inches='tight')

        plt.close()


def MVA_plot(metrics, labels, title='', filename='MVA', density=True, legend_fontsize=7) :
    """
    MVA output plots
    """

    # Check input
    for i in range(len(metrics)):

        if metrics[i] is None:
            print(f'Error: metrics[{i}] ({labels[i]}) is None (check per class statistics), return -1')
            return -1
        else:
            class_ids = metrics[i].class_ids
        
        if len(metrics[i].mva_hist) != len(class_ids):
            print(f'Error: num_classes != len(metrics[{i}].mva_hist) (check per class statistics), return -1')
            return -1
    
    for k in [0,1]: # linear & log
        
        fig,ax = plt.subplots(1, len(class_ids), figsize=(7*len(class_ids), 5))

        # Loop over classes
        for c in class_ids:
            
            # Loop over cells
            N      = len(metrics[0].mva_hist[c]) # Number of bins
            counts = np.zeros((N, len(metrics)))
            cbins  = np.zeros((N, len(metrics)))
            
            for i in range(len(metrics)):

                if metrics[i] is None:
                    print(f'metrics[{i}] is None, continue without')
                    continue

                if len(metrics[i].mva_bins) == 0:
                    print(f'len(metrics[{i}].mva_bins) == 0, continue without')
                    continue

                bins       = metrics[i].mva_bins
                cbins[:,i] = (bins[0:-1] + bins[1:]) / 2

                if metrics[i].mva_hist[c] != []:
                    counts[:,i] = metrics[i].mva_hist[c]

            plt.sca(ax[c])
            plt.hist(x=cbins, bins=bins, weights=counts, histtype='bar', stacked=True, \
                density=density, label=labels, linewidth=2.0)

            # Adjust
            plt.sca(ax[c])
            plt.legend(fontsize=legend_fontsize, loc='upper center')
            ax[c].set_xlabel('MVA output $f(\\mathbf{x})$')
            ax[c].set_ylabel('counts' if not density else 'density')
            ax[c].set_title(f'{title} | class = {c}', fontsize=10)
        
        if k == 0:
            #plt.ylim(0.0, 1.0)
            #plt.xlim(0.0, 1.0)
            #ax.set_aspect(1.0/ax.get_data_ratio() * 1.0)
            plt.savefig(filename + '.pdf', bbox_inches='tight')
            print(f'Saved: ' + filename + '.pdf','green')
        
        if k == 1:
            #plt.ylim(0.0, 1.0)
            #plt.xlim(1e-4, 1.0)
            for i in range(len(class_ids)):
                plt.sca(ax[i])
                plt.gca().set_yscale('log')
                #ax.set_aspect(1.0/ax.get_data_ratio() * 0.75)
            plt.savefig(filename + '--log.pdf', bbox_inches='tight')

        # --------        
        fig.clf()
        plt.close()
        gc.collect()


def plot_contour_grid(pred_func, X, y, ids, targetdir = '.', transform = 'numpy', reso=50, npoints=400):
    """
    Classifier decision contour evaluated pairwise for each dimension,
    other dimensions evaluated at zero=(0,0,...0) (thus z-normalized with 0-mean is suitable)
    
    Args:
        pred_func:  prediction function handle
        X:          input matrix
        y:          class targets
        ids:        variable label strings
        targetdir:  output directory
        transform:  'numpy', 'torch'
        reso:       evaluation resolution
        npoints:    number of points to draw
    """
    
    print(f'Evaluating, X.shape = {X.shape}...')
    MAXP = min(npoints, X.shape[0])
    D    = X.shape[1]
    pad  = 0.5

    for dim1 in tqdm(range(D)) :
        x_min, x_max = X[:, dim1].min() - pad, X[:, dim1].max() + pad
        for dim2 in range(D) :
            if dim2 <= dim1 :
                continue
            
            # (x,y)-plane limits
            y_min, y_max = X[:, dim2].min() - pad, X[:, dim2].max() + pad

            # Grid points
            PX,PY = np.meshgrid(np.linspace(x_min, x_max, reso), np.linspace(y_min, y_max, reso))
            
            # -------------------------------------
            ## Evaluate function values

            XX = np.zeros((reso*reso, D))
            XX[:, dim1] = PX.ravel()
            XX[:, dim2] = PY.ravel()

            if   transform == 'torch':
                Z = pred_func(torch.from_numpy(XX).type(torch.FloatTensor))
            elif transform == 'numpy':
                Z = pred_func(XX)
            else:
                raise Exception(__name__ + f'.plot_decision_contour: Unknown matrix type = {transform}')

            Z = Z.reshape(PX.shape)
            
            # --------------------------------------
            ## Plot

            fig, axs = plt.subplots()

            # Contour
            cs = plt.contourf(PX, PY, Z, cmap = plt.cm.Spectral)

            # Events as dots
            plt.scatter(X[0:MAXP, dim1], X[0:MAXP, dim2], c = y[0:MAXP], cmap = plt.cm.binary)

            plt.xlabel(f'x[{dim1}] {ids[dim1]}')
            plt.ylabel(f'x[{dim2}] {ids[dim2]}')
            plt.colorbar(cs, ticks = np.linspace(0.0, 1.0, 11))
            
            filename = targetdir + f'{dim1}-{dim2}.pdf'
            plt.savefig(filename, bbox_inches='tight')
            print(f'Saved: ' + filename,'green')
            
            # --------        
            fig.clf()
            plt.close()
            gc.collect()


def plot_xgb_importance(model, tick_label, importance_type='gain', label=None, sort=True, default_names=False):
    """
    Plot XGBoost model feature importance
    
    Args:
        model:             xgboost model object
        dim:               feature space dimension
        tick_label:        feature names
        importance_type:   type of importance metric ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
        default_names:     True for xgboost default, else set False (uses tick_label)
    Returns
        fig, ax
    """
    fscores = model.get_score(importance_type=importance_type)
    
    dim    = len(tick_label) 
    xx     = np.arange(dim)
    yy     = np.zeros(dim)
    labels = []

    # Try, Except needed because xgb does Not return (always) for all of them
    for i in range(dim):
        try:
            if default_names:
                yy[i] = fscores[f'f{i}']       # XGBoost default feature name 'f{i}'
            else:
                yy[i] = fscores[tick_label[i]] # Definite names
        except:
            yy[i] = 0.0
        
        labels.append(f'{tick_label[i]} ({yy[i]:0.1f}) [{i}]')
    
    # Sort them
    if sort:
        s_ind  = np.array(np.argsort(yy), dtype=int)
        yy     = yy[s_ind]
        labels = [labels[i] for i in s_ind]
    
    # Plot
    fig,ax = plt.subplots(figsize=(0.5 * (np.ceil(dim/6) + 2), np.ceil(dim/6) + 2))
    plt.barh(xx, yy, align='center', height=0.5, tick_label=labels)
    plt.xlabel(f'Score ({importance_type})')
    plt.title(f'[{label}]')
    
    return fig, ax


def table_writer(filename, label, sublabel, tau, chi2_table, print_to_screen=False):
    """
    Helper function to write a chi2 table to a file
    """
    with open(filename, "a") as text_file:
        
        text_file.write(f'category: {sublabel} | AI model: {label} | tau: {tau:0.2f}')
        text_file.write('\n')
        text_file.write(chi2_table.get_string())
        text_file.write('\n')
        text_file.write('\n')
        if print_to_screen:
            print(chi2_table, 'magenta')
            print('')


def plot_AIRW(X, y, ids, weights, y_pred, pick_ind,
              label, sublabel, param, tau=1.0, targetdir=None, num_cpus=0):
    """
    Plot AI based reweighting results
    """
    
    print(f'label = {label} | sublabel = {sublabel} | tau = {tau}', 'green')
    
    dir       = aux.makedir(f'{targetdir}/{label}/{sublabel}')
    local_dir = aux.makedir(dir + f'/tau_{tau:0.2f}')
    
    ## Print stats
    output_file = f'{local_dir}/stats_AIRW_weight_flow.log'
    prints.print_weights(weights=weights, y=y, output_file=output_file,
        header='Step 1. Input event weights', write_mode='w')
    
    # ---------------------------------------------------
    ## 1. Transform model output scores to AI weights
    
    EPS   = param['EPS']
    maxW  = param['maxW']
    C0    = param['C0']
    C1    = param['C1']
    RN_ID = param['renorm_origin']
    mode  = param['transform_mode']
    
    # Handle logits vs probabilities
    min_y_pred, max_y_pred = np.min(y_pred), np.max(y_pred)
    
    THRESH = 1E-5
    if min_y_pred < (-THRESH) or max_y_pred > (1.0 + THRESH):
        print(f'Detected raw logit output [{min_y_pred:0.4f}, {max_y_pred:0.4f}] from the model')
        logits = y_pred[y == C0]
        probs  = aux.sigmoid(logits)
        print(f'Corresponding probability output [{np.min(probs):0.4f}, {np.max(probs):0.4f}]')
    else:
        print(f'Detected probability output [{min_y_pred:0.4f}, {max_y_pred:0.4f}] from the model')
        logits = aux.inverse_sigmoid(p = y_pred[y == C0], EPS=EPS)
        print(f'Corresponding logit output [{np.min(logits):0.4f}, {np.max(logits):0.4f}]')
    
    # 1. Apply temperature scaling
    logits /= tau
    
    # 2. Get weights after the re-weighting transform
    AIw0 = reweight.rw_transform_with_logits(logits=logits, mode=mode)
    
    ## Print stats
    prints.print_weights(weights=AIw0, y=np.zeros(len(AIw0)), output_file=output_file,
        header='Step 2. AI weights [raw]', write_mode='a')
    
    # 3. Cut-off regularize anomalous high weights before event weights
    AIw0 = np.clip(AIw0, 0.0, maxW)
    
    ## Print stats
    prints.print_weights(weights=AIw0, y=np.zeros(len(AIw0)), output_file=output_file,
        header=f'Step 3. AI weights [cutoff regularized with maxW = {maxW}]', write_mode='a')
    
    # 4. Apply multiplicatively to event weights (which can be negative)
    AIw0 = AIw0 * weights[y == C0]
    
    ## Print stats
    prints.print_weights(weights=AIw0, y=np.zeros(len(AIw0)), output_file=output_file,
        header=f'Step 4. Total weights [input weights x AI weights]', write_mode='a')
    
    # ---------------------------------------------------
    ## 2. Renormalize (optional) (e.g. we want fixed overall normalization, or debug)
    
    if RN_ID is not None:
        print(f'Renormalizing with class [{RN_ID}] weight sum', 'yellow')
        
        sum_before = AIw0.sum()
        print(f'Sum(before): {sum_before:0.1f}')
        
        AIw0 /= np.sum(AIw0)                # Normalize
        AIw0 *= np.sum(weights[y == RN_ID]) # Scale
        
        sum_after = AIw0.sum()
        print(f'Sum(after):  {sum_after:0.1f}')
        ratio = f'Ratio = Sum(after) / Sum(before) = {sum_after/sum_before:0.2f}'
        print(ratio)
        print('')
        
        # Print stats
        prints.print_weights(weights=AIw0, y=np.zeros(len(AIw0)), output_file=output_file,
            header=f'Step 5. Total weights sum renormalized [wrt. class ID = {RN_ID}] | {ratio}', write_mode='a')
    
    
    # ===================================================
    # Save data to a parquet file
    
    tau_str = str(np.round(tau, 3))
    
    # Create DataFrame
    w_post_S2    = copy.deepcopy(weights)
    w_post_S2[y == C0] = AIw0 # only C0
    
    df = pd.DataFrame(w_post_S2, columns=[f'w_post_S2__tau_{tau_str}'])
    
    
    # ===================================================
    # Visualization
    # ---------------------------------------------------
    
    total_ndf     = 0.0
    total_chi2    = 0.0
    total_chi2_AI = 0.0
    
    chi2_table    = PrettyTable(["observable", "ndf", "chi2 / ndf", "(AI) chi2 / ndf"]) 
    
    # ---------------------------------------------------
    param = {
        'i':         None,
        
        'X':         None,
        'y':         None,
        'weights':   None,
        'ids':       None,
        'AIw0':      None,
        
        'C0':        C0,
        'C1':        C1,
        
        'label':     label,
        'param':     param,
        'local_dir': local_dir
    }
    
    # Loop over each observable (specific column of X)
    paramlist = []
    
    for i in pick_ind:
        paramlist.append(copy.deepcopy(param)) 
        
        # big numpy arrays should share the memory (i.e. no copy of X)
        paramlist[-1]['i']       = i
        paramlist[-1]['X']       = X
        paramlist[-1]['y']       = y
        paramlist[-1]['weights'] = weights
        paramlist[-1]['ids']     = ids
        paramlist[-1]['AIw0']    = AIw0
    
    # Start multiprocessing
    num_workers = multiprocessing.cpu_count() // 2 if num_cpus == 0 else num_cpus
    print(f'Multiprocessing {len(pick_ind)} plots with {num_workers} processes', 'yellow')

    pool = multiprocessing.Pool(processes=num_workers)
    tic  = time.time()
    r    = pool.map(multiprocess_AIRW_wrapper, paramlist)
    pool.close() # no more tasks
    pool.join()  # wrap up current tasks
    
    toc  = time.time()
    print(f'Took {toc-tic:0.2f} sec')
    
    # Collect summary statistics from the pool
    for i in range(len(r)):
        total_ndf     += r[i]['ndf_0']
        total_chi2    += r[i]['chi2_0']
        total_chi2_AI += r[i]['chi2_0_AI']

        chi2_table.add_row(r[i]['chi2_row'], divider=True if i == len(r)-1 else False)
    
    chi2_table.add_row(['total', f'{total_ndf}', f'{total_chi2/total_ndf:0.1f}', f'{total_chi2_AI/total_ndf:0.1f}'])
    
    # -----------------------------------------------
    filename = local_dir + f"/stats_chi2.log"
    open(filename, 'w').close() # Clear content
    table_writer(filename=filename, label=label, sublabel=sublabel, tau=tau, chi2_table=chi2_table, print_to_screen=True)
    # -----------------------------------------------
    
    return chi2_table, df


def multiprocess_AIRW_wrapper(p):
    """
    Multiprocessing plots
    """
    
    i         = p['i'] # Column index
    
    X         = p['X']
    y         = p['y']
    w         = p['weights']
    ids       = p['ids']
    
    AIw0      = p['AIw0']
    
    C0        = p['C0']
    C1        = p['C1']
    
    label     = p['label']
    param     = p['param']
    local_dir = p['local_dir']
    
    X_col = copy.deepcopy(X[:, i])
    bins  = binengine(bindef=param['edges'], x=X_col)
    
    # ----------------------------------------------------
    
    obs_X = {
        
        # Axis limits
        'xlim'    : (bins[0], bins[-1]),
        'ylim'    : None,
        'xlabel'  : f'{ids[i]}',
        'ylabel'  : r'Weighted counts',
        'units'   : {'x': r'a.u.', 'y' : r'1'},
        'label'   : f'{label}',
        'figsize' : (4, 3.75),
        
        # Ratio
        'ylim_ratio' : (0.75, 1.25),
        
        # Histogramming
        'bins'    : None,
        'density' : False,
        
        # Function to calculate
        'func'    : None
    }
    
    data_template = {
        'data'   : None,
        'weights': None,
        'label'  : 'Data',
        'hfunc'  : 'errorbar',
        'style'  : iceplot.errorbar_style,
        'obs'    : obs_X,
        'hdata'  : None,
        'color'  : None
    }
    
    # Data source <-> Observable collections
    class1    = data_template.copy() # Deep copies
    class0    = data_template.copy()
    class0_ML = data_template.copy()
    
    class1.update({
        'label' : '$C_1$',
        'hfunc' : 'hist',
        'style' : iceplot.hist_style_step,
        'color' : (0,0,0)
    })
    class0.update({
        'label' : '$C_0$',
        'hfunc' : 'hist',
        'style' : iceplot.hist_style_step,
        'color' : iceplot.imperial_green
    })
    class0_ML.update({
        'label' : '$C_0$ (AI)',
        'hfunc' : 'hist',
        'style' : iceplot.hist_style_step,
        'color' : iceplot.imperial_dark_red
    })

    data = [class1, class0, class0_ML]
    
    data[0]['hdata'] = iceplot.hist_obj(X_col[y == C1], bins=bins, weights=w[y == C1])
    data[1]['hdata'] = iceplot.hist_obj(X_col[y == C0], bins=bins, weights=w[y == C0])
    data[2]['hdata'] = iceplot.hist_obj(X_col[y == C0], bins=bins, weights=AIw0)
    
    # Update labels with chi2
    chi2_0, ndf_0       = iceplot.chi2_cost(h_mc=data[1]['hdata'], h_data=data[0]['hdata'], return_nbins=True)
    chi2_0_AI, ndf_0_AI = iceplot.chi2_cost(h_mc=data[2]['hdata'], h_data=data[0]['hdata'], return_nbins=True)
    
    data[1]['label'] += f' | $\\chi^2 = {chi2_0:0.0f} \, / \, {ndf_0} = {chi2_0/ndf_0:0.1f}$'
    data[2]['label'] += f' | $\\chi^2 = {chi2_0_AI:0.0f} \, / \, {ndf_0_AI} = {chi2_0_AI/ndf_0_AI:0.1f}$'
    
    # --------------------
    
    # Plot it
    fig0, ax0 = iceplot.superplot(data, ratio_plot=True, yscale='log', ratio_error_plot=True)
    fig1, ax1 = iceplot.superplot(data, ratio_plot=True, yscale='linear', ratio_error_plot=True)
    
    fig0.savefig(aux.makedir(local_dir + '/pdf/') + f'/reweight_[{ids[i]}]__log.pdf', bbox_inches='tight')
    fig0.savefig(aux.makedir(local_dir + '/png/') + f'/reweight_[{ids[i]}]__log.png', bbox_inches='tight', dpi=300)
    
    fig1.savefig(aux.makedir(local_dir + '/pdf/') + f'/reweight_[{ids[i]}]__linear.pdf', bbox_inches='tight')
    fig1.savefig(aux.makedir(local_dir + '/png/') + f'/reweight_[{ids[i]}]__linear.png', bbox_inches='tight', dpi=300)
    
    fig0.clf()
    fig1.clf()
    plt.close(fig0)
    plt.close(fig1)

    chi2_row = [ids[i], f'{ndf_0}', f'{chi2_0/ndf_0:0.1f}', f'{chi2_0_AI/ndf_0:0.1f}']
    
    return {'ndf_0': ndf_0, 'ndf_0_AI': ndf_0_AI, 'chi2_0': chi2_0, 'chi2_0_AI': chi2_0_AI, 'chi2_row': chi2_row}
