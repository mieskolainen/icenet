# Binned histogram chi2/likelihood fits with iminuit (minuit from python)
# 
#
# Notes:
#   
#   Be careful with the "floating integral normalizations",
#   i.e. input array x (range, discretization) defines the
#   numerical integral if norm == True. Fitting a sum of pdf templates
#   without proper normalization can result in wrong yield uncertainties,
#   to remark.
# 
# 
# pytest icefit/peakfit.py -rP
# 
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk


# --------------------------------------
# JAX for autograd
# !pip install jax jaxlib

#import jax
#from jax.config import config
#config.update("jax_enable_x64", True) # enable float64 precision
#from jax import numpy as np           # jax replacement for normal numpy
#from jax.scipy.special import erf,erfc
#from jax import jit, grad
# --------------------------------------

import numpy as np

import yaml
import copy
import os
import pickle
import iminuit
import matplotlib.pyplot as plt
import uproot

from scipy import interpolate
import scipy.special as special
import scipy.integrate as integrate

# iceplot
import sys
sys.path.append(".")

from iceplot import iceplot
import icefit.statstools as statstools

# numpy
import numpy as onp # original numpy
from numpy.random import default_rng


"""
# Raytune
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import multiprocessing
import torch
"""


"""
def raytune_main(param, loss_func=None, inputs={}, num_samples=20, max_num_epochs=20):
    #
    # Raytune mainloop
    #
    def raytune_loss(p, **args):
        #
        # Loss Wrapper
        #
        par_arr = np.zeros(len(param['name']))
        i = 0
        for key in param['name']:
            par_arr[i] = p[key]
            i += 1

        loss = loss_func(par_arr)
        
        yield {'loss': loss}


    ### Construct hyperparameter config (setup) from yaml
    config = {}
    i = 0
    for key in param['name']:
        config[key] = tune.uniform(param['limits'][i][0], param['limits'][i][1])
        i += 1


    # Raytune basic metrics
    reporter = CLIReporter(metric_columns = ["loss", "training_iteration"])

    # Raytune search algorithm
    metric   = 'loss'
    mode     = 'min'

    # Hyperopt Bayesian / 
    search_alg = HyperOptSearch(metric=metric, mode=mode)

    # Raytune scheduler
    scheduler = ASHAScheduler(
        metric = metric,
        mode   = mode,
        max_t  = max_num_epochs,
        grace_period     = 1,
        reduction_factor = 2)

    # Raytune main setup
    analysis = tune.run(
        partial(raytune_loss, **inputs),
        search_alg          = search_alg,
        resources_per_trial = {"cpu": multiprocessing.cpu_count(), "gpu": 1 if torch.cuda.is_available() else 0},
        config              = config,
        num_samples         = num_samples,
        scheduler           = scheduler,
        progress_reporter   = reporter)
    
    # Get the best config
    best_trial = analysis.get_best_trial(metric=metric, mode=mode, scope="last")

    print(f'raytune: Best trial config:                {best_trial.config}', 'green')
    print(f'raytune: Best trial final validation loss: {best_trial.last_result["loss"]}', 'green')
    #cprint(f'raytune: Best trial final validation chi2:  {best_trial.last_result["chi2"]}', 'green')

    # Get the best config
    config = best_trial.config

    # Load the optimal values for the given hyperparameters
    optimal_param = np.zeros(len(param['name']))
    i = 0
    for key in param['name']:
        optimal_param[i] = config[key]
        i += 1
    
    print('Best parameters:')
    print(optimal_param)

    return optimal_param
"""


def TH1_to_numpy(hist):
    """
    Convert TH1 (ROOT) histogram to numpy array
    
    Args:
        hist: TH1 object (from uproot)
    """

    #for n, v in hist.__dict__.items(): # class generated on the fly
    #   print(f'{n} {v}')

    hh         = hist.to_numpy()
    counts     = np.array(hist.values())
    errors     = np.array(hist.errors())

    bin_edges  = np.array(hh[1])
    bin_center = np.array((bin_edges[1:] + bin_edges[:-1]) / 2)

    return {'counts': counts, 'errors': errors, 'bin_edges': bin_edges, 'bin_center': bin_center}


def gauss_pdf(x, par, norm=True):
    """
    Normal (Gaussian) density
    
    Args:
        par: parameters
    """
    mu, sigma = par
    y = 1.0 / (sigma * np.sqrt(2*np.pi)) * np.exp(- 0.5 * ((x - mu)/sigma)**2)
    
    if norm:
        y = y / integrate.simpson(y=y, x=x)
    return y


def CB_pdf(x, par, norm=True):
    """
    https://en.wikipedia.org/wiki/Crystal_Ball_function

    Consists of a Gaussian core portion and a power-law low-end tail,
    below a certain threshold.
    
    Args:
        par: mu > 0, sigma > 0, n > 1, alpha > 0
    """

    mu, sigma, n, alpha = par
    abs_a = np.abs(alpha) # Protect floats

    A = (n / abs_a)**n * np.exp(-0.5 * abs_a**2)
    B =  n / abs_a - abs_a

    C = (n / abs_a) * (1 / (n-1)) * np.exp(-0.5 * abs_a**2)
    D = np.sqrt(np.pi/2) * (1 + special.erf(abs_a / np.sqrt(2)))
    N =  1 / (sigma * (C + D))

    # Piece wise definition
    y = np.zeros(len(x))

    for i in range(len(y)):
        if (x[i] - mu)/sigma > -alpha:
            y[i] = N * np.exp(-(x[i] - mu)**2 / (2*sigma**2))
        else:
            y[i] = N * A*(B - (x[i] - mu)/sigma)**(-n)

    if norm:
        y = y / integrate.simpson(y=y, x=x)
    return y


def cauchy_pdf(x, par, norm=True):
    """
    Cauchy pdf (non-relativistic fixed width Breit-Wigner)
    """
    M0, W0 = par

    y = 1 / (np.pi*W0) * (W0**2 / ((x - M0)**2 + W0**2))

    if norm:
        y = y / integrate.simpson(y=y, x=x)
    return y


def RBW_pdf(x, par, norm=True):
    """
    Relativistic Breit-Wigner pdf
    https://en.wikipedia.org/wiki/Relativistic_Breit%E2%80%93Wigner_distribution
    """
    M0, W0 = par

    # Normalization
    gamma = np.sqrt(M0**2 * (M0**2 + W0**2))
    k     = (2*np.sqrt(2)*M0*W0*gamma) / (np.pi * np.sqrt(M0**2 + gamma))
    y = k / ((x**2 - M0**2)**2 + M0**2 * W0**2)

    if norm:
        y = y / integrate.simpson(y=y, x=x)
    return y


def asym_RBW_pdf(x, par, norm=True):
    """
    Asymmetric Relativistic Breit-Wigner pdf
    https://en.wikipedia.org/wiki/Relativistic_Breit%E2%80%93Wigner_distribution
    """
    M0, W0, a = par

    # Normalization
    gamma = np.sqrt(M0**2 * (M0**2 + W0**2))
    k     = (2*np.sqrt(2)*M0*W0*gamma) / (np.pi * np.sqrt(M0**2 + gamma))

    # Asymmetric running width
    W = 2*W0 / (1 + np.exp(a * (x - M0)))
    y = k / ((x**2 - M0**2)**2 + M0**2 * W**2)

    if norm:
        y = y / integrate.simpson(y=y, x=x)
    return y


def asym_BW_pdf(x, par, norm=True):
    """
    Breit-Wigner with asymmetric tail shape

    Param: a < 0 gives right hand tail, a == 0 without, a > 0 left hand tail
    """
    M0, W0, a = par

    # Asymmetric running width
    W = 2*W0 / (1 + np.exp(a * (x - M0)))
    y = 1 / (np.pi*W0) * (W**2 / ((x - M0)**2 + W**2))

    if norm:
        y = y / integrate.simpson(y=y, x=x)
    return y


def exp_pdf(x, par, norm=True):
    """
    Exponential density
    
    Args:
        par: rate parameter (1/mean)
    """
    y = par[0] * np.exp(-par[0] * x)
    y[x < 0] = 0

    if norm:
        y = y / integrate.simpson(y=y, x=x)
    return y


def poly_pdf(x, par, norm=True):
    """
    Polynomial density y = p0 + p1*x + p2*x**2 + ...
    
    Args:
        par: polynomial function params
    """
    y   = np.zeros(len(x))
    for i in range(len(par)):
        y = y + par[i]*(x**i)

    if norm:
        y = y / integrate.simpson(y=y, x=x)
    return y


def highres_x(x, xfactor=0.2, Nmin=256):
    """
    Extend range and sampling of x
        
    Args:
        x:       array of values
        factor:  domain extension factor
        Nmin:    minimum number of samples
    """
    e = xfactor * (x[0] + x[-1])/2
    return np.linspace(x[0]-e, x[-1]+e, np.maximum(len(x), Nmin))


def CB_G_conv_pdf(x, par, norm=True, xfactor=0.2, Nmin=256):
    """
    Crystall Ball (*) Gaussian, with the same center value as CB,
    where (*) is a convolution product.
    
    Args:
        par: CB parameters (4), Gaussian width (1)
    """
    mu     = par[0]
    reso   = par[-1]

    # High-resolution extended range convolution
    xp = highres_x(x=x, xfactor=xfactor, Nmin=Nmin)
    f1 = CB_pdf_(x=xp, par=par[:-1], norm=False)
    f2 = gauss_pdf(x=xp, par=np.array([mu, reso]), norm=False)
    yp = np.convolve(a=f1, v=f2, mode='same')
    y  = interpolate.interp1d(xp, yp)(x)
    
    if norm:
        y = y / integrate.simpson(y=y, x=x)
    return y


def CB_asym_RBW_conv_pdf(x, par, norm=True, xfactor=0.2, Nmin=256):
    """
    Crystall Ball (*) Asymmetric Relativistic Breit-Wigner, with the same center values,
    where (*) is a convolution product.
    
    Args:
        par: CB and asym RBW parameters as below
    """

    CB_param   = par[0],par[1],par[2],par[3]
    aRBW_param = par[0],par[4],par[5]

    # High-resolution extended range convolution
    xp = highres_x(x=x, xfactor=xfactor, Nmin=Nmin)
    f1 = CB_pdf(x=xp, par=CB_param, norm=False)
    f2 = asym_RBW_pdf(x=xp, par=aRBW_param, norm=False)
    yp = np.convolve(a=f1, v=f2, mode='same')
    y  = interpolate.interp1d(xp, yp)(x)
    
    if norm:
        y = y / integrate.simpson(y=y, x=x)
    return y


def CB_RBW_conv_pdf(x, par, norm=True, xfactor=0.2, Nmin=256):
    """
    Crystall Ball (*) Relativistic Breit-Wigner, with the same center values,
    where (*) is a convolution product.
    
    Args:
        par: CB and RBW parameters as below
    """

    CB_param  = par[0],par[1],par[2],par[3]
    RBW_param = par[0],par[4]

    # High-resolution extended range convolution
    xp = highres_x(x=x, xfactor=xfactor, Nmin=Nmin)
    f1 = CB_pdf(x=xp, par=CB_param, norm=False)
    f2 = RBW_pdf(x=xp, par=RBW_param, norm=False)
    yp = np.convolve(a=f1, v=f2, mode='same')
    y  = interpolate.interp1d(xp, yp)(x)
    
    if norm:
        y = y / integrate.simpson(y=y, x=x)
    return y


def binned_1D_fit(hist, param, fitfunc, techno):
    """
    Main fitting function for a binned fit
    
    Args:
        hist:           TH1 histogram object (from uproot)
        param:          Fitting parametrization dictionary
        fitfunc:        Fit function
        
        techno:
            losstype:       Loss function type
            ncall_simplex:  Number of calls
            ncall_gradient: Number of calls
            use_limits:     Use parameter limits
            max_trials:     Maximum number of restarts
            max_chi2:       Maximum chi2/ndf threshold for restarts
            min_count:      Minimum number of histogram counts
            
            hesse:          Hesse uncertainties
            minos:          Minos uncertainties
    """

    # -------------------------------------------------------------------------------
    # Histogram data

    h = TH1_to_numpy(hist)

    counts = h['counts']
    errs   = h['errors']
    cbins  = h['bin_center']

    # Limit the fit range
    fit_range_ind = (cbins >= param['fitrange'][0]) & (cbins <= param['fitrange'][1])

    # Extract out
    losstype = techno['losstype']

    ### Return fitbins
    def get_fitbins():

        posdef = (errs > techno['zerobin']) & (counts > techno['zerobin'])
        return fit_range_ind & posdef

    fitbins = get_fitbins()


    ### Chi2 loss function definition
    #@jit
    def chi2_loss(par):
        if np.sum(fitbins) == 0:
            return 1e9

        yhat = fitfunc(cbins[fitbins], par)
        xx   = (yhat - counts[fitbins])**2 / (errs[fitbins])**2
        
        return onp.sum(xx)

    ### Poissonian negative log-likelihood loss function definition
    #@jit
    def poiss_nll_loss(par):
        if np.sum(fitbins) == 0:
            return 1e9
        
        yhat = fitfunc(cbins[fitbins], par)
        T1 = counts[fitbins] * np.log(yhat)
        T2 = yhat

        return (-1)*(np.sum(T1[np.isfinite(T1)]) - np.sum(T2[np.isfinite(T2)]))

    # --------------------------------------------------------------------
    if   losstype == 'chi2':
        loss = chi2_loss
    elif losstype == 'nll':
        loss = poiss_nll_loss
    else:
        raise Exception(f'Unknown losstype chosen <{losstype}>')
    # --------------------------------------------------------------------

    # ====================================================================

    trials = 0

    while True:

        if trials == 0:
            start_values = param['start_values']
        else:
            start_values = param['start_values'] + np.random.rand(len(param['start_values']))



        # ------------------------------------------------------------
        # Nelder-Mead search
        from scipy.optimize import minimize
        options = {'maxiter': techno['ncall_simplex_scipy'], 'disp': True}

        res = minimize(loss, x0=start_values, method='nelder-mead', \
            bounds=param['limits'] if techno['use_limits'] else None, options=options)
        print(res)
        start_values = res.x


        # --------------------------------------------------------------------
        # Now reset fixed values
        for k in range(len(param['fixed'])):
            if param['fixed'][k]:
                start_values[k] = param['start_values'][k]
        # --------------------------------------------------------------------

        ## Initialize Minuit
        m1 = iminuit.Minuit(loss, start_values, name=param['name'])

        # Fix parameters 
        for k in range(len(param['fixed'])):
            m1.fixed[k] = param['fixed'][k]
        # --------------------------------------------------------------------
        

        if   techno['losstype'] == 'chi2':
            m1.errordef = iminuit.Minuit.LEAST_SQUARES
        elif techno['losstype'] == 'nll':
            m1.errordef = iminuit.Minuit.LIKELIHOOD


        if techno['use_limits']:
            m1.limits   = param['limits']

        # Optimizer parameters
        m1.strategy = techno['strategy']
        m1.tol      = techno['tol']
        
        # Brute force 1D-scan per dimension
        m1.scan(ncall=techno['ncall_scan'])
        print(m1.fmin)
        
        # Simplex (Nelder-Mead search)
        m1.simplex(ncall=techno['ncall_simplex'])
        print(m1.fmin)
        

        # --------------------------------------------------------------------
        # Raytune
        """
        values = m1.values
        param_new = copy.deepcopy(param)

        for i in range(len(param_new['start_values'])):
            param_new['limits'][i] = [values[i]-1, values[i]+1]

        out = raytune_main(param=param_new, loss_func=loss)
        """
        # --------------------------------------------------------------------

        # Gradient search
        m1.migrad(ncall=techno['ncall_gradient'])
        print(m1.fmin)

        # Finalize with error analysis [migrad << hesse << minos (best)]
        if techno['hesse']:
            m1.hesse()

        if techno['minos']:
            try:
                m1.minos()
            except:
                print(f'binned_1D_fit: Error occured with MINOS uncertainty estimation, trying HESSE')
                m1.hesse()

        ### Output
        par     = m1.values
        cov     = m1.covariance
        var2pos = m1.var2pos
        chi2    = chi2_loss(par)
        ndof    = np.sum(fitbins) - len(par) - 1
        
        trials += 1

        if (chi2 / ndof < techno['max_chi2']):
            break
        elif trials == techno['max_trials']:
            break
        

    print(f'Parameters: {par}')
    print(f'Covariance: {cov}')

    if cov is None:
        print('binned_1D_fit: Uncertainty estimation failed!')
        cov = -1 * np.ones((len(par), len(par)))

    if np.sum(counts[fit_range_ind]) < techno['min_count']:
        print(f'binned_1D_fit: Input histogram count < min_count = {techno["min_count"]} ==> fit not realistic')
        if techno['set_to_nan']:
            print('--> Setting parameters to NaN')
            par = np.nan*np.ones(len(par))
            cov = -1 * np.ones((len(par), len(par)))

    if (chi2 / ndof) > techno['max_chi2']:
        print(f'binned_1D_fit: chi2/ndf = {chi2/ndof} > {techno["max_chi2"]} ==> fit not succesful')
        if techno['set_to_nan']:
            print('--> Setting parameters to NaN')
            par = np.nan*np.ones(len(par))
            cov = -1 * np.ones((len(par), len(par)))

    print(f"chi2 / ndf = {chi2:.2f} / {ndof} = {chi2/ndof:.2f}")

    return par, cov, var2pos, chi2, ndof


def analyze_1D_fit(hist, param, fitfunc, cfunc, par, cov, var2pos, chi2, ndof, nsamples=1000):
    """
    Analyze and visualize fit results
    
    Args:
        hist:     TH1 histogram object (from uproot)
        param:    Input parameters of the fit
        fitfunc:  Total fit function
        cfunc:    Component functions
        
        par:      Parameters obtained from the fit
        cov:      Covariance matrix obtained from the fit
        var2pos:  Variable name to position index
        chi2:     Chi2 value of the fit
        ndof:     Number of dof

        nsamples: Number of samples of the functions
    
    Returns:
        fig, ax
    """

    h = TH1_to_numpy(hist)

    counts = h['counts']
    errs   = h['errors']
    cbins  = h['bin_center']

    # --------------------------------------------------------------------
    ## Create fit functions

    fitind = (param['fitrange'][0] <= cbins) & (cbins <= param['fitrange'][1])

    x   = np.linspace(param['fitrange'][0], param['fitrange'][1], int(nsamples))

    # Function by function
    y   = {}
    for key in cfunc.keys():
        weight = par[param['w_pind'][key]]
        y[key] = weight * cfunc[key](x=x, par=par[param['p_pind'][key]], **param['args'][key])

        # Protect for NaN/Inf
        if np.sum(~np.isfinite(y[key])) > 0:
            print(f'analyze_1D_fit: Evaluated function contain NaN/Inf values !')
            y[key][~np.isfinite(y[key])] = 0.0
    
    print(f'Input bin count sum: {np.sum(counts):0.1f} (full range)')
    print(f'Input bin count sum: {np.sum(counts[fitind]):0.1f} (fit range)')    
    
    
    # --------------------------------------------------------------------
    # Compute count value integrals inside fitrange

    N          = {}
    N_err      = {}

    # Normalize integral measures to event counts
    # [normalization given by the data histogram binning because we fitted against it]
    deltaX     = np.mean(cbins[fitind][1:] - cbins[fitind][:-1])

    for key in y.keys():
        N[key] = integrate.simpson(y[key], x) / deltaX


    # Use the scale error as the leading uncertainty
    # [neglect the functional shape uncertanties affecting the integral]
    
    for key in N.keys():
        ind = var2pos[f'w__{key}']
        
        if cov[ind][ind] > 0:
            N_err[key] = N[key] * np.sqrt(cov[ind][ind]) / par[ind]
        else:
            print('analyze_1D_fit: Non-positive definite covariance, using Poisson error as a proxy')
            N_err[key] = np.sqrt(np.maximum(1e-9, N[key]))

    # --------------------------------------------------------------------
    # Print out

    for key in N.keys():
        print(f"N_{key}: {N[key]:0.1f} +- {N_err[key]:0.1f}")


    # --------------------------------------------------------------------
    # Plot it

    obs_M = {

    # Axis limits
    'xlim'    : (cbins[0]*0.98, cbins[-1]*1.02),
    'ylim'    : None,
    'xlabel'  : r'$M$',
    'ylabel'  : r'Counts',
    'units'   : {'x': 'GeV', 'y': '1'},
    'label'   : r'Observable',
    'figsize' : (5,4),
    'density' : False,

    # Function to calculate
    'func'    : None
    }

    fig, ax = iceplot.create_axes(**obs_M, ratio_plot=True)
    
    ## UPPER PLOT
    ax[0].errorbar(x=cbins, y=counts, yerr=errs, color=(0,0,0), 
                   label=f'Data, $N = {np.sum(counts[fitind]):0.1f}$ (in fit)', **iceplot.errorbar_style)
    ax[0].legend(frameon=False)
    ax[0].set_ylabel('Counts / bin')

    ## Plot fits
    plt.sca(ax[0])

    yy = fitfunc(x=x, par=par)
    plt.plot(x, yy, label="Total fit", color=(0.5,0.5,0.5))
    
    colors     = [(0.7, 0.2, 0.2), (0.2, 0.2, 0.7)]
    linestyles = ['-', '--']
    i = 0
    for key in y.keys():
        if i < 2:
            color     = colors[i]
            linestyle = linestyles[i]
        else:
            color     = np.random.rand(3)
            linestyle = '--'

        plt.plot(x, y[key], label=f"{key}: $N_{key} = {N[key]:.1f} \\pm {N_err[key]:.1f}$", color=color, linestyle=linestyle)
        i += 1

    plt.ylim(bottom=0)
    plt.legend(fontsize=7)

    # chi2 / ndf
    title = f"$\\chi^2 / n_\\mathrm{{dof}} = {chi2:.2f} / {ndof} = {chi2/ndof:.2f}$"
    plt.title(title)

    ## LOWER PULL PLOT
    plt.sca(ax[1])
    iceplot.plot_horizontal_line(ax[1], ypos=0.0)

    # Compute fit function values
    fnew = interpolate.interp1d(x=x, y=yy)
    pull = (fnew(cbins[fitind]) - counts[fitind]) / (np.maximum(1e-12, errs[fitind]))

    # Plot pull
    ax[1].bar(x=cbins[fitind], height=pull, width=cbins[1]-cbins[0], color=(0.7,0.7,0.7), label=f'Fit')
    ax[1].set_ylabel('(fit - count) / $\\sigma$')
    ticks = iceplot.tick_calc([-3,3], 1.0, N=6)
    iceplot.set_axis_ticks(ax=ax[1], ticks=ticks, dim='y')
    
    return fig,ax,h,N,N_err


def iminuit2python(par, cov, var2pos):
    """
    Convert iminuit objects into standard python

    Args:
        par     : Parameter values object
        cov     : Covariance object
        var2pos : Variable string to index

    Returns:
        par_dict : Parameter values dictionary
        cov_arr  : Covariance matrix
    """
    par_dict = {}
    cov_arr  = np.zeros((len(var2pos), len(var2pos)))
    
    for key in var2pos.keys():
        i = var2pos[key]
        par_dict[key] = par[i]
        cov_arr[i][i] = cov[i][i]

    return par_dict, cov_arr


def read_yaml_input(inputfile):
    """
    Parse input YAML file for fitting

    Args:
        inputfile: yaml file path
    
    Returns:
        dictionary with parsed content
    """

    with open(inputfile) as file:
        steer = yaml.full_load(file)
        print(steer)

    # Function handles
    fmaps = {'exp_pdf':              exp_pdf,
             'poly_pdf':             poly_pdf,
             'asym_BW_pdf':          asym_BW_pdf,
             'asym_RBW_pdf':         asym_RBW_pdf,
             'RBW_pdf':              RBW_pdf,
             'cauchy_pdf':           cauchy_pdf,
             'CB_G_conv_pdf':        CB_G_conv_pdf,
             'CB_RBW_conv_pdf':      CB_RBW_conv_pdf,
             'CB_asym_RBW_conv_pdf': CB_asym_RBW_conv_pdf,
             'CB_pdf':               CB_pdf,
             'gauss_pdf':            gauss_pdf}

    name         = []
    start_values = []
    limits       = []
    fixed        = []

    args   = {}
    cfunc  = {}
    w_pind = {}
    p_pind = {}

    # Parse all fit functions, combine them into linear arrays and dictionaries
    i = 0
    for key in steer['fit'].keys():
        f = steer['fit'][key]['func']
        N = len(steer['fit'][key]['p_name'])

        cfunc[key] = fmaps[f]
        args[key]  = steer['fit'][key]['args']

        name.append(f'w__{key}')
        w_pind[key] = i
        start_values.append(steer['fit'][key]['w_start'])
        limits.append(steer['fit'][key]['w_limit'])
        fixed.append(steer['fit'][key]['w_fixed'])

        i += 1
        pind__ = []
        for p in steer['fit'][key]['p_name']:
            name.append(f'p__{p}')
            pind__.append(i)
            i += 1
        p_pind[key] = pind__

        for k in range(N):
            start_values.append(steer['fit'][key]['p_start'][k])
            limits.append(steer['fit'][key]['p_limit'][k])
            fixed.append(steer['fit'][key]['p_fixed'][k])

    # Total fit function as a linear (incoherent) superposition
    def fitfunc(x, par):
        y = 0
        for key in w_pind.keys():
            y += par[w_pind[key]] * cfunc[key](x=x, par=par[p_pind[key]], **args[key])
        return y

    # Finally collect all
    param  = {'path':         steer['path'],
              'start_values': start_values,
              'limits':       limits,
              'fixed':        fixed,
              'name':         name,
              'fitrange':     steer['fitrange'],
              'args':         args,
              'w_pind':       w_pind,
              'p_pind':       p_pind}

    print(param)

    return param, fitfunc, cfunc, steer['techno']


def get_rootfiles_jpsi(path='/', years=[2016, 2017, 2018]):
    """
    Return rootfile names for the J/psi study.
    """
    all_years = []

    # Loop over datasets
    for YEAR     in years:
        info = {}
        
        for TYPE in ['JPsi_pythia8', f'Run{YEAR}']: # Data or MC
            files = []
            
            # 1D-observables
            for OBS in ['absdxy']:
                for BIN in [1,2,3]:
                    for PASS in ['Pass', 'Fail']:

                        rootfile = f'{path}/Run{YEAR}/{TYPE}/Nominal/NUM_LooseID_DEN_TrackerMuons_{OBS}.root'
                        tree     = f'NUM_LooseID_DEN_TrackerMuons_{OBS}_{BIN}_{PASS}'

                        file = {'OBS': OBS, 'BIN': BIN, 'rootfile': rootfile, 'tree': tree}
                        files.append(file)                
            
            # 2D-observables
            for OBS1 in ['absdxy_sig', 'absdxy']:
                OBS2 = 'pt'

                # Binning
                for BIN1 in [1,2,3]:
                    for BIN2 in [1,2,3,4,5]:
                        for PASS in ['Pass', 'Fail']:

                            rootfile = f'{path}/Run{YEAR}/{TYPE}/Nominal/NUM_LooseID_DEN_TrackerMuons_{OBS1}_{OBS2}.root'
                            tree     = f'NUM_LooseID_DEN_TrackerMuons_{OBS1}_{BIN1}_{OBS2}_{BIN2}_{PASS}'

                            file = {'OBS1': OBS1, 'BIN1': BIN1, 'OBS2': OBS2, 'BIN2': BIN2, 'rootfile': rootfile, 'tree': tree}
                            files.append(file)

            info[TYPE] = files

        all_years.append({'YEAR': YEAR, 'info': info})

    return all_years


def run_jpsi_fitpeak(inputfile='tune0.yml', savepath='output/peakfit'):
    """
    J/psi peak fitting
    """

    import pytest

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    rng = default_rng(seed=1)

    
    # ====================================================================
    # Fit parametrization setup

    param, fitfunc, cfunc, techno = read_yaml_input(inputfile=inputfile)


    # ====================================================================
    #np.seterr(all='print') # Numpy floating point error treatment

    all_years = get_rootfiles_jpsi(path=param['path'])

    from pprint import pprint
    pprint(all_years)

    for y in all_years:
        YEAR = y['YEAR']

        for TYPE in y['info']:
            for f in y['info'][TYPE]:

                tree = f["tree"]
                hist = uproot.open(f["rootfile"])[tree]

                # Fit and analyze
                par,cov,var2pos,chi2,ndof = binned_1D_fit(hist=hist, param=param, fitfunc=fitfunc, techno=techno)
                fig,ax,h,N,N_err          = analyze_1D_fit(hist=hist, param=param, fitfunc=fitfunc, cfunc=cfunc, \
                                                           par=par, cov=cov, chi2=chi2, var2pos=var2pos, ndof=ndof)

                # Create savepath
                total_savepath = f'{savepath}/Run{YEAR}/{TYPE}/Nominal'
                if not os.path.exists(total_savepath):
                    os.makedirs(total_savepath)

                # Save the fit plot
                plt.savefig(f'{total_savepath}/{tree}.pdf')
                plt.close('all')
                
                # Save the fit numerical data
                par_dict, cov_arr = iminuit2python(par=par, cov=cov, var2pos=var2pos)
                outdict  = {'par':     par_dict,
                            'cov':     cov_arr,
                            'var2pos': var2pos,
                            'chi2':    chi2,
                            'ndof':    ndof,
                            'N':       N,
                            'N_err':   N_err,
                            'h':       h,
                            'param':   param}

                filename = f"{total_savepath}/{tree}.pkl"
                pickle.dump(outdict, open(filename, "wb"))
                print(f'Fit results saved to: {filename} (pickle) \n\n')


def run_jpsi_tagprobe(inputfile='tune0.yml', savepath='./output/peakfit'):
    """
    Tag & Probe efficiency (& scale factors)
    """

    import pytest
    
    def tagprobe(tree, total_savepath):

        N      = {}
        N_err  = {}

        for PASS in ['Pass', 'Fail']:

            filename = f"{total_savepath}/{f'{tree}_{PASS}'}.pkl"
            print(f'Reading fit results from: {filename} (pickle)')         
            outdict  = pickle.load(open(filename, "rb"))
            #pprint(outdict)

            # Read out signal peak fit event count yield and its uncertainty
            N[PASS]     = outdict['N']['S']
            N_err[PASS] = outdict['N_err']['S']

        return N, N_err

    # ====================================================================
    # Fit parametrization setup

    param, fitfunc, cfunc, techno = read_yaml_input(inputfile=inputfile)

    ## Read filenames
    all_years = get_rootfiles_jpsi(path=param['path'])


    ### Loop over datasets
    for y in all_years:
        
        YEAR     = y['YEAR']
        data_tag = f'Run{YEAR}'
        mc_tag   = 'JPsi_pythia8'

        # Create savepath
        total_savepath = f'{savepath}/Run{YEAR}/Efficiency'
        if not os.path.exists(total_savepath):
            os.makedirs(total_savepath)
        
        # Loop over observables -- pick 'data_tag' (both data and mc have the same observables)
        for f in y['info'][data_tag]:

            # Pick Pass -- just a pick (both Pass and Fail will be used)
            if '_Pass' in f['tree']:

                tree    = f['tree'].replace("_Pass", "")
                eff     = {}
                eff_err = {}

                # Loop over data and MC
                for TYPE in [data_tag, mc_tag]:

                    ### Compute Tag & Probe efficiency
                    N,N_err       = tagprobe(tree=tree, total_savepath=f'{savepath}/Run{YEAR}/{TYPE}/Nominal')
                    eff[TYPE]     = N['Pass'] / (N['Pass'] + N['Fail'])
                    eff_err[TYPE] = statstools.tpratio_taylor(x=N['Pass'], y=N['Fail'], x_err=N_err['Pass'], y_err=N_err['Fail'])

                    ### Print out
                    print(f'[{TYPE}]')
                    print(f'N_pass:     {N["Pass"]:0.1f} +- {N_err["Pass"]:0.1f} (signal fit)')
                    print(f'N_fail:     {N["Fail"]:0.1f} +- {N_err["Fail"]:0.1f} (signal fit)')
                    print(f'Efficiency: {eff[TYPE]:0.3f} +- {eff_err[TYPE]:0.3f} \n')

                ### Compute scale factor Data / MC
                scale     = eff[data_tag] / eff[mc_tag]
                scale_err = statstools.prodratio_eprop(A=eff[data_tag], B=eff[mc_tag], \
                            sigmaA=eff_err[data_tag], sigmaB=eff_err[mc_tag], sigmaAB=0, mode='ratio')

                print(f'Data / MC:  {scale:0.3f} +- {scale_err:0.3f} (scale factor) \n')

                ### Save results
                outdict  = {'eff': eff, 'eff_err': eff_err, 'scale': scale, 'scale_err': scale_err}
                filename = f"{total_savepath}/{tree}.pkl"
                pickle.dump(outdict, open(filename, "wb"))
                print(f'Efficiency and scale factor results saved to: {filename} (pickle)')


if __name__ == "__main__":

    inputfile = 'configs/peakfit/tune0.yml'

    run_jpsi_fitpeak(inputfile=inputfile)
    run_jpsi_tagprobe(inputfile=inputfile)
