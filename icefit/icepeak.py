# Binned histogram chi2/likelihood/Wasserstein fit tools with
# mystic + iminuit (minuit from python)
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import numba
import iminuit
import copy
import yaml
import random

from numba.typed import List
from termcolor import cprint
import matplotlib.pyplot as plt

from scipy import interpolate
from scipy.stats import wasserstein_distance
import scipy.special as special
import scipy.integrate as integrate

from scipy.optimize import minimize
from mystic.solvers import diffev2
from mystic.solvers import fmin_powell


from iceplot import iceplot
from icefit import statstools

def get_function_maps():
    """
    Function handles
    """
    fmaps = {'generic_conv_pdf':  generic_conv_pdf,
             'exp_pdf':           exp_pdf,
             'poly_pdf':          poly_pdf,
             'asym_BW_pdf':       asym_BW_pdf,
             'asym_RBW_pdf':      asym_RBW_pdf,
             'RBW_pdf':           RBW_pdf,
             'cauchy_pdf':        cauchy_pdf,
             'CB_pdf':            CB_pdf,
             'DSCB_pdf':          DSCB_pdf,
             'gauss_pdf':         gauss_pdf,
             'voigt_pdf':         voigt_pdf}

    return fmaps

# ========================================================================
# Fit functions

@numba.njit
def voigt_FWHM(gamma: float, sigma: float):
    """
    Full width at half-maximum (FWHM) of a Voigtian function
    (Voigtian == a convolution integral between Lorentzian and Gaussian)
    
    Args:
        gamma: Lorentzian (non-relativistic Breit-Wigner) parameter (half width!)
        sigma: Gaussian distribution parameter
    Returns:
        full width at half-maximum
    
    See: Olivero, J. J.; R. L. Longbothum (1977).
        "Empirical fits to the Voigt line width: A brief review"
    
        https://en.wikipedia.org/wiki/Voigt_profile
    """
    
    # Full widths at half-maximum for Gaussian and Lorentzian
    G = 2*sigma*np.sqrt(2*np.log(2))
    L = 2*gamma

    # Approximate result
    return 0.5346*L + np.sqrt(0.2166*L**2 + G**2)


@numba.njit
def exp_pdf(x: np.ndarray, par: np.ndarray, norm: bool=True):
    """
    Exponential density
    
    Args:
        par: rate parameter (1/mean)
    """
    y = par[0] * np.exp(-par[0] * x)
    y[x < 0] = 0

    if norm:
        y = y / np.trapz(y=y, x=x)
    return y

@numba.njit
def poly_pdf(x: np.ndarray, par: np.ndarray, norm: bool=True):
    """
    Polynomial density y = p0 + p1*x + p2*x**2 + ...
    
    Args:
        par: polynomial function params
    """
    y   = np.zeros(len(x))
    for i in range(len(par)):
        y = y + par[i]*(x**i)

    if norm:
        y = y / np.trapz(y=y, x=x)
    return y

@numba.njit
def gauss_pdf(x: np.ndarray, par: np.ndarray, norm: bool=True):
    """
    Normal (Gaussian) density
    
    Args:
        par: parameters mu, sigma
    """
    mu, sigma = par
    y = 1.0 / (sigma * np.sqrt(2*np.pi)) * np.exp(- 0.5 * ((x - mu)/sigma)**2)
    
    if norm:
        y = y / np.trapz(y=y, x=x)
    return y

@numba.njit
def cauchy_pdf(x: np.ndarray, par: np.ndarray, norm: bool=True):
    """
    Cauchy (Lorentzian) pdf (non-relativistic fixed width Breit-Wigner)
    """
    M0, W0 = par

    y = 1 / (np.pi*W0) * (W0**2 / ((x - M0)**2 + W0**2))

    if norm:
        y = y / np.trapz(y=y, x=x)
    return y

#@numba.njit # (voig_profile not Numba compatible)
def voigt_pdf(x: np.ndarray, par: np.ndarray, norm: bool=True):
    """
    Voigtian pdf (Breit-Wigner convoluted with Gaussian)
    """
    M0, sigma, gamma = par
    
    y = special.voigt_profile(x - M0, sigma, gamma)

    if norm:
        y = y / np.trapz(y=y, x=x)
    return y

@numba.njit
def RBW_pdf(x: np.ndarray, par: np.ndarray, norm: bool=True):
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
        y = y / np.trapz(y=y, x=x)
    return y

@numba.njit
def asym_RBW_pdf(x: np.ndarray, par: np.ndarray, norm: bool=True):
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
        y = y / np.trapz(y=y, x=x)
    return y

@numba.njit
def asym_BW_pdf(x: np.ndarray, par: np.ndarray, norm: bool=True):
    """
    Breit-Wigner with asymmetric tail shape

    Param: a < 0 gives right hand tail, a == 0 without, a > 0 left hand tail
    """
    M0, W0, a = par

    # Asymmetric running width
    W = 2*W0 / (1 + np.exp(a * (x - M0)))
    y = 1 / (np.pi*W0) * (W**2 / ((x - M0)**2 + W**2))

    if norm:
        y = y / np.trapz(y=y, x=x)
    return y

@numba.njit
def CB_pdf(x: np.ndarray, par: np.ndarray, norm: bool=True, EPS: float=1E-12):
    """
    https://en.wikipedia.org/wiki/Crystal_Ball_function

    Consists of a Gaussian core portion and a power-law low-end tail,
    below a certain threshold.
    
    Args:
        par: mu > 0, sigma > 0, n > 1, alpha > 0
    """

    mu, sigma, n, alpha = par

    # Protect float
    abs_a = np.abs(alpha)

    A = (n / abs_a)**n * np.exp(-0.5 * abs_a**2)
    B =  n / abs_a - abs_a

    # Normalization (Wikipedia)
    #C = (n / abs_a) * (1 / max(n-1, EPS)) * np.exp(-0.5 * abs_a**2)
    #D = np.sqrt(np.pi/2) * (1 + special.erf(abs_a / np.sqrt(2)))
    #N = 1 / (sigma * (C + D))
    N = 1 # Simply use this and do numerical normalization
    
    # Piece wise definition
    y = np.zeros(len(x))
    t = (x - mu) / max(sigma, EPS)
    
    ind_0 = t > -alpha
    ind_1 = ~ind_0
    
    y[ind_0] = N * np.exp( - 0.5 * t[ind_0]**2)
    y[ind_1] = N * A * (B - t[ind_1])**(-n)

    if norm:
        y = y / np.trapz(y=y, x=x)
    return y

@numba.njit
def DSCB_pdf(x: np.ndarray, par: np.ndarray, norm: bool=True, EPS: float=1E-12):
    """
    Double sided Crystal-Ball
    
    https://arxiv.org/abs/1606.03833
    
    Args:
        par: mu > 0, sigma > 0, n_low > 1, alpha_low > 0, n_high > 1, alpha_high > 0
    """
    
    mu, sigma, n_low, alpha_low, n_high, alpha_high = par

    # Normalization
    N = 1 # Simply use this and do numerical normalization
    
    # Piece wise definition
    y = np.zeros(len(x))
    t = (x - mu) / max(sigma, EPS)
    
    ind_0 = (-alpha_low <= t) & (t <= alpha_high)
    ind_1 = t < -alpha_low
    ind_2 = t >  alpha_high

    y[ind_0] = N * np.exp(- 0.5 * t[ind_0]**2)
    y[ind_1] = N * np.exp(- 0.5 * alpha_low**2)  * (alpha_low / n_low   * (n_low / alpha_low   - alpha_low  - t[ind_1]))**(-n_low)
    y[ind_2] = N * np.exp(- 0.5 * alpha_high**2) * (alpha_high / n_high * (n_high / alpha_high - alpha_high + t[ind_2]))**(-n_high)
    
    if norm:
        y = y / np.trapz(y=y, x=x)
    return y

@numba.njit
def highres_x(x: np.ndarray, xfactor: float=0.2, Nmin: int=256):
    """
    Extend range and sampling of x
        
    Args:
        x:       array of values
        factor:  domain extension factor
        Nmin:    minimum number of samples
    """
    e = xfactor * (x[0] + x[-1])/2
    return np.linspace(x[0]-e, x[-1]+e, max(len(x), Nmin))

def generic_conv_pdf(x: np.ndarray, par: np.ndarray, pdf_pair: List[str],
                     par_index: List[int], norm: bool=True, xfactor: float=0.2, Nmin: int=256):
    """
    Convolution between two functions.
    
    Args:
        x:         function argument
        par:       function parameters from the (fit) routine
        pdf_pair:  names of two functions, e.g. ['CBW_pdf', 'G_pdf']
        par_index: indices of the parameter per pair, e.g. [[0,1,2], [3,4]]
        norm:      normalize the integral
        xfactor:   domain extension factor
        Nmin:      minimum number of function sample points
    """

    func0 = eval(f'{pdf_pair[0]}')
    func1 = eval(f'{pdf_pair[1]}')
    
    f0_param = np.array([par[i] for i in par_index[0]])
    f1_param = np.array([par[i] for i in par_index[1]])

    # High-resolution extended range convolution
    xp = highres_x(x=x, xfactor=xfactor, Nmin=Nmin)
    f0 = func0(x=xp, par=f0_param, norm=False)
    f1 = func1(x=xp, par=f1_param, norm=False)
    yp = np.convolve(a=f0, v=f1, mode='same')
    y  = interpolate.interp1d(xp, yp)(x)
    
    if norm:
        y = y / np.trapz(y=y, x=x)
    return y

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
    bin_width  = bin_edges[1:] - bin_edges[:-1]
    
    return {'counts': counts, 'errors': errors, 'bin_width': bin_width,
            'bin_edges': bin_edges, 'bin_center': bin_center}

def hist_decompose(h, param, techno):
    """
    Decompose histogram dictionary further
    """
    
    counts = h['counts']
    errors = h['errors']
    cbins  = h['bin_center']
    edges  = h['bin_edges']
    
    # Limit the fit range
    range_mask = (param['fitrange'][0] <= cbins) & (cbins <= param['fitrange'][1])
    
    # Fitbins mask, i.e. which are within 'fitrange' and are positive definite
    posdef = (errors > techno['zerobin']) & (counts > techno['zerobin'])
    fitbin_mask = range_mask & posdef

    # Counts in the fit
    num_counts_in_fit = np.sum(counts[fitbin_mask])
    
    # For differential normalization
    index   = np.where(fitbin_mask)[0]
    mean_dx = np.mean(edges[index + 1] - edges[index])
    
    out = {'fitbin_mask': fitbin_mask, 'range_mask': range_mask,
           'num_counts_in_fit': num_counts_in_fit, 'mean_dx': mean_dx}

    return h | out # combine

def set_random_seeds(seed: int):
    """
    Set random seeds for numpy and random module.
    
    Args:
        seed: The seed value to set
    """
    print(__name__ + f'.set_random_seeds: Set seed {seed} \n')
    
    np.random.seed(seed)
    random.seed(seed)

def make_positive_semi_definite(cov_matrix: np.ndarray, epsilon=1e-6):
    """
    Ensure the covariance matrix is positive semi-definite by
    adding a small value to the diagonal.
    
    Args:
        cov_matrix: The covariance matrix
        epsilon:    A small value to add to the diagonal elements
    
    Returns:
        Adjusted covariance matrix
    """
    
    new_cov        = np.array(cov_matrix)
    min_eigenvalue = np.min(np.linalg.eigvalsh(new_cov))
    
    if min_eigenvalue < 0:
        new_cov -= min_eigenvalue * np.eye(new_cov.shape[0])
    new_cov += epsilon * np.eye(new_cov.shape[0])
    
    return new_cov

def get_ndf(fitbin_mask: np.ndarray, par: np.ndarray, fit_type: str):
    """
    Return the number of degrees of freedom
    """
    
    if   fit_type == 'single':
        return np.sum(fitbin_mask) - len(par)
    elif fit_type == 'dual':
        return np.sum(fitbin_mask) - len(par) / 2
    elif fit_type == 'dual-unitary-I':
        return np.sum(fitbin_mask) - ((len(par) - 3) / 2 + 3) 
    elif fit_type == 'dual-unitary-II':
        return np.sum(fitbin_mask) - ((len(par) - 2) / 2 + 2) 
    else:
        raise Exception(__name__ + f'.get_ndf: Unknown fit_type')

def binned_1D_fit(hist: dict, param: dict, fitfunc: dict, techno: dict, par_fixed: dict):
    """
    Main fitting function for a binned fit of multiple histograms
    fitted simultanously with some parameters shared and some unshared.
    
    Args:
        hist:           Multiple TH1 histogram objects in a dictionary (from uproot)
        param:          Fitting parametrization dictionary
        fitfunc:        Fit functions per histogram
        techno:         Technical parameters (see yaml steering cards)
        par_fixed:      Fixed external parameters of the fit
        
    Returns:
        par, cov, var2pos
    """
    
    h = {}
    for key in hist.keys():
        h[key] = None
    
    counts            = copy.deepcopy(h)
    errors            = copy.deepcopy(h)
    cbins             = copy.deepcopy(h)
    fitbin_mask       = copy.deepcopy(h)
    num_counts_in_fit = 0
    
    # Pick single or two histograms (when 'dual' modes)
    for key in h.keys():
        
        h[key] = TH1_to_numpy(hist[key])
        d      = hist_decompose(h[key], param=param, techno=techno)

        counts[key]        = d['counts']
        errors[key]        = d['errors']
        cbins[key]         = d['bin_center']
        fitbin_mask[key]   = d['fitbin_mask']
        num_counts_in_fit += d['num_counts_in_fit'] 
    
    # Extract out
    losstype = techno['losstype']

    ### [Chi2 loss function]
    def chi2_loss(par):

        total = 0
        
        # Over histograms
        for key in counts.keys():
            
            mask = fitbin_mask[key]
            if np.sum(mask) == 0: return 1e9
            
            yhat = fitfunc[key](cbins[key][mask], par, par_fixed)
            xx   = (yhat - counts[key][mask])**2 / (errors[key][mask])**2

            total += np.sum(xx)

        return total

    ### [Poissonian negative log-likelihood loss function]
    def poiss_nll_loss(par):
        
        total = 0
        
        # Over histograms
        for key in counts.keys():
            
            mask = fitbin_mask[key]
            if np.sum(mask) == 0: return 1e9

            yhat = fitfunc[key](cbins[key][mask], par, par_fixed)
            T1 = counts[key][mask] * np.log(yhat)
            T2 = yhat

            total += (-1)*(np.sum(T1[np.isfinite(T1)]) - np.sum(T2[np.isfinite(T2)]))

        return total
    
    ### [Wasserstein (1st type) optimal transport distance -- experimental]
    def wasserstein_loss(par):
        
        total = 0
        
        # Over histograms
        for key in counts.keys():
            
            mask = fitbin_mask[key]
            if np.sum(mask) == 0: return 1e9
            
            yhat   = fitfunc[key](cbins[key][mask], par, par_fixed)
            total += wasserstein_distance(yhat, counts[key][mask], u_weights=None, v_weights=None)

        return total
    
    # --------------------------------------------------------------------
    if   losstype == 'chi2':
        loss = chi2_loss
    elif losstype == 'nll':
        loss = poiss_nll_loss
    elif losstype == 'wasserstein':
        loss = wasserstein_loss
    else:
        raise Exception(f'Unknown losstype chosen <{losstype}>')
    # --------------------------------------------------------------------
    
    trials = 0

    while True:

        # Execute optimizers
        m1 = optimizer_execute(trials=trials, loss=loss, param=param, techno=techno)
        
        ### Collect output
        par     = m1.values
        cov     = m1.covariance
        var2pos = m1.var2pos
        chi2    = chi2_loss(par)
        
        ndof = 0
        for key in fitbin_mask.keys():
            ndof += np.sum(fitbin_mask[key])
        ndof -= len(par) # data - param
        
        trials += 1
        
        if (chi2 / ndof < techno['max_chi2']):
            break
        
        # Could improve this logic (save the best trial if all are ~ weak, recover that)
        elif trials == techno['max_trials']:
            break
    
    print(f'Parameters: {par}')
    print(f'Covariance: {cov}')

    # --------------------------------------------------------------------
    # Improve numerical stability
    if cov is not None and techno['cov_eps'] > 0:
        cov = make_positive_semi_definite(cov, epsilon=techno['cov_eps'])
    
    # --------------------------------------------------------------------
    ## Inspect special edge cases
    
    par, cov = edge_cases(par=par, cov=cov, techno=techno,
                num_counts_in_fit=num_counts_in_fit, chi2=chi2, ndof=ndof)
    
    # --------------------------------------------------------------------
    # This is 'joint' chi2 for 'dual' type fits
    
    print(f"[{param['fit_type']}] chi2 / ndf = {chi2:.2f} / {ndof} = {chi2/ndof:.2f}")

    return par, cov, var2pos

def edge_cases(par, cov, techno, num_counts_in_fit, chi2, ndof):
    """
    Check edge cases after the fit
    """
    
    if cov is None:
        cprint('edge_cases: Uncertainty estimation failed (Minuit cov = None), return cov = -1', 'red')
        cov = -1 * np.ones((len(par), len(par)))
    
    if  num_counts_in_fit < techno['min_count']:
        cprint(f'edge_cases: Input histogram count < min_count = {techno["min_count"]} ==> fit not reliable', 'red')
        if techno['set_to_nan']:
            cprint('--> Setting parameters to NaN', 'red')
            par = np.nan*np.ones(len(par))
            cov = -1 * np.ones((len(par), len(par)))

    if ndof < techno['min_ndof']:
        cprint(f'edge_cases: Fit ndf = {ndof} < {techno["min_ndof"]} ==> fit not reliable', 'red')
        if techno['set_to_nan']:
            cprint('--> Setting parameters to NaN', 'red')
            par = np.nan*np.ones(len(par))
            cov = -1 * np.ones((len(par), len(par)))

    elif (chi2 / ndof) > techno['max_chi2']:
        cprint(f'edge_cases: Fit chi2/ndf = {chi2/ndof} > {techno["max_chi2"]} ==> fit not succesfull', 'red')
        if techno['set_to_nan']:
            cprint('--> Setting parameters to NaN', 'red')
            par = np.nan*np.ones(len(par))
            cov = -1 * np.ones((len(par), len(par)))

    return par, cov

def optimizer_execute(trials, loss, param, techno):
    """
    Optimizer execution wrapper
    """
    
    if trials == 0:
        start_values = param['start_values']
    else:
        # Randomly perturb around the default starting point and clip
        start_values = param['start_values']
        for i in range(len(start_values)):
            start_values[i] = np.clip(start_values[i] + 0.2 * start_values[i] * np.random.randn(), param['limits'][i][0], param['limits'][i][1])
    
    # ------------------------------------------------------------
    # Nelder-Mead search from scipy
    if techno['ncall_scipy_simplex'] > 0:
        
        options = {'maxiter': techno['ncall_scipy_simplex'], 'disp': True}

        res = minimize(loss, x0=start_values, method='nelder-mead', \
            bounds=param['limits'] if techno['use_limits'] else None, options=options)
        start_values = res.x

    # ------------------------------------------------------------
    # Mystic solver
    
    # Set search range limits
    bounds = []
    for i in range(len(param['limits'])):
        bounds.append(param['limits'][i])

    # Differential evolution 2 solver
    if techno['ncall_mystic_diffev2'] > 0:
        
        x0 = start_values
        start_values = diffev2(loss, x0=x0, bounds=bounds)
        cprint(f'Mystic diffev2 solution: {start_values}', 'green')

    # Fmin-Powell solver
    if techno['ncall_mystic_fmin_powell'] > 0:
        
        x0 = start_values
        start_values = fmin_powell(loss, x0=x0, bounds=bounds)
        cprint(f'Mystic fmin_powell solution: {start_values}', 'green')

    # --------------------------------------------------------------------
    # Set fixed parameter values, if True
    for k in range(len(param['fixed'])):
        if param['fixed'][k]:
            start_values[k] = param['start_values'][k]

    # --------------------------------------------------------------------
    ## Initialize Minuit
    m1 = iminuit.Minuit(loss, start_values, name=param['name'])

    # Fix parameters (minuit allows parameter fixing)
    for k in range(len(param['fixed'])):
        m1.fixed[k] = param['fixed'][k]
    # --------------------------------------------------------------------
    
    if   techno['losstype'] == 'chi2':
        m1.errordef = iminuit.Minuit.LEAST_SQUARES
    elif techno['losstype'] == 'nll':
        m1.errordef = iminuit.Minuit.LIKELIHOOD

    # Set parameter bounds
    if techno['use_limits']:
        m1.limits   = param['limits']

    # Optimizer parameters
    m1.strategy = techno['strategy']
    m1.tol      = techno['tol']
    
    # Minuit Brute force 1D-scan per dimension
    if techno['ncall_minuit_scan'] > 0:
        m1.scan(ncall=techno['ncall_minuit_scan'])
        print(m1.fmin)
    
    # Minuit Simplex (Nelder-Mead search)
    if techno['ncall_minuit_simplex'] > 0:
        m1.simplex(ncall=techno['ncall_minuit_simplex'])
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

    # Minuit Gradient search
    m1.migrad(ncall=techno['ncall_minuit_gradient'])
    print(m1.fmin)

    # Finalize with stat. uncertainty analysis [migrad << hesse << minos (best)]
    if techno['minos']:
        try:
            print(f'binned_1D_fit: Computing MINOS stat. uncertainties')
            m1.minos()
        except:
            cprint(f'binned_1D_fit: Error occured with MINOS stat. uncertainty estimation, trying HESSE', 'red')
            m1.hesse()
    else:
        print('binned_1D_fit: Computing HESSE stat. uncertainties')
        m1.hesse()

    return m1

def analyze_1D_fit(hist, param: dict, techno: dict, fitfunc,
                   par, cov, par_fixed=None, nsamples=1000, num_MC=500):
    """
    Analyze and visualize fit results
    
    Args:
        hist:      TH1 histogram object (from uproot)
        param:     Input parameters of the fit
        techno:    Tech parameters
        fitfunc:   Total fit function
        
        par:       Parameters obtained from the fit
        cov:       Covariance matrix obtained from the fit
        par_fixed: 
        
        nsamples:  Number of samples of the function (on x-axis)
        num_MC:    Number of MC samples for uncertainty estimation
    
    Returns:
        output dictionary
    """

    h = TH1_to_numpy(hist)
    d = hist_decompose(h, param=param, techno=techno)

    counts      = d['counts']
    errors      = d['errors']
    cbins       = d['bin_center']
    range_mask  = d['range_mask']
    fitbin_mask = d['fitbin_mask']
    mean_dx     = d['mean_dx']
    
    # --------------------------------------------------------------------
    ## Create fit functions
    
    # Samples on x-axis
    x = np.linspace(np.min(cbins[range_mask]), np.max(cbins[range_mask]), int(nsamples))

    # Loop over function components
    y = {}
    for key in param['components']:
        
        y[key] = fitfunc(x, par, par_fixed, components=[key])
        
        # Protect for nan/inf
        if np.sum(~np.isfinite(y[key])) > 0:
            print(__name__ + f'.analyze_1D_fit: Evaluated function [{key}] contain nan/inf values (set to 0.0)')
            y[key][~np.isfinite(y[key])] = 0.0
    
    print(__name__ + f'.analyze_1D_fit: Input bin count sum: {np.sum(counts):0.1f} (full range)')
    print(__name__ + f'.analyze_1D_fit: Input bin count sum: {np.sum(counts[fitbin_mask]):0.1f} (in fit)')    
    
    # --------------------------------------------------------------------
    # Compute count value integrals inside fitrange

    N     = {}
    N_err = {}

    # Normalize integral measures to event counts
    # [normalization given by the data histogram binning because we fitted against it]
    
    for key in param['components']:
        N[key]     = integrate.simpson(y=y[key], x=x) / mean_dx
        N_err[key] = 0.0
    
    # --------------------------------------------------------------------
    # Estimate the yield uncertainty via Monte Carlo
    # (takes correlations taken into account via the covariance matrix)
    
    for key in param['components']:
        
        # Check that it is positive (semi)-definite (we did tag it -1 if not)
        if cov[0][0] > 0:
            
            PAR = np.random.multivariate_normal(par, cov, num_MC)
            
            # Apply parameter space boundaries
            for j in range(len(param['limits'])):
                PAR[:,j] = np.clip(PAR[:,j], param['limits'][j][0], param['limits'][j][1])
            
            # Evaluate integrals
            N_hat = np.zeros(num_MC)
            
            # ** Possible extension **
            # we assume external par_fixed to be 'exact', i.e. do not fluctuate them
            
            for i in range(num_MC):
                
                yy = fitfunc(x, PAR[i,:], par_fixed, components=[key])
                yy[~np.isfinite(yy)] = 0.0 # Protect for nan/inf
                
                N_hat[i] = integrate.simpson(y=yy, x=x) / mean_dx
            
            # Compute percentiles
            N_high = np.percentile(N_hat, 84)
            N_med  = np.percentile(N_hat, 50)
            N_low  = np.percentile(N_hat, 16)
            
            # Take symmetric +-1 sigma error (use abs for protection)
            N_err[key] = np.mean([np.abs(N_med - N_low), np.abs(N_med - N_high)])
            
        else:
            print('analyze_1D_fit: Missing covariance matrix, using Poisson (or weighted count) error as a proxy')
            
            N_sum = np.sum(counts[fitbin_mask])
            if N_sum > 0:
                N_err[key] = (N[key] / N_sum) * np.sqrt(np.sum(errors[fitbin_mask]**2))
    
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

    ## ------------------------------------------------
    ## UPPER PLOT
    
    fig, ax = iceplot.create_axes(**obs_M, ratio_plot=True)
    
    ax[0].errorbar(x=cbins, y=counts, yerr=errors, color=(0,0,0), 
                   label=f'Data, $N = {np.sum(counts[fitbin_mask]):0.1f}$ (in fit)', **iceplot.errorbar_style)
    ax[0].legend(frameon=False)
    ax[0].set_ylabel('Counts / bin')

    ## ------------------------------------------------
    # Compute chi2 and ndf
    
    yf   = fitfunc(x=cbins, par=par, par_fixed=par_fixed)
    chi2 = np.sum( (yf[fitbin_mask] - counts[fitbin_mask])**2 / (errors[fitbin_mask])**2 )
    ndof = get_ndf(fitbin_mask=fitbin_mask, par=par, fit_type=param['fit_type'])
    
    # --------------------------------------------------
    # Fit plots

    plt.sca(ax[0])

    # Plot total fit
    ytot = fitfunc(x=x, par=par, par_fixed=par_fixed)
    ftot = interpolate.interp1d(x=x, y=ytot)
    
    plt.plot(x, ytot, label="Total fit", color=(0.5,0.5,0.5))
    
    # Plot components
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
    chi2_ndf = chi2 / ndof if ndof > 0 else -999
    title = f"$\\chi^2 / n_\\mathrm{{dof}} = {chi2:.2f} / {ndof:0.0f} = {chi2_ndf:.2f}$"
    plt.title(title)
    
    # ---------------------------------------------------------------
    ## LOWER PULL PLOT
    
    plt.sca(ax[1])
    iceplot.plot_horizontal_line(ax[1], ypos=0.0)

    # Compute pulls
    pulls = (ftot(cbins[fitbin_mask]) - counts[fitbin_mask]) / errors[fitbin_mask]
    pulls[~np.isfinite(pulls)] = 0.0
    
    # Plot pulls
    ax[1].bar(x=cbins[fitbin_mask], height=pulls,
              width=cbins[1]-cbins[0], align='center', color=(0.7,0.7,0.7), label=f'Fit')
    
    label = f'(fit - count) / $\\sigma$'
    ax[1].set_ylabel(label)
    ticks = iceplot.tick_calc([-3,3], 1.0, N=6)
    iceplot.set_axis_ticks(ax=ax[1], ticks=ticks, dim='y')
    
    # Get pull histogram
    fig_pulls, ax_pulls = plot_pulls(pulls=pulls, xlabel=label)
    
    return {'fig': fig, 'ax': ax, 'h': h, 'N': N,
            'N_err': N_err, 'chi2': chi2, 'ndof': ndof,
            'pulls': pulls, 'fig_pulls': fig_pulls, 'ax_pulls': ax_pulls}


def plot_pulls(pulls, nbins=30, xlabel='pull', xlim=[-4,4], ylim=[0,None], density=True):
    """
    Create histogram of pulls with unit Gaussian N=(0,1) on top
    """
    
    fig, ax = plt.subplots(figsize=(5,4))
    plt.sca(ax)
    
    # Create a histogram and plot it
    counts, bin_edges = np.histogram(pulls, bins=np.linspace(xlim[0], xlim[1], nbins), density=density)
    bin_centers       = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    
    mu    = np.mean(pulls)
    sigma = np.std(pulls)
    
    plt.bar(bin_centers, counts, width=bin_edges[1] - bin_edges[0],
            color='grey', edgecolor='black', alpha=0.7,
            label=f'$\\mu = {mu:0.2f}, \\sigma = {sigma:0.2f}$')
    
    # Plot the unit Gaussian
    x = np.linspace(xlim[0], xlim[1], 1000)
    y = (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * x**2)
    plt.plot(x, y, color='red', linewidth=2, label=r'$\mathcal{N}(0,1)$')

    # Add labels and legend
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend(loc='upper right', fontsize=8)

    return fig,ax


"""
Parametrizations docs

'single'
--------

Pass           Fail

[theta_S       [theta_S
 p_S0          p_S0
 p_S1          p_S1
 ...      [x]  ...
 theta_B       theta_B
 p_B0          p_B0
 ...           ...
]              ]

'dual'
--------

[theta_S
 theta_B
 eps_S
 eps_B
 
 p_S0^pass
 p_S0^fail
 p_S1^pass
 p_S1^fail
 ...
 p_B0^pass
 p_B0^fail
 ...
]

'dual-unitary-I' and 'dual-unitary-II'
--------

[eps_S
 (eps_B)
 (f_S)
 
 p_S0^pass
 p_S0^fail
 p_S1^pass
 p_S1^fail
 ...
 p_B0^pass
 p_B0^fail
 ...
]
"""

def read_yaml_input(inputfile, fit_type=None):
    """
    Parse input YAML file for fitting

    Args:
        inputfile: yaml file path
        fit_type:  override fit type (optional)
    
    Returns:
        dictionary with parsed content
    """

    with open(inputfile) as file:
        steer = yaml.full_load(file)
        print(steer)
    
    # Function handles
    fmaps = get_function_maps()
    
    name         = []
    start_values = []
    limits       = []
    fixed        = []

    args   = {}
    cfunc  = {}
    w_pind = {}
    p_pind = {}

    if fit_type is not None: # Override
        steer['fit_type'] = fit_type

    fit_type   = steer['fit_type']
    components = list(steer['fit'].keys())
    
    if   fit_type == 'single':
        
        # Parse all fit functions, combine them into linear arrays and dictionaries
        i = 0
        for ckey in components:
            
            arg = steer['fit'][ckey]
            
            # Pick function
            ff          = arg['func']
            cfunc[ckey] = fmaps[ff]
            args[ckey]  = arg['args']

            # 1. Fit component weight parameter
            name.append(f'theta__{ckey}')
            w_pind[ckey] = i
            i += 1
            
            # Parameter starting values and limits
            start_values.append(arg['theta_start'])
            limits.append(arg['theta_limit'])
            fixed.append(arg['theta_fixed'])

            # ----------------------------------------------------------------
            
            # 2. Fit component shape parameters
            pind__ = []
            for p in arg['p_name']:
                name.append(f'p__{p}')
                pind__.append(i)
                i += 1
            p_pind[ckey] = pind__

            # Parameter starting values and limits
            N = len(arg['p_name'])
            
            for k in range(N):
                start_values.append(arg['p_start'][k])
                limits.append(arg['p_limit'][k])
                fixed.append(arg['p_fixed'][k])
            
    elif fit_type == 'dual' or fit_type == 'dual-unitary-I' or fit_type == 'dual-unitary-II':
        
        # ------------------------------------------

        p_pind = {'Pass': {}, 'Fail': {}}
        
        # Parse all fit functions, combine them into linear arrays and dictionaries
        i = 0
        for ckey in components:
            
            arg = steer['fit'][ckey]
            
            # Pick function
            ff          = arg['func']
            cfunc[ckey] = fmaps[ff]
            args[ckey]  = arg['args']
            
            # -------------------------------------------
            
            if fit_type in ['dual', 'dual-unitary-I']:
                
                if (fit_type == 'dual') or (fit_type == 'dual-unitary-I' and ckey == 'S'):
                    
                    # 1. Fit component weight parameter
                    name.append(f'theta__{ckey}')
                    w_pind[f'theta_{ckey}'] = i
                    i += 1
                    
                    # Parameter starting values and limits
                    start_values.append(arg['theta_start'])
                    limits.append(arg['theta_limit'])
                    fixed.append(arg['theta_fixed'])

                # Efficiency parameter
                name.append(f'eps__{ckey}')
                w_pind[f'eps_{ckey}'] = i
                i += 1
                
                # Parameter starting values and limits
                start_values.append(0.5)
                limits.append([0.0, 1.0])
                fixed.append(False)

            elif (fit_type == 'dual-unitary-II') and (ckey == 'S'):
                
                # Efficiency parameter
                name.append(f'eps__{ckey}')
                w_pind[f'eps_{ckey}'] = i
                i += 1
                
                # Parameter starting values and limits
                start_values.append(0.5)
                limits.append([0.0, 1.0])
                fixed.append(False)
                
                # Signal fraction parameter
                name.append(f'f__{ckey}')
                w_pind[f'f_{ckey}'] = i
                i += 1
                
                # Parameter starting values and limits
                start_values.append(0.5)
                limits.append([0.0, 1.0])
                fixed.append(False)
            
            # -------------------------------------------
            
            for pass_type in p_pind.keys():
                
                # Function shape parameters
                pind__ = []
                for p in arg['p_name']:
                    name.append(f'p__{p}_{pass_type}')
                    pind__.append(i)
                    i += 1
                p_pind[pass_type][ckey] = pind__
                
                # Parameter starting values and limits
                N = len(arg['p_name'])
                
                for k in range(N):
                    start_values.append(arg['p_start'][k])
                    limits.append(arg['p_limit'][k])
                    fixed.append(arg['p_fixed'][k])
            
    else:
        raise Exception(__name__ + f'.read_yaml_input: Unknown fit input')
    
    # Finally collect all
    param  = {'input_path':   steer['input_path'],
              'years':        steer['years'],
              'systematics':  steer['systematics'],
              'variations':   steer['variations'],
              'fitrange':     steer['fitrange'],
              'num_cpus':     steer['num_cpus'],
              'output_name':  steer['output_name'],
              'start_values': start_values,
              'limits':       limits, 
              'fixed':        fixed,
              'name':         name,
              'args':         args,
              'w_pind':       w_pind,
              'p_pind':       p_pind,
              'fit_type':     fit_type,
              'components':   components}

    # Collect fit functions
    fitfunc = get_total_fit_functions(fit_type, cfunc, w_pind, p_pind, args)
    
    return param, fitfunc, cfunc, steer['techno']

def get_total_fit_functions(fit_type, cfunc, w_pind, p_pind, args):
    
    # Single fit of 1 histogram with one scale W_i per fit component (S,B,...)
    if   (fit_type == 'single'):   
        
        # Total fit function as a linear (incoherent) superposition
        def fitfunc(x, par, par_fixed=None, components=None):
            y   = 0
            par = np.array(par) # Make sure it is numpy array
            
            components = p_pind.keys() if components is None else components

            for key in components:
                
                W = par[w_pind[key]]
                y += W * cfunc[key](x=x, par=par[p_pind[key]], **args[key])
            
            return y

        return fitfunc

    # Joint fit of 2 histograms (pass, fail) with 4 free scale param (eps_S, eps_B, theta_S, theta_B)
    # Compatible only with 'S' and 'B' (2 fit components per histogram)
    elif (fit_type == 'dual'):
        
        # Total fit function as a linear (incoherent) superposition
        def fitfunc_pass(x, par, par_fixed=None, components=['S', 'B']):
            y   = 0
            par = np.array(par) # Make sure it is numpy array
            
            # Collect parameters    
            theta_S = par[w_pind['theta_S']]
            eps_S   = par[w_pind['eps_S']]
            theta_B = par[w_pind['theta_B']]
            eps_B   = par[w_pind['eps_B']]
                        
            for key in components:
                
                if key == 'S':
                    W = eps_S * theta_S
                else:
                    W = eps_B * theta_B
                
                y += W * cfunc[key](x=x, par=par[p_pind['Pass'][key]], **args[key])
            return y

        def fitfunc_fail(x, par, par_fixed=None, components=['S', 'B']):
            y   = 0
            par = np.array(par) # Make sure it is numpy array
            
            # Collect parameters    
            theta_S = par[w_pind['theta_S']]
            eps_S   = par[w_pind['eps_S']]
            theta_B = par[w_pind['theta_B']]
            eps_B   = par[w_pind['eps_B']]
            
            for key in components:
                
                if key == 'S':
                    W = (1 - eps_S) * theta_S
                else:
                    W = (1 - eps_B) * theta_B
                
                y += W * cfunc[key](x=x, par=par[p_pind['Fail'][key]], **args[key])
            return y
        
        return {'Pass': fitfunc_pass, 'Fail': fitfunc_fail}
    
    # Joint fit of 2 histograms (pass, fail) with 3 free scale param (eps_S, eps_B, theta_S)
    # Compatible only with 'S' and 'B' (2 fit components per histogram)
    elif (fit_type == 'dual-unitary-I'):
        
        # Total fit function as a linear (incoherent) superposition
        def fitfunc_pass(x, par, par_fixed=None, components=['S', 'B']):
            
            # External data
            C_tot = par_fixed['C_pass'] + par_fixed['C_fail']
            
            y   = 0
            par = np.array(par) # Make sure it is numpy array
            
            # Collect parameters
            theta_S = par[w_pind['theta_S']]
            eps_S   = par[w_pind['eps_S']]
            eps_B   = par[w_pind['eps_B']]
            
            for key in components:
                
                if key == 'S':
                    W = eps_S * theta_S
                else:
                    W = eps_B * max(1E-9, C_tot - theta_S)
                
                y += W * cfunc[key](x=x, par=par[p_pind['Pass'][key]], **args[key])
            return y

        def fitfunc_fail(x, par, par_fixed=None, components=['S', 'B']):
            
            # External data
            C_tot = par_fixed['C_pass'] + par_fixed['C_fail']
            
            y   = 0
            par = np.array(par) # Make sure it is numpy array
            
            # Collect parameters
            theta_S = par[w_pind['theta_S']] 
            eps_S   = par[w_pind['eps_S']]
            eps_B   = par[w_pind['eps_B']]
            
            for key in components:
                
                if key == 'S':
                    W = (1 - eps_S) * theta_S
                else:
                    W = (1 - eps_B) * max(1E-9, C_tot - theta_S)
                
                y += W * cfunc[key](x=x, par=par[p_pind['Fail'][key]], **args[key])
            return y

        return {'Pass': fitfunc_pass, 'Fail': fitfunc_fail}
    
    # Joint fit of 2 histograms (pass, fail) with 2 free scale param (eps_S, f_S)
    # Compatible only with 'S' and 'B' (2 fit components per histogram)
    elif (fit_type == 'dual-unitary-II'):
        
        # Total fit function as a linear (incoherent) superposition
        def fitfunc_pass(x, par, par_fixed=None, components=['S', 'B']):
            
            # External data    
            C_tot = par_fixed['C_pass'] + par_fixed['C_fail']
            Omega = par_fixed['C_pass'] / C_tot
            
            y   = 0
            par = np.array(par) # Make sure it is numpy array
            
            # Collect parameters
            eps_S = par[w_pind['eps_S']]
            f_S   = par[w_pind['f_S']]
            
            for key in components:
                
                if key == 'S':
                    W = eps_S * f_S * C_tot
                else:
                    W = max(1E-9, Omega - eps_S * f_S) * C_tot
                
                y += W * cfunc[key](x=x, par=par[p_pind['Pass'][key]], **args[key])
            return y

        def fitfunc_fail(x, par, par_fixed=None, components=['S', 'B']):
            
            # External data    
            C_tot = par_fixed['C_pass'] + par_fixed['C_fail']
            Omega = par_fixed['C_pass'] / C_tot
            
            y   = 0
            par = np.array(par) # Make sure it is numpy array
            
            # Collect parameters    
            eps_S = par[w_pind['eps_S']]
            f_S   = par[w_pind['f_S']]
            
            for key in components:
                
                if key == 'S':
                    W = (1 - eps_S) * f_S * C_tot
                else:
                    W = max(1E-9, 1 - f_S - Omega + eps_S * f_S) * C_tot
                
                y += W * cfunc[key](x=x, par=par[p_pind['Fail'][key]], **args[key])
            return y
        
        return {'Pass': fitfunc_pass, 'Fail': fitfunc_fail}
    
    else:
        raise Exception('get_fit_functions: Unknown fit_type chosen')


"""
# Raytune
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
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
