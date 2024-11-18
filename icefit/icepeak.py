# Binned histogram chi2, Huber, Poisson likelihood fit tools with
# mystic + iminuit (minuit with python)
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
import scipy.special as special
import scipy.integrate as integrate

from scipy.optimize import minimize
from mystic.solvers import diffev2
from mystic.solvers import fmin_powell

from iceplot import iceplot


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
def exp_pdf(x: np.ndarray, par: np.ndarray, norm=False):
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
def poly_pdf(x: np.ndarray, par: np.ndarray, norm=False):
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
def gauss_pdf(x: np.ndarray, par: np.ndarray, norm=False):
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
def cauchy_pdf(x: np.ndarray, par: np.ndarray, norm=False):
    """
    Cauchy (Lorentzian) pdf (non-relativistic fixed width Breit-Wigner)
    """
    M0, W0 = par
    y = 1 / (np.pi*W0) * (W0**2 / ((x - M0)**2 + W0**2))

    if norm:
        y = y / np.trapz(y=y, x=x)
    return y

#@numba.njit # (voig_profile not Numba compatible)
def voigt_pdf(x: np.ndarray, par: np.ndarray, norm=False):
    """
    Voigtian pdf (Breit-Wigner convoluted with Gaussian)
    """
    M0, sigma, gamma = par
    y = special.voigt_profile(x - M0, sigma, gamma)

    if norm:
        y = y / np.trapz(y=y, x=x)
    return y

@numba.njit
def RBW_pdf(x: np.ndarray, par: np.ndarray, norm=False):
    """
    Relativistic Breit-Wigner pdf
    https://en.wikipedia.org/wiki/Relativistic_Breit%E2%80%93Wigner_distribution
    """
    M0, W0 = par

    gamma = np.sqrt(M0**2 * (M0**2 + W0**2))
    k     = (2*np.sqrt(2)*M0*W0*gamma) / (np.pi * np.sqrt(M0**2 + gamma))
    y = k / ((x**2 - M0**2)**2 + M0**2 * W0**2)

    if norm:
        y = y / np.trapz(y=y, x=x)
    return y

@numba.njit
def asym_RBW_pdf(x: np.ndarray, par: np.ndarray, norm=False):
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
def asym_BW_pdf(x: np.ndarray, par: np.ndarray, norm=False):
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
def CB_pdf(x: np.ndarray, par: np.ndarray, EPS: float=1E-12, norm=False):
    """
    https://en.wikipedia.org/wiki/Crystal_Ball_function

    Consists of a Gaussian core portion and a power-law low-end tail,
    below a certain threshold.
    
    [floating normalization]
    
    Args:
        par: mu > 0, sigma > 0, n > 1, alpha > 0
    """

    mu, sigma, n, alpha = par
    
    # Protect float
    abs_a = np.abs(alpha)

    A = (n / abs_a)**n * np.exp(-0.5 * abs_a**2)
    B =  n / abs_a - abs_a

    # Piece wise definition
    y = np.zeros(len(x))
    t = (x - mu) / max(sigma, EPS)
    
    ind_0 = t > -alpha
    ind_1 = ~ind_0
    
    y[ind_0] = np.exp( - 0.5 * t[ind_0]**2)
    y[ind_1] = A * (B - t[ind_1])**(-n)

    if norm:
        y = y / np.trapz(y=y, x=x)
    return y

@numba.njit
def DSCB_pdf(x: np.ndarray, par: np.ndarray, EPS: float=1E-12, norm=False):
    """
    Double sided Crystal-Ball
    
    https://arxiv.org/abs/1606.03833
    
    [floating normalization]
    
    Args:
        par: mu > 0, sigma > 0, n_low > 1, alpha_low > 0, n_high > 1, alpha_high > 0
    """
    
    mu, sigma, n_low, alpha_low, n_high, alpha_high = par

    # Piece wise definition
    y = np.zeros(len(x))
    t = (x - mu) / max(sigma, EPS)
    
    ind_0 = (-alpha_low <= t) & (t <= alpha_high)
    ind_1 = t < -alpha_low
    ind_2 = t >  alpha_high

    y[ind_0] = np.exp(- 0.5 * t[ind_0]**2)
    y[ind_1] = np.exp(- 0.5 * alpha_low**2)  * (alpha_low / n_low   * (n_low / alpha_low   - alpha_low  - t[ind_1]))**(-n_low)
    y[ind_2] = np.exp(- 0.5 * alpha_high**2) * (alpha_high / n_high * (n_high / alpha_high - alpha_high + t[ind_2]))**(-n_high)
    
    if norm:
        y = y / np.trapz(y=y, x=x)
    return y

def generic_conv_pdf(x: np.ndarray, par: np.ndarray, pdf_pair: List[str],
                     par_index: List[int], xfactor: float=0.2, Nmin: int=256,
                     kind='linear', norm=False):
    """
    Convolution between two functions
    
    Args:
        x:         function argument
        par:       function parameters from the (fit) routine
        pdf_pair:  names of two functions, e.g. ['CBW_pdf', 'G_pdf']
        par_index: indices of the parameter per pair, e.g. [[0,1,2], [3,4]]
        xfactor:   domain extension factor
        Nmin:      minimum number of function sample points
        kind:      final sampling interpolation type ('linear', 'quadratic', ...)
        norm:      normalize the integral
    """

    func0 = eval(f'{pdf_pair[0]}')
    func1 = eval(f'{pdf_pair[1]}')
    
    f0_param = np.array([par[i] for i in par_index[0]])
    f1_param = np.array([par[i] for i in par_index[1]])

    # High-resolution extended range convolution
    xp = highres_x(x=x, xfactor=xfactor, Nmin=Nmin)
    f0 = func0(x=xp, par=f0_param, norm=False)
    f1 = func1(x=xp, par=f1_param, norm=False)
    
    dx = xp[1] - xp[0]
    yp = np.convolve(a=f0, v=f1, mode='same') * dx
    y  = interpolate.interp1d(x=xp, y=yp, kind=kind)(x)
    
    if norm:
        y = y / np.trapz(y=y, x=x)
    
    return y

@numba.njit
def highres_x(x: np.ndarray, xfactor: float=0.5, Nmin: int=256):
    """
    Extend range and sampling of x.
    
    Args:
        x:       Array of values
        xfactor: Domain extension factor (e.g. 0.2 gives +- 20%)
        Nmin:    Minimum number of samples
    
    Returns:
        np.ndarray: Extended high-resolution array.
    """
    e = xfactor * (x[-1] - x[0]) / 2  # Extend based on the range of x
    return np.linspace(x[0] - e, x[-1] + e, max(len(x), Nmin))

@numba.njit
def logzero(x: np.ndarray):
    """
    log(x) for x > 0 and 0 otherwise elementwise
    """
    r    = np.zeros_like(x)
    mask = (x > 0)
    r[mask] = np.log(x[mask])
    return r

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

def edges2centerbins(edges):
    """ Bin edges to center bins"""
    return (edges[1:] + edges[:-1]) / 2

def edges2binwidth(edges):
    """ Bin edges to binwidths """
    return edges[1:] - edges[:-1]

def TH1_to_numpy(hist, dtype=np.float64):
    """
    Convert TH1 (ROOT) histogram to numpy array
    
    Args:
        hist: TH1 object (from uproot)
    """

    #for n, v in hist.__dict__.items(): # class generated on the fly
    #   print(f'{n} {v}')

    hh         = hist.to_numpy()
    counts     = np.array(hist.values(), dtype=dtype)
    errors     = np.array(hist.errors(), dtype=dtype)

    bin_edges  = np.array(hh[1], dtype=dtype)
    bin_center = edges2centerbins(bin_edges)
    bin_width  = edges2binwidth(bin_edges)
    
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
    
    # Check positive definiteness within the range-limited errors and counts
    fitbin_mask = (errors[range_mask] > techno['zerobin']) & (counts[range_mask] > techno['zerobin'])
    
    # Counts in the fit
    num_counts_in_fit = np.sum(counts[range_mask][fitbin_mask])
    
    # Set True for both the start and end of each bin based on range mask
    edges_range_mask       = np.zeros_like(edges, dtype=bool)
    edges_range_mask[:-1] |= range_mask   # Start of each bin
    edges_range_mask[1:]  |= range_mask   # End of each bin
    
    out = {'fitbin_mask':       fitbin_mask,
           'range_mask' :       range_mask,
           'edges_range_mask':  edges_range_mask,
           'num_counts_in_fit': num_counts_in_fit}

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
    
    # Check eigenvalues of the matrix
    min_eigenvalue = np.min(np.linalg.eigvalsh(cov_matrix))
    
    # Only adjust if the matrix has negative eigenvalues or if we need to add epsilon
    if min_eigenvalue < 0 or epsilon > 0:
        adjustment = max(0, -min_eigenvalue + epsilon)
        cov_matrix = cov_matrix + adjustment * np.eye(cov_matrix.shape[0])
    
    return cov_matrix

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

def huber_lossfunc(y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray, delta: float=1.0):
    """
    Compute the Huber loss (robust statistics) with uncertainties
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        sigma:  Array of uncertainties for each data point
        delta:  Threshold parameter (if None, use MAD based)
    Returns:
        float:  The total Huber loss.
    """
    
    # Residuals
    r     = (y_true - y_pred) / sigma
    abs_r = np.abs(r)

    # Compute the Huber loss
    loss = np.zeros_like(r)
    
    # Quadratic part
    quadratic_mask = (abs_r <= delta)
    loss[quadratic_mask] = 0.5 * r[quadratic_mask]**2
    
    # Linear part
    linear_mask = (abs_r > delta)
    loss[linear_mask] = delta * (abs_r[linear_mask] - 0.5 * delta)

    return np.sum(loss)

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
    bin_edges         = copy.deepcopy(h)
    fitbin_mask       = copy.deepcopy(h)
    range_mask        = copy.deepcopy(h)
    edges_range_mask  = copy.deepcopy(h)
    num_counts_in_fit = []
    
    # Pick single or two histograms (when 'dual' modes)
    for key in h.keys():
        
        h[key] = TH1_to_numpy(hist[key])
        d      = hist_decompose(h[key], param=param, techno=techno)

        counts[key]           = d['counts']
        errors[key]           = d['errors']
        cbins[key]            = d['bin_center']
        bin_edges[key]        = d['bin_edges']
        fitbin_mask[key]      = d['fitbin_mask']
        range_mask[key]       = d['range_mask']
        edges_range_mask[key] = d['edges_range_mask']
        num_counts_in_fit.append( d['num_counts_in_fit'] )
    

    ### [Chi2 loss function]
    def chi2_loss(par):

        tot = 0
        
        # Over histograms
        for key in counts.keys():
            
            edgermask = edges_range_mask[key]
            rmask     = range_mask[key]
            fmask     = fitbin_mask[key]
            if np.sum(fmask) == 0: return 1e9 # Empty input
            
            # ** Note use proper range_masks here due to trapz integral in fitfunc ! **
            y_pred = fitfunc[key](cbins[key][rmask], par, par_fixed=par_fixed, edges=bin_edges[key][edgermask])
            
            residual = (y_pred[fmask] - counts[key][rmask][fmask]) / errors[key][rmask][fmask]

            tot += np.sum(residual**2)

        return tot

    ### [Huber loss function]
    def huber_loss(par):

        tot = 0
        delta = techno['huber_delta']
        
        # Over histograms
        for key in counts.keys():
            
            edgermask = edges_range_mask[key]
            rmask     = range_mask[key]
            fmask     = fitbin_mask[key]
            if np.sum(fmask) == 0: return 1e9 # Empty input
            
            # ** Note use proper range_masks here due to trapz integral in fitfunc ! **
            y_pred = fitfunc[key](cbins[key][rmask], par, par_fixed=par_fixed, edges=bin_edges[key][edgermask])
            
            T = huber_lossfunc(y_true=counts[key][rmask][fmask], y_pred=y_pred[fmask],
                               sigma=errors[key][rmask][fmask], delta=delta)
            
            tot += T

        return tot

    ### [Poissonian negative delta log-likelihood loss function]
    def poiss_nll_loss(par):
        
        nll = 0
        
        # Over histograms
        for key in counts.keys():
            
            edgermask = edges_range_mask[key]
            rmask     = range_mask[key]
            fmask     = fitbin_mask[key]
            if np.sum(fmask) == 0: return 1e9 # Empty input
            
            # ** Note use proper range_masks here due to trapz integral in fitfunc ! **
            y_pred = fitfunc[key](cbins[key][rmask], par, par_fixed=par_fixed, edges=bin_edges[key][edgermask])
            
            # Bohm-Zech scale transform for weighted events (https://arxiv.org/abs/1309.1287)
            # https://scikit-hep.org/iminuit/notebooks/weighted_histograms.html
            s  = counts[key][rmask][fmask] / (errors[key][rmask][fmask]**2) # Per bin
            
            n  = counts[key][rmask][fmask] # Observed
            mu = y_pred[fmask]             # Predicted
            
            # Factor 2 x taken into account by setting `errordef = iminuit.Minuit.LIKELIHOOD`
            
            # Negative Delta log-likelihood with Bohm-Zech scale
            nll += np.sum( s * (n * (logzero(n) - logzero(mu)) + mu - n) )
            
            # Simple Negative log-likelihood
            #nll += np.sum(-n * logzero(mu) + mu)
        
        return nll
    
    # --------------------------------------------------------------------
    
    loss_type = techno['loss_type']

    cprint(__name__ + f".binned_1D_fit: Executing fit with loss_type: '{loss_type}'", 'magenta')
    
    if   loss_type == 'chi2':
        lossfunc = chi2_loss
    elif loss_type == 'huber':
        lossfunc = huber_loss
    elif loss_type == 'nll':
        lossfunc = poiss_nll_loss
    else:
        raise Exception(f'Unknown loss_type chosen <{loss_type}>')
    
    # --------------------------------------------------------------------
    # Optimization loop
    
    best_loss  = 1e20
    best_trial = None
    m1         = []
    
    trial = 0
    
    while trial < techno['trials']:
        
        start_values = generate_start_values(trial=trial, param=param, techno=techno)
        
        # Execute optimizers
        m1.append(optimizer_execute(start_values, lossfunc=lossfunc, param=param, techno=techno))

        if (m1[-1].fval < best_loss) or (trial == 0):
            best_loss  = m1[-1].fval
            best_trial = trial
        
        trial += 1
    
    # Pick the best
    best_m1 = m1[best_trial]
    
    # --------------------------------------------------------------------    
    # Finalize with stat. uncertainty analysis [migrad << hesse << minos (best)]
    if techno['minos']:
        try:
            print(f'Computing MINOS stat. uncertainties')
            best_m1.minos()
        except Exception as e:
            print(e)
            cprint(f'Error occured with MINOS stat. uncertainty estimation, trying HESSE', 'red')
            best_m1.hesse()
    else:
        print('Computing HESSE stat. uncertainties')
        best_m1.hesse()
    # --------------------------------------------------------------------    
    
    # --------------------------------------------------------------------
    # Collect values
    
    var2pos = best_m1.var2pos
    par     = best_m1.values
    cov     = best_m1.covariance
    chi2    = chi2_loss(par)
    
    # Calculate total DOF over each histogram (handles both single and dual fits)
    ndof    = sum(np.sum(mask) for mask in fitbin_mask.values()) - best_m1.nfit
    # -------------------------------
    
    print('')
    cprint(f'Best trial number: {best_trial} | Loss: {best_loss:0.3E}')
    cprint(f'Valid minimum: {best_m1.valid} | Accurate covariance: {best_m1.accurate}', 'magenta')
    print('')
    print(best_m1.params)
    print(f'Covariance:')
    print(cov)

    # --------------------------------------------------------------------
    # Improve numerical stability of the covariance matrix
    
    if cov is not None and techno['cov_eps'] > 0:
        cov = make_positive_semi_definite(cov, epsilon=techno['cov_eps'])
    
    # --------------------------------------------------------------------
    # Inspect special edge cases
    
    par, cov = edge_cases(par=par, cov=cov, techno=techno,
                num_counts_in_fit=num_counts_in_fit, chi2=chi2, ndof=ndof)
    
    # --------------------------------------------------------------------
    # This is 'joint' chi2 for 'dual' type fits
    
    str = 'joint metric' if 'dual' in param['fit_type'] else 'single metric'
    cprint(f"[{param['fit_type']}] chi2 / ndf = {chi2:.2f} / {ndof} = {chi2/ndof:.2f} [{str}]", 'yellow')
    print('')
    
    return par, cov, var2pos, best_m1

def generate_start_values(trial, param, techno):
    """
    Get optimization starting values
    
    1. First the default starting values.
    2. Then Gaussian local perturbations within limits.
    3. Then global uniform random sample within limits.
    """
    
    print('')
    
    if trial == 0:
        
        start_values = np.array(param['start_values'])
        cprint(f"Trial = {trial} (Using default start values)", 'magenta')

    elif trial < techno['trials'] // 2:
        
        start_values = np.zeros(len(param['start_values']))
        cprint(f"Trial = {trial} (Gaussian perturbation around default start values within limits)", 'magenta')
        for i in range(len(start_values)):
            start_values[i] = np.clip(start_values[i] + techno['rand_sigma'] * np.random.randn(),
                                    param['limits'][i][0], param['limits'][i][1])
    else:
        
        start_values = np.zeros(len(param['start_values']))
        cprint(f"Trial = {trial} (Uniform sampling of start values within hypercube limits)", 'magenta')
        for i in range(len(param['start_values'])):
            lower, upper = param['limits'][i]
            start_values[i] = np.random.uniform(lower, upper)
    
    # --------------------------------------------------------------------
    # Re-Set fixed parameter values, if True
    for k in range(len(param['fixed'])):
        if param['fixed'][k]:
            start_values[k] = param['start_values'][k]
    # --------------------------------------------------------------------
    
    return start_values

def edge_cases(par, cov, techno, num_counts_in_fit, chi2, ndof):
    """
    Check edge cases after the fit
    """
    
    if cov is None:
        cprint('Uncertainty estimation failed (Minuit cov = None), return cov = -1', 'red')
        cov = -1 * np.ones((len(par), len(par)))
    
    for i in range(len(num_counts_in_fit)):
        if  num_counts_in_fit[i] < techno['min_count']:
            cprint(f'Input histogram[{i}] count < min_count = {techno["min_count"]} ==> fit not reliable', 'red')
            if techno['set_to_nan']:
                cprint('--> Setting parameters to NaN', 'red')
                par = np.nan*np.ones(len(par))
                cov = -1 * np.ones((len(par), len(par)))

    if ndof < techno['min_ndof']:
        cprint(f'Fit ndf = {ndof} < {techno["min_ndof"]} ==> fit not reliable', 'red')
        if techno['set_to_nan']:
            cprint('--> Setting parameters to NaN', 'red')
            par = np.nan*np.ones(len(par))
            cov = -1 * np.ones((len(par), len(par)))

    elif (chi2 / ndof) > techno['max_chi2']:
        cprint(f'Fit chi2/ndf = {chi2/ndof} > {techno["max_chi2"]} ==> fit not succesfull', 'red')
        if techno['set_to_nan']:
            cprint('--> Setting parameters to NaN', 'red')
            par = np.nan*np.ones(len(par))
            cov = -1 * np.ones((len(par), len(par)))

    return par, cov

def optimizer_execute(start_values, lossfunc, param, techno):
    """
    Optimizer execution wrapper
    """
    
    # ------------------------------------------------------------
    # Nelder-Mead search from scipy
    if techno['ncall_scipy_simplex'] > 0:
        
        options = {'maxiter': techno['ncall_scipy_simplex'], 'disp': True}

        for _ in range(techno['ncall_scipy_simplex']): # Recursive calls
            res = minimize(lossfunc, x0=start_values, method='nelder-mead', \
                bounds=param['limits'] if techno['use_limits'] else None, options=options)
            start_values = res.x

    # ------------------------------------------------------------
    # Mystic solver
    
    # Set search range limits
    if techno['use_limits']:
        bounds = []
        for i in range(len(param['limits'])):
            bounds.append(param['limits'][i])
    else:
        bounds = None
    
    # Differential evolution 2 solver
    if techno['ncall_mystic_diffev2'] > 0:
        
        for _ in range(techno['ncall_mystic_diffev2']): # Recursive calls
            start_values = diffev2(lossfunc, x0=start_values, bounds=bounds)
            cprint(f'Mystic diffev2 solution: {start_values}', 'green')
    
    # Fmin-Powell solver
    if techno['ncall_mystic_fmin_powell'] > 0:
        
        for _ in range(techno['ncall_mystic_fmin_powell']): # Recursive calls       
            start_values = fmin_powell(lossfunc, x0=start_values, bounds=bounds)
            cprint(f'Mystic fmin_powell solution: {start_values}', 'green')

    # --------------------------------------------------------------------
    # Make sure parameters are within bounds
    # (optimizer might not exactly always enforce it)
    if techno['use_limits']:
        for k in range(len(start_values)):
            start_values[k] = np.clip(start_values[k], param['limits'][k][0], param['limits'][k][1])
        
    # --------------------------------------------------------------------
    # Re-set fixed parameter values, if True
    for k in range(len(param['fixed'])):
        if param['fixed'][k]:
            start_values[k] = param['start_values'][k]

    # --------------------------------------------------------------------
    ## Initialize Minuit
    m1 = iminuit.Minuit(lossfunc, start_values, name=param['name'])

    # Fix parameters (minuit allows parameter fixing)
    for k in range(len(param['fixed'])):
        m1.fixed[k] = param['fixed'][k]
    # --------------------------------------------------------------------
    
    if   'nll' in techno['loss_type']:
        cprint('Setting errordef "LIKELIHOOD" ', 'yellow')
        m1.errordef = iminuit.Minuit.LIKELIHOOD
    else:
        cprint('Setting errordef "LEAST_SQUARES" ', 'yellow')
        m1.errordef = iminuit.Minuit.LEAST_SQUARES
    
    ## Set parameter bounds
    if techno['use_limits']:
        m1.limits = param['limits']

    ## Set initial step size (of the first gradient step)
    if techno['migrad_first_step'] is not None:
        m1.errors = techno['migrad_first_step'] * np.abs(start_values)
    
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
    
    # Minuit Gradient search
    m1.migrad(ncall=techno['ncall_minuit_gradient'])
    print(m1.fmin)

    return m1

def analyze_1D_fit(hist, param: dict, techno: dict, fitfunc,
                   par, cov, par_fixed=None, num_visual: int=1000, num_MC: int=500, alpha: float=0.2):
    """
    Analyze and visualize fit results
    
    Args:
        hist:        TH1 histogram object (from uproot)
        param:       Input parameters of the fit
        techno:      Tech parameters
        fitfunc:     Total fit function
        
        par:         Parameters obtained from the fit
        cov:         Covariance matrix obtained from the fit
        par_fixed:   External fixed parameters ('dual' fits)
        
        num_visual:  Number of visualization samples of the function (on x-axis)
        num_MC:      Number of MC samples for uncertainty estimation
        alpha:       Uncertainty band transparency
    
    Returns:
        output dictionary fit data and figures
    """
    
    cprint(__name__ + f'.analyze_1D_fit:', 'magenta')
    
    h = TH1_to_numpy(hist)
    d = hist_decompose(h, param=param, techno=techno)

    counts           = d['counts']
    errors           = d['errors']
    bin_edges        = d['bin_edges']
    cbins            = d['bin_center']
    range_mask       = d['range_mask']
    edges_range_mask = d['edges_range_mask']
    fitbin_mask      = d['fitbin_mask']

    print(f'Input bin count sum: {np.sum(counts):0.1f} (full range)')
    print(f'Input bin count sum: {np.sum(counts[range_mask][fitbin_mask]):0.1f} (in fit)')    
    print('')
    
    # --------------------------------------------------------------------
    ## Compute event count yields
    
    components = ['tot']
    for key in param['components']:
        components.append(key)
    
    # Loop over function components
    y     = {}
    y_up  = {}
    y_lo  = {}
    N     = {}
    N_err = {}
    
    for key in components:
        
        y[key] = fitfunc(cbins[range_mask], par, par_fixed,
                         components=None if key == 'tot' else [key], edges=bin_edges[edges_range_mask])
        
        y[key][~np.isfinite(y[key])] = 0.0 # Protect for nan/inf

        # Compute total count as a bin-wise sum
        N[key] = np.sum(y[key])
    
    # --------------------------------------------------------------------
    # Estimate the yield uncertainty via Monte Carlo
    # (takes correlations taken into account via the covariance matrix)
    
    # Check that it is positive (semi)-definite (we did tag it -1 if not)
    if cov[0][0] > 0:

        PAR = np.random.multivariate_normal(par, cov, num_MC)

        # ** Possible extension **
        # we assume external par_fixed to be 'exact', i.e. do not fluctuate them
            
        # Apply parameter space boundaries
        for j in range(len(param['limits'])):
            PAR[:,j] = np.clip(PAR[:,j], param['limits'][j][0], param['limits'][j][1])

        for key in components:
            
            # Evaluate event counts
            N_hat = np.zeros(num_MC)
            Y     = np.zeros( (num_MC, len(y[key])) )
            
            for i in range(num_MC):

                yy = fitfunc(cbins[range_mask], PAR[i,:], par_fixed,
                             components=None if key == 'tot' else [key], edges=bin_edges[edges_range_mask])
                
                yy[~np.isfinite(yy)] = 0.0 # Protect for nan/inf
                Y[i,:]   = yy
                
                # Compute total count as a bin-wise sum
                N_hat[i] = np.sum(yy)
            
            # Take symmetric +-1 sigma error (use abs for protection)
            N_med      = np.percentile(N_hat, 50)
            N_err[key] = np.mean([np.abs(N_med - np.percentile(N_hat, 16)),
                                  np.abs(N_med - np.percentile(N_hat, 84))])
            
            # Confidence band
            y_lo[key] = np.percentile(Y, 16, axis=0)
            y_up[key] = np.percentile(Y, 84, axis=0)
            
    else:
        cprint('Missing covariance matrix, using Poisson (or weighted count) error as a proxy', 'red')
        
        N_sum = np.sum(counts[range_mask][fitbin_mask])
        if N_sum > 0:
            for key in components:
                
                N_err[key] = (N[key] / N_sum) * np.sqrt(np.sum(errors[range_mask][fitbin_mask]**2))
                
                y_up[key] = y[key] + np.sqrt(y[key] + 1E-12)
                y_lo[key] = y[key] - np.sqrt(y[key] + 1E-12)
    
    # --------------------------------------------------------------------
    # Print out

    cprint('Fit results:', 'yellow')
    for key in N.keys():
        print(f"N_{key}: \t {N[key]:0.1f} +- {N_err[key]:0.1f}")
    print('')
    
    # --------------------------------------------------------------------
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
                   label=f'Data, $N = {np.sum(counts[range_mask][fitbin_mask]):0.1f}$ (in fit)', **iceplot.errorbar_style)
    ax[0].set_ylabel('Counts / bin')

    ## ------------------------------------------------
    # Compute chi2 and ndf
    
    r    = (y['tot'][fitbin_mask] - counts[range_mask][fitbin_mask]) / errors[range_mask][fitbin_mask]
    chi2 = np.sum(r**2)
    ndof = get_ndf(fitbin_mask=fitbin_mask, par=par, fit_type=param['fit_type'])
    
    # --------------------------------------------------
    ## Compute fit functions for the plots

    # Loop over function components
    yfine    = {}
    yfine_up = {}
    yfine_lo = {}
    xfine    = np.linspace(np.min(cbins[range_mask]), np.max(cbins[range_mask]), num_visual)
    
    for key in components:
        
        x    = cbins[range_mask]
        
        kind = 'linear' if len(x) != len(np.unique(x)) else 'quadratic'
        
        # Central value
        ff            = interpolate.interp1d(x=x, y=y[key],    kind=kind)
        yfine[key]    = ff(xfine)

        # +1 sigma
        ff            = interpolate.interp1d(x=x, y=y_up[key], kind=kind)
        yfine_up[key] = ff(xfine)

        # -1 sigma
        ff            = interpolate.interp1d(x=x, y=y_lo[key], kind=kind)
        yfine_lo[key] = ff(xfine)
    
    # --------------------------------------------------
    ## Plot them
    
    plt.sca(ax[0])
    
    # Plot total
    color = (0.5,0.5,0.5)
    plt.plot(xfine, yfine['tot'], label="Total fit", color=color, zorder=1)
    plt.fill_between(xfine, yfine_lo['tot'], yfine_up['tot'], color=color, alpha=alpha, edgecolor="none", zorder=0.5)
    
    # Plot components
    colors     = [(0.7, 0.2, 0.2), (0.2, 0.2, 0.7)]
    linestyles = ['-', '--']
    i          = 0
    
    for key in yfine.keys():
        if key == 'tot':
            continue
        
        if i < 2:
            color     = colors[i]
            linestyle = linestyles[i]
        else:
            color     = np.random.rand(3)
            linestyle = '--'

        plt.plot(xfine, yfine[key], label=f"{key}: $N_{key} = {N[key]:.1f} \\pm {N_err[key]:.1f}$", color=color, linestyle=linestyle, zorder=i+2)
        plt.fill_between(xfine, yfine_lo[key], yfine_up[key], color=color, alpha=alpha, edgecolor="none", zorder=0.5)
        
        i += 1
    
    plt.ylim(bottom=0)
    plt.legend(frameon=True, fontsize=6.5)
    
    # chi2 / ndf
    chi2_ndf = chi2 / ndof if ndof > 0 else -999
    title    = f"$\\chi^2 / n_\\mathrm{{dof}} = {chi2:.2f} / {ndof:0.0f} = {chi2_ndf:.2f}$"
    print(f"plot: {title.replace('$', '')} \n")
    plt.title(title)
    
    # Remove the lowest tick mark from the upper axis (because it overlaps with the lower plot)
    ticks = ax[0].get_yticks()
    ax[0].set_yticks(ticks[1:])

    # ---------------------------------------------------------------
    ## LOWER PULL PLOT
    
    plt.sca(ax[1])
    iceplot.plot_horizontal_line(ax[1], ypos=0.0)

    ## Compute pulls at the bin centers
    
    # ** Note use proper range_masks here due to trapz integral in fitfunc ! **
    ftot  = fitfunc(cbins[range_mask], par, par_fixed, edges=bin_edges[edges_range_mask])
    pulls = (ftot[fitbin_mask] - counts[range_mask][fitbin_mask]) / errors[range_mask][fitbin_mask]
    pulls[~np.isfinite(pulls)] = 0.0
    
    # Plot pulls
    ax[1].bar(x=cbins[range_mask][fitbin_mask], height=pulls,
              width=cbins[1]-cbins[0], align='center', color=(0.7,0.7,0.7), label=f'Fit')
    
    label = f'(fit - count) / $\\sigma$'
    ax[1].set_ylabel(label)
    ticks = iceplot.tick_calc([-3,3], 1.0, N=6)
    iceplot.set_axis_ticks(ax=ax[1], ticks=ticks, dim='y')
    
    # ---------------------------------------------------------------
    ## Get pull histogram
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
                start_values.append(arg['eps_start'])
                limits.append(arg['eps_limit'])
                fixed.append(arg['eps_fixed'])

            elif (fit_type == 'dual-unitary-II') and (ckey == 'S'):
                
                # Efficiency parameter
                name.append(f'eps__{ckey}')
                w_pind[f'eps_{ckey}'] = i
                i += 1
                
                # Parameter starting values and limits
                start_values.append(arg['eps_start'])
                limits.append(arg['eps_limit'])
                fixed.append(arg['eps_fixed'])
                
                # Signal fraction parameter
                name.append(f'f__{ckey}')
                w_pind[f'f_{ckey}'] = i
                i += 1
                
                # Parameter starting values and limits
                start_values.append(arg['f_start'])
                limits.append(arg['f_limit'])
                fixed.append(arg['f_fixed'])
            
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

def integral_wrapper(lambdafunc, x, edges, norm=False, N_int: int=128, EPS=1E-8, **args):
    """
    Wrapper function for integral normalization
    
    Do not put too high N_int, otherwise numerical problems may arrise
    e.g. with numerical convolution pdfs.
    """
    
    # Evaluate
    f = lambdafunc(x)
    
    if norm:
        # Normalization based on a numerical integral over edge bounds
        x_fine = np.linspace(edges[0], edges[-1], N_int)
        y_fine = lambdafunc(x_fine)
        I = max(np.trapz(x=x_fine, y=y_fine), EPS)
        
        return f / I * edges2binwidth(edges)
    else:
        return f

def get_total_fit_functions(fit_type, cfunc, w_pind, p_pind, args):
    
    # Single fit of 1 histogram with one scale W_i per fit component (S,B,...)
    if   (fit_type == 'single'):   
        
        # Total fit function as a linear (incoherent) superposition
        def fitfunc(x, par, par_fixed=None, edges=None, components=None):
            ytot = 0
            par  = np.array(par) # Make sure it is numpy array
            
            components = p_pind.keys() if components is None else components

            for key in components:
                
                W = par[w_pind[key]]
                
                args_ = copy.deepcopy(args[key]); args_.pop('norm') # remove norm
                lambdafunc = lambda x_val: cfunc[key](x_val, par[p_pind[key]], **args_)
                ytot += W * integral_wrapper(lambdafunc, x=x, edges=edges, norm=args[key]['norm'])
            
            return ytot
        
        return fitfunc

    # Joint fit of 2 histograms (pass, fail) with 4 free scale param (eps_S, eps_B, theta_S, theta_B)
    # Compatible only with 'S' and 'B' (2 fit components per histogram)
    elif (fit_type == 'dual'):
        
        # Total fit function as a linear (incoherent) superposition
        def fitfunc_pass(x, par, par_fixed=None, edges=None, components=None):
            
            components = ['S', 'B'] if components is None else components
            
            ytot = 0
            par  = np.array(par) # Make sure it is numpy array
            
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
                
                args_ = copy.deepcopy(args[key]); args_.pop('norm') # remove norm
                lambdafunc = lambda x_val: cfunc[key](x_val, par[p_pind['Pass'][key]], **args_)
                ytot += W * integral_wrapper(lambdafunc, x=x, edges=edges, norm=args[key]['norm'])
            
            return ytot

        def fitfunc_fail(x, par, par_fixed=None, edges=None, components=None):
            
            components = ['S', 'B'] if components is None else components
            
            ytot = 0
            par  = np.array(par) # Make sure it is numpy array
            
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
                
                args_ = copy.deepcopy(args[key]); args_.pop('norm') # remove norm
                lambdafunc = lambda x_val: cfunc[key](x_val, par[p_pind['Fail'][key]], **args_)
                ytot += W * integral_wrapper(lambdafunc, x=x, edges=edges, norm=args[key]['norm'])
            
            return ytot
        
        return {'Pass': fitfunc_pass, 'Fail': fitfunc_fail}
    
    # Joint fit of 2 histograms (pass, fail) with 3 free scale param (eps_S, eps_B, theta_S)
    # Compatible only with 'S' and 'B' (2 fit components per histogram)
    elif (fit_type == 'dual-unitary-I'):
        
        # Total fit function as a linear (incoherent) superposition
        def fitfunc_pass(x, par, par_fixed=None, edges=None, components=None):
            
            components = ['S', 'B'] if components is None else components
            
            # External data
            C_tot = par_fixed['C_pass'] + par_fixed['C_fail']
            
            ytot = 0
            par  = np.array(par) # Make sure it is numpy array
            
            # Collect parameters
            theta_S = par[w_pind['theta_S']]
            eps_S   = par[w_pind['eps_S']]
            eps_B   = par[w_pind['eps_B']]
            
            for key in components:
                
                if key == 'S':
                    W = eps_S * theta_S
                else:
                    W = eps_B * max(1E-9, C_tot - theta_S)
                
                args_ = copy.deepcopy(args[key]); args_.pop('norm') # remove norm
                lambdafunc = lambda x_val: cfunc[key](x_val, par[p_pind['Pass'][key]], **args_)
                ytot += W * integral_wrapper(lambdafunc, x=x, edges=edges, norm=args[key]['norm'])
            
            return ytot

        def fitfunc_fail(x, par, par_fixed=None, edges=None, components=None):
            
            components = ['S', 'B'] if components is None else components
            
            # External data
            C_tot = par_fixed['C_pass'] + par_fixed['C_fail']
            
            ytot = 0
            par  = np.array(par) # Make sure it is numpy array
            
            # Collect parameters
            theta_S = par[w_pind['theta_S']] 
            eps_S   = par[w_pind['eps_S']]
            eps_B   = par[w_pind['eps_B']]
            
            for key in components:
                
                if key == 'S':
                    W = (1 - eps_S) * theta_S
                else:
                    W = (1 - eps_B) * max(1E-9, C_tot - theta_S)
                
                args_ = copy.deepcopy(args[key]); args_.pop('norm') # remove norm
                lambdafunc = lambda x_val: cfunc[key](x_val, par[p_pind['Fail'][key]], **args_)
                ytot += W * integral_wrapper(lambdafunc, x=x, edges=edges, norm=args[key]['norm'])
            
            return ytot

        return {'Pass': fitfunc_pass, 'Fail': fitfunc_fail}
    
    # Joint fit of 2 histograms (pass, fail) with 2 free scale param (eps_S, f_S)
    # Compatible only with 'S' and 'B' (2 fit components per histogram)
    elif (fit_type == 'dual-unitary-II'):
        
        # Total fit function as a linear (incoherent) superposition
        def fitfunc_pass(x, par, par_fixed=None, edges=None, components=None):
            
            components = ['S', 'B'] if components is None else components
            
            # External data    
            C_tot = par_fixed['C_pass'] + par_fixed['C_fail']
            Omega = par_fixed['C_pass'] / C_tot
            
            ytot = 0
            par  = np.array(par) # Make sure it is numpy array
            
            # Collect parameters
            eps_S = par[w_pind['eps_S']]
            f_S   = par[w_pind['f_S']]
            
            for key in components:
                
                if key == 'S':
                    W = eps_S * f_S * C_tot
                else:
                    W = max(1E-9, Omega - eps_S * f_S) * C_tot
                
                args_ = copy.deepcopy(args[key]); args_.pop('norm') # remove norm
                lambdafunc = lambda x_val: cfunc[key](x_val, par[p_pind['Pass'][key]], **args_)
                ytot += W * integral_wrapper(lambdafunc, x=x, edges=edges, norm=args[key]['norm'])
            
            return ytot

        def fitfunc_fail(x, par, par_fixed=None, edges=None, components=None):
            
            components = ['S', 'B'] if components is None else components
            
            # External data    
            C_tot = par_fixed['C_pass'] + par_fixed['C_fail']
            Omega = par_fixed['C_pass'] / C_tot
            
            ytot = 0
            par  = np.array(par) # Make sure it is numpy array
            
            # Collect parameters    
            eps_S = par[w_pind['eps_S']]
            f_S   = par[w_pind['f_S']]
            
            for key in components:
                
                if key == 'S':
                    W = (1 - eps_S) * f_S * C_tot
                else:
                    W = max(1E-9, 1 - f_S - Omega + eps_S * f_S) * C_tot
                
                args_ = copy.deepcopy(args[key]); args_.pop('norm') # remove norm
                lambdafunc = lambda x_val: cfunc[key](x_val, par[p_pind['Fail'][key]], **args_)
                ytot += W * integral_wrapper(lambdafunc, x=x, edges=edges, norm=args[key]['norm'])
            
            return ytot
        
        return {'Pass': fitfunc_pass, 'Fail': fitfunc_fail}
    
    else:
        raise Exception('get_fit_functions: Unknown fit_type chosen')
