# Linear (correlation) and non-linear dependency tools
#
# Run with: pytest icefit/cortools.py -rP (can take few minutes)
#
# m.mieskolainen@imperial.ac.uk, 2024

import sys
sys.path.append(".")

import numpy as np
import torch
#import numba
import copy
import scipy
import scipy.special as special
import scipy.stats as stats
import dcor

# Needed for tests only
import pandas as pd

from icefit import mine


def distance_corr_torch(x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor=None, p: float=2.0):
    """
    Distance Correlation, complexity O(N^2)
    
    (torch autograd compatible)
    
    https://en.wikipedia.org/wiki/Distance_correlation
    
    See: https://dcor.readthedocs.io/en/latest/theory.html
    
    Args:
        x       : data (N x dim)
        y       : data (N x dim), where dim can be the same or different than for x
        weights : event weights (not implemented)
        p       : p-norm [0,inf] value
    """
    N = x.shape[0]

    def double_centering(D):
        
        row_means = D.mean(dim=0, keepdim=True) # [1 x N]
        col_means = D.mean(dim=1, keepdim=True) # [N x 1]
        tot_mean  = D.flatten().mean()          # [scalar]
        
        return D - row_means - col_means + tot_mean
    
    # Add feature dimension if 1D
    if len(x.shape) == 1:
        x = x.unsqueeze(-1)
    if len(y.shape) == 1:
        y = y.unsqueeze(-1)
    
    # Unsqueeze(0) needed for cdist() (expects batched input)
    x = x.unsqueeze(0)
    y = y.unsqueeze(0)
    
    # Pairwise distance matrices
    Dx = torch.cdist(x, x, p=p).squeeze()
    Dy = torch.cdist(y, y, p=p).squeeze()
    
    # Double centered matrices
    A = double_centering(Dx)
    B = double_centering(Dy)

    dcovXY = (A * B).flatten().mean()  # V^2(X,Y) >= 0
    dvarXX = (A * A).flatten().mean()  # V^2(X,X) >= 0
    dvarYY = (B * B).flatten().mean()  # V^2(Y,Y) >= 0
    dvarXX_dvarYY = dvarXX * dvarYY
    
    if dvarXX_dvarYY > 0: # definition
        return torch.clip(torch.sqrt(dcovXY / torch.sqrt(dvarXX_dvarYY)), min=0.0, max=1.0)
    else:
        return torch.zeros(1).to(x.device)


def corrcoeff_weighted_torch(x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor=None):
    """
    Per event weighted linear Pearson 2x2 correlation matrix cf. torch.corrcoef()
    
    (torch autograd compatible)
    
    Args:
        x  :  data (Nx1)
        y  :  data (Nx1)
        weights : event weights
    """
    
    data   = torch.vstack((x.squeeze(), y.squeeze()))
    C      = torch.cov(data, aweights=weights.squeeze())
    N      = len(C)
    D      = torch.eye(N).to(C.device)
    
    D[range(N), range(N)] = 1.0 / torch.sqrt(torch.diagonal(C))
    
    return D @ C @ D

def percentile_per_dim(x: np.ndarray, q: float, **kwargs):
    """
    Compute percentiles for each dimension (column) over samples (rows)

    Args:
        input: (N x dim)
        q:     percentile (0,100)

    Returns:
        out:   array with length dim
    """
    return np.percentile(a=x, q=q, axis=0, **kwargs)

def prc_CI(x: np.ndarray, alpha: float):
    return np.array([np.percentile(x, 100*(alpha/2)), np.percentile(x, 100*(1-alpha/2))])

def hacine_entropy_bin(x: np.ndarray, mode: str="nbins", alpha: float=0.01):
    """
    Hacine-Gharbi et al. 
    “Low Bias Histogram-Based Estimation of Mutual Information for Feature Selection.”
    Pattern Recognition Letters, 2012.
    
    Args:
        See scott_bin()
    """

    N  = len(x)
    xi = (8 + 324*N + 12*np.sqrt(36*N + 729*N**2))**(1/3)
    nb = np.round(xi/6 + 2/(3*xi) + 1/3)

    if mode == "width":
        return (np.percentile(x, 100*(1-alpha/2)) - np.percentile(x, 100*alpha/2)) / nb
    else:
        return int(nb)

def hacine_joint_entropy_bin(x: np.ndarray, rho: float, mode: str="nbins", alpha: float=0.01):
    """
    Hacine-Gharbi, Ravier. "A Binning Formula of Bi-histogram
    for Joint Entropy Estimation Using Mean Square Error Minimization.”
    Pattern Recognition Letters, 2018.

    Args:
        See scott_bin()
    """

    N = len(x)

    # BX=BY
    nb = np.round(1/np.sqrt(2) * np.sqrt(1 + np.sqrt(1 + 24*N/(1-rho**2))))

    if mode == "width":
        return (np.percentile(x, 100*(1-alpha/2)) - np.percentile(x, 100*alpha/2)) / nb
    else:
        return int(nb)

def freedman_diaconis_bin(x: np.ndarray, mode: str="nbins", alpha: float=0.01):
    """
    Freedman-Diaconis rule for a 1D-histogram bin width
    
    D. Freedman & P. Diaconis (1981)
    “On the histogram as a density estimator: L2 theory”.
    
    ~ N**(-1/3)

    Args:
        x     : array of 1D data
        mode  : return 'width' or 'nbins'
        alpha : outlier percentile

    """
    IQR  = stats.iqr(x, rng=(25, 75), scale=1.0, nan_policy="omit")
    N    = len(x)
    bw   = (2 * IQR) / N**(1.0/3.0)

    if mode == "width":
        return bw
    else:
        return bw2bins(bw=bw, x=x, alpha=alpha)


def scott_bin(x: np.ndarray, rho: float, mode: str="nbins", alpha: float=0.01):

    """ 
    Scott rule for a 2D-histogram bin widths
    
    Scott, D.W. (1992),
    Multivariate Density Estimation: Theory, Practice, and Visualization -- 2D-Gaussian case
    
    ~ N**(-1/4)

    Args:
        x     : array of 1D data (one dimension of the bivariate distribution)
        rho   : Linear correlation coefficient
        mode  : return 'width' or 'nbins'
        alpha : outlier percentile
    """
    N  = len(x)
    bw = 3.504*np.std(x)*(1 - rho**2)**(3.0/8.0)/len(x)**(1.0/4.0)

    if mode == "width":
        return bw
    else:
        return bw2bins(bw=bw, x=x, alpha=alpha)

def bw2bins(x: np.ndarray, bw: float, alpha: float):
    """
    Convert a histogram binwidth to number of bins
    
    Args:
        x     : data array
        bw    : binwidth
        alpha : outlier percentile

    Returns:
        number of bins, if something fails return 1
    """
    if not np.isfinite(bw):
        return 1
    elif bw > 0:
        return int(np.ceil((np.percentile(x, 100*(1-alpha/2)) - np.percentile(x, 100*alpha/2)) / bw))
    else:
        return 1

def H_score(p: float, EPS: float=1E-15):
    """
    Shannon Entropy (log_e ~ nats units)

    Args:
        p : probability vector
    Returns:
        entropy
    """
    # Make sure it is normalized
    ind = (p > EPS)
    p_  = (p[ind]/np.sum(p[ind])).astype(np.float64)

    return -np.sum(p_*np.log(p_))

def I_score(C: np.ndarray, normalized: str=None, EPS: float=1E-15):
    """
    Mutual information score (log_e ~ nats units)

    Args:
        C          : (X,Y) 2D-histogram array with positive definite event counts
        normalized : return normalized version (None, 'additive', 'multiplicative')
    
    Returns:
        mutual information score
    """
    # Make sure it is positive definite
    C[C < 0] = 0

    # Joint 2D-density
    P_ij   = C / np.sum(C.flatten())

    # Marginal densities by summing over the other dimension
    P_i    = np.sum(C, axis=1); P_i /= np.sum(P_i)
    P_j    = np.sum(C, axis=0); P_j /= np.sum(P_j)
    
    # Factorized (1D) x (1D) density
    Pi_Pj  = np.outer(P_i, P_j)
    Pi_Pj /= np.sum(Pi_Pj.flatten())

    # Choose non-zero
    ind = (P_ij > EPS) & (Pi_Pj > EPS)

    # Mutual Information Definition
    I   = np.sum(P_ij[ind] * np.log(P_ij[ind] / Pi_Pj[ind]))
    I   = np.clip(I, 0.0, None)

    # Normalization
    if   normalized == None:
        return I
    elif normalized == 'additive':
        return 2*I/(H_score(P_i) + H_score(P_j))
    elif normalized == 'multiplicative':
        return I/np.sqrt(H_score(P_i) * H_score(P_j))
    else:
        raise Exception(f'I_score: Error with unknown normalization parameter "{normalized}"')


def mutual_information(x: np.ndarray, y: np.ndarray, weights = None,
                       bins_x=None, bins_y=None, normalized=None, alpha=0.32, n_bootstrap=300,
                       automethod='Scott2D', minbins=5, maxbins=100, outlier=0.01):
    """
    Mutual information entropy (non-linear measure of dependency)
    between x and y variables
    
    Args:
        x          : array of values
        y          : array of values
        weights    : weights (default None)
        bins_x     : x binning array  If None, then automatic.
        bins_y     : y binning array.
        normalized : normalize the mutual information (see I_score() function)
        n_bootstrap: number of percentile bootstrap samples
        alpha      : bootstrap confidence interval
    
    Autobinning args:    
        automethod : 'Hacine2D', 'Scott2D'
        minbins    : minimum number of bins per dimension
        outlier    : outlier protection percentile
    
    Returns:
        mutual information, confidence interval
    """

    if len(x) != len(y):
        raise Exception('mutual_information: x and y with different size.')
    
    if len(x) < 10:
        print(__name__ + f'.mutual_information: Error: len(x) < 10')
        return np.nan, np.array([np.nan, np.nan])
    
    x = np.asarray(x, dtype=float) # Require float for precision
    y = np.asarray(y, dtype=float)

    if weights is None:
        weights = np.ones(len(x), dtype=float)

    # Normalize to sum to one
    w = weights / np.sum(weights) 

    # For autobinning methods
    rho,_,_ = pearson_corr(x=x,y=y, weights=weights)

    def autobinwrap(data):
        if   automethod == 'Scott2D':
            NB = scott_bin(x=data, rho=rho, mode='nbins', alpha=outlier)
        elif automethod == 'Hacine2D':
            NB = hacine_joint_entropy_bin(x=data, rho=rho, mode='nbins', alpha=outlier)
        else:
            raise Exception(f'mutual_information: Unknown autobinning parameter <{automethod}>')

        NB = int(np.minimum(np.maximum(NB, minbins), maxbins))

        return np.linspace(np.percentile(data, outlier/2*100), np.percentile(data, 100*(1-outlier/2)), NB + 1)

    if bins_x is None:
        bins_x = autobinwrap(x)
    if bins_y is None:
        bins_y = autobinwrap(y)

    r_star = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):

        # Random values by sampling with replacement
        ind = np.random.choice(range(len(x)), size=len(x), replace=True)
        if i == 0:
            ind = np.arange(len(w))
        w_ = w[ind] / np.sum(w[ind])

        h2d = np.histogram2d(x=x[ind], y=y[ind], bins=[bins_x, bins_y], weights=w_)[0]

        # Compute MI
        r_star[i] = I_score(C=h2d, normalized=normalized)
    
    # The non-bootstrapped value (original sample based)
    r    = r_star[0] 

    # Percentile bootstrap based CI
    r_CI = prc_CI(r_star, alpha)

    return r, r_CI

def distance_corr(x: np.ndarray, y: np.ndarray,
                  weights: np.ndarray=None, alpha: float=0.32, n_bootstrap: int=30):
    """
    Distance correlation
    https://en.wikipedia.org/wiki/Distance_correlation
    
    Args:
        x,y        : arrays of values
        weights    : possible event weights
        alpha      : confidence interval [alpha/2, 1-alpha/2] level 
        n_bootstrap: number of percentile bootstrap samples
    
    Returns:
        correlation coefficient [0,1], confidence interval
    """
    
    if len(x) != len(y):
        raise Exception('distance_corr: x and y with different size.')

    if len(x) < 10:
        print(__name__ + '.distance_corr: Error: len(x) < 10')
        return np.nan, np.array([np.nan, np.nan])

    x = np.asarray(x, dtype=float) # Require float for precision
    y = np.asarray(y, dtype=float)

    if weights is None:
        weights = np.ones(len(x), dtype=float)

    # Normalize to sum to one
    w = weights / np.sum(weights) 

    # Obtain estimates and sample uncertainty via bootstrap
    r_star = np.zeros(n_bootstrap)

    for i in range(n_bootstrap):

        # Random values by sampling with replacement
        ind = np.random.choice(range(len(x)), size=len(x), replace=True)
        if i == 0:
            ind = np.arange(len(w))

        w_ = w[ind] / np.sum(w[ind])

        # Compute [T.B.D. add weighted version]
        r_star[i] = dcor.distance_correlation(x[ind], y[ind])

    # The non-bootstrapped value (original sample based)
    r    = r_star[0] 

    # Percentile bootstrap based CI
    r_CI = prc_CI(r_star, alpha)
    
    return r, r_CI


def pearson_corr(x: np.ndarray, y: np.ndarray, weights: np.ndarray=None,
                 return_abs=False, alpha=0.32, n_bootstrap=300):
    """
    Pearson Correlation Coefficient
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    
    Args:
        x,y        : arrays of values
        weights    : possible event weights
        return_abs : return absolute value
        alpha      : confidence interval [alpha/2, 1-alpha/2] level 
        n_bootstrap: number of percentile bootstrap samples
    
    Returns: 
        correlation coefficient [-1,1], confidence interval, p-value
    """

    if len(x) != len(y):
        raise Exception('pearson_corr: x and y with different size.')

    if len(x) < 10:
        print(__name__ + '.pearson_corr: Error: len(x) < 10')
        return np.nan, np.array([np.nan, np.nan]), np.nan

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if weights is None:
        weights = np.ones(len(x), dtype=float)

    # Normalize to sum to one
    w = weights / np.sum(weights) 

    # Loss of precision might happen here.
    x_ = x - np.sum(w*x)
    y_ = y - np.sum(w*y)

    # Obtain estimates and sample uncertainty via bootstrap
    r_star = np.zeros(n_bootstrap)

    # See: Efron, B. (1988). "Bootstrap confidence intervals:
    #      Good or bad?" Psychological Bulletin, 104, 293-296.
    for i in range(n_bootstrap):

        # Random values by sampling with replacement
        ind = np.random.choice(range(len(x)), size=len(x), replace=True)
        if i == 0:
            ind = np.arange(len(w))

        w_ = w[ind] / np.sum(w[ind])

        # corr(x,y; w) = cov(x,y; w) / [cov(x,x; w) * cov(y,y; w)]^{1/2}
        denom = np.sum(w_*(x_[ind]**2))*np.sum(w_*(y_[ind]**2))
        if denom > 0:
            r = np.sum(w_*x_[ind]*y_[ind]) / np.sqrt(denom)
        else:
            r = 0
        
        # Safety
        r = np.clip(r, -1.0, 1.0)

        # if we want absolute value
        if return_abs: r = np.abs(r)

        r_star[i] = r

    # The non-bootstrapped value (original sample based)
    r    = r_star[0] 

    # Percentile bootstrap based CI
    r_CI = prc_CI(r_star, alpha)

    # 2-sided p-value from the Beta-distribution
    ab   = len(x)/2 - 1
    dist = scipy.stats.beta(ab, ab, loc=-1, scale=2)
    prob = 2*dist.cdf(-abs(r))

    return r, r_CI, prob


def gaussian_mutual_information(rho: float):
    """
    Analytical 2D-Gaussian mutual information
    using a correlation coefficient rho.
    
    I(X1,X2) = H(X1) + H(X2) - H(X1,X2)
    Args:
        rho : correlation coefficient between (-1,1)
    Returns:
        mutual information
    """
    return -0.5*np.log(1-rho**2)


def optbins(x: np.ndarray, maxM: int=150, mode="nbins", alpha=0.025):
    """

    NOTE: Weak performance, study the method !!
    
    Optimal 1D-histogram binning via Bayesian Brute Force search algorithm.
    
    K.H Knuth, 2012, Entropy.
    https://arxiv.org/abs/physics/0605197
    
    Args:
        x     : data points
        maxM  : maximum number of bins
        mode  : "nbins" or "width"
        alpha : outlier protection percentile

    Returns:
        optimal number of bins
    """
    N = len(x)

    # Outlier protection
    lo,hi  = np.percentile(x, alpha/2*100), np.percentile(x, 100*(1-alpha/2))
    ind = (x > lo) & (x < hi)

    # Loop over number of bins and compute (relative) posterior probability
    logp = np.ones(maxM)*(-1E32) # keep it negative for 0-bin

    for M in range(1,maxM):
        n       = np.histogram(x[ind], bins=M)[0]
        part1   = N*np.log(M) + special.gammaln(M/2) - special.gammaln(N+M/2)
        part2   = -M*special.gammaln(1/2) + np.sum(special.gammaln(n+0.5))
        logp[M] = part1 + part2;

    optM = np.argmax(logp)

    if mode == "width":
        return (np.max(x[ind]) - np.min(x[ind])) / optM
    else:
        return optM


def optbins2d(x: np.ndarray, y: np.ndarray, maxM=(40,40), mode="nbins", alpha=0.025):
    """
    
    NOTE: Weak performance, study the method !!

    Optimal 2D-histogram binning via Bayesian Brute Force search algorithm.
    
    K.H Knuth, 2012, Entropy.
    https://arxiv.org/abs/physics/0605197
    
    Args:
        x     : data points
        y     : data points
        maxM  : maximum number of bins per dimension
        mode  : "nbins" or "width"
        alpha : outlier protection percentile
    
    Returns:
        optimal number of bins
    """
    N = len(x)

    if len(x) != len(y):
        raise Exception('optbins2d: len(x) != len(y)')

    # Outlier protection
    x_lo,x_hi  = np.percentile(x, alpha/2*100), np.percentile(x, 100*(1-alpha/2))
    y_lo,y_hi  = np.percentile(y, alpha/2*100), np.percentile(y, 100*(1-alpha/2))

    ind = (x > x_lo) & (x < x_hi) & (y > y_lo) & (y < y_hi)

    # Loop over number of bins and compute (relative) posterior probability
    logp = np.ones(maxM)*(-1E32) # keep it negative for 0-bin

    for Mx in range(1,maxM[0]):
        for My in range(1,maxM[1]):
            n         = np.histogram2d(x=x[ind],y=y[ind], bins=(Mx,My))[0].flatten()
            M         = Mx*My
            part1     = N*np.log(M) + special.gammaln(M/2) - special.gammaln(N+M/2)
            part2     = -M*special.gammaln(1/2) + np.sum(special.gammaln(n+0.5))
            logp[Mx,My] = part1 + part2;

    # Find optimal number of (x,y) bins
    optM = np.unravel_index(logp.argmax(), logp.shape)

    if mode == "width":
        return ((np.max(x[ind]) - np.min(x[ind])) / optM[0],
                (np.max(y[ind]) - np.min(y[ind])) / optM[1])
    else:
        return optM


def test_gaussian():
    """
    Gaussian unit test of the estimators.
    """
    import pytest
    
    
    ## Create synthetic Gaussian data
    for N in [int(1e4)]:
        
        print(f'*************** statistics N = {N} ***************')

        for rho in np.linspace(-0.99, 0.99, 6):
            
            EPS = 0.2

            print(f'<<<rho = {rho:.3f}>>>')

            # Create correlation via 2D-Cholesky
            z1  = np.random.randn(N)
            z2  = np.random.randn(N)
            
            x1  = z1
            x2  = rho*z1 + np.sqrt(1-rho**2)*z2
            
            # ---------------------------------------------------------------
            
            ## Linear correlation
            r,r_CI,prob = pearson_corr(x=x1, y=x2)
            assert  r == pytest.approx(rho, abs=EPS)
            print(f'pearson_corr = {r:.3f}, CI = {r_CI}, p-value = {prob:0.3E}')

            ## Distance correlation (own function)
            dcorr = distance_corr_torch(x=torch.Tensor(x1), y=torch.Tensor(x2)).item()
            print(f'distance_corr_torch = {dcorr:.3f}')

            ## Distance correlation (dcorr package based)
            dc, dc_CI = distance_corr(x=x1, y=x2)
            print(f'distance_corr = {dc:.3f}, CI = {dc_CI}')

            assert dcorr == pytest.approx(dc, abs=1E-3)
            
            ## MI Reference (exact analytic)
            MI_REF = gaussian_mutual_information(rho)
            print(f'Gaussian exact MI = {MI_REF:.3f}')
            
            ## MI with different histogram autobinnings
            automethod = ['Scott2D', 'Hacine2D']
            
            for method in automethod:
                MI, MI_CI = mutual_information(x=x1, y=x2, automethod=method)
                assert MI == pytest.approx(MI_REF, abs=EPS)
                print(f'Histogram     MI = {MI:0.3f}, CI = {MI_CI} ({method})')

            ## Neural MI
            EPS = 0.5 # accept some more tolerance due to fluctuations
            
            neuromethod = ['MINE', 'MINE_EMA', 'DENSITY']
            
            X = torch.Tensor(x1)
            Z = torch.Tensor(x2) 
            
            for losstype in neuromethod:
                
                ## -----------------
                MI,MI_err,model = mine.estimate(X=X, Z=Z, losstype=losstype)
                device = MI.device
                MI     = MI.item()
                MI_err = MI_err.item()
                
                assert MI == pytest.approx(MI_REF, abs=EPS)
                print(f'Neural (.estimate)            MI = {MI:0.3f} +- {MI_err:0.3f} ({losstype})')
                
                ## -----------------
                MI_full = mine.apply_mine(X=X.to(device), Z=Z.to(device), model=model, losstype=losstype)[0].item()
                assert MI_full == pytest.approx(MI_REF, abs=EPS)
                print(f'Neural (.apply_mine)          MI = {MI_full:0.3f} ({losstype})')

                ## -----------------
                MI_batch = mine.apply_mine_batched(X=X.to(device), Z=Z.to(device), model=model, losstype=losstype, batch_size=512).item()
                assert MI_batch == pytest.approx(MI_REF, abs=EPS)
                print(f'Neural (.apply_mine_batched) MI = {MI_batch:0.3f} ({losstype})')
            
            print('')


def test_constant(N=10000):
    """
    Constant input unit test of the estimators.
    """

    import pytest

    EPS = 1E-3

    ### Both ones
    x1 = np.ones(N)
    x2 = np.ones(N)

    r     = pearson_corr(x=x1, y=x2)[0]
    assert   r == pytest.approx(1, abs=EPS)

    dcorr = distance_corr_torch(x=torch.Tensor(x1), y=torch.Tensor(x2)).item()
    assert  dcorr == pytest.approx(0, abs=EPS)
    
    dc, dc_CI = distance_corr(x=x1, y=x2)
    assert  dc == pytest.approx(0, abs=EPS)
    
    # <->
    assert  dcorr == pytest.approx(dc, abs=EPS)
    
    MI      = mutual_information(x=x1, y=x2)[0]
    assert  MI == pytest.approx(0, abs=EPS)

    MI_mine = mine.estimate(X=torch.Tensor(x1), Z=torch.Tensor(x2))[0].item()
    assert  MI_mine == pytest.approx(0, abs=EPS)


    ### Other one zeros
    x1 = np.ones(N)
    x2 = np.zeros(N)
    
    r     = pearson_corr(x=x1, y=x2)[0]
    assert   r == pytest.approx(0, abs=EPS)

    dcorr = distance_corr_torch(x=torch.Tensor(x1), y=torch.Tensor(x2)).item()
    assert  dcorr == pytest.approx(0, abs=EPS)
    
    dc, dc_CI = distance_corr(x=x1, y=x2)
    assert  dc == pytest.approx(0, abs=EPS)
    
    # <->
    assert  dcorr == pytest.approx(dc, abs=EPS)
    
    MI      = mutual_information(x=x1, y=x2)[0]
    assert  MI == pytest.approx(0, abs=EPS)

    MI_mine = mine.estimate(X=torch.Tensor(x1), Z=torch.Tensor(x2))[0].item()
    assert  MI_mine == pytest.approx(0, abs=EPS)


    ### Other one minus ones
    x1 = np.ones(N)
    x2 = (-1)*np.ones(N)
    
    r     = pearson_corr(x=x1, y=x2)[0]
    assert   r == pytest.approx(-1, abs=EPS)

    dcorr = distance_corr_torch(x=torch.Tensor(x1), y=torch.Tensor(x2)).item()
    assert  dcorr == pytest.approx(0, abs=EPS)
    
    dc, dc_CI = distance_corr(x=x1, y=x2)
    assert  dc == pytest.approx(0, abs=EPS)
    
    # <->
    assert  dcorr == pytest.approx(dc, abs=EPS)
    
    MI      = mutual_information(x=x1, y=x2)[0]
    assert  MI == pytest.approx(0, abs=EPS)

    MI_mine = mine.estimate(X=torch.Tensor(x1), Z=torch.Tensor(x2))[0].item()
    assert  MI_mine == pytest.approx(0, abs=EPS)
    
    
    EPS = 5E-2
    
    ### Both uniform random
    x1 = np.random.rand(N)
    x2 = np.random.rand(N)

    r     = pearson_corr(x=x1, y=x2)[0]
    assert   r == pytest.approx(0, abs=EPS)

    dcorr = distance_corr_torch(x=torch.Tensor(x1), y=torch.Tensor(x2)).item()
    assert  dcorr == pytest.approx(0, abs=EPS)
    
    dc, dc_CI = distance_corr(x=x1, y=x2)
    assert  dc == pytest.approx(0, abs=EPS)
    
    # <->
    assert  dcorr == pytest.approx(dc, abs=EPS)
    
    MI      = mutual_information(x=x1, y=x2)[0]
    assert  MI == pytest.approx(0, abs=EPS)
    
    MI_mine = mine.estimate(X=torch.Tensor(x1), Z=torch.Tensor(x2))[0].item()
    assert  MI_mine == pytest.approx(0, abs=EPS)
    



#if __name__ == "__main__":
    #test_gaussian()
    #test_constant()



"""
def test_data():

    # Read toy dataset
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00451/dataR2.csv")
    #target = df['Classification']

    df.drop(['Classification'], axis=1, inplace=True)

    # Build MI matrix for each pair of features
    D = df.shape[1]
    MI   = np.zeros((D,D))
    MI_A = np.zeros((D,D))
    MI_M = np.zeros((D,D))

    for i,col_i in enumerate(df):
        for j,col_j in enumerate(df):

            MI[i,j]   = mutual_information(x=df[col_i], y=df[col_j], normalized=None)
            MI_A[i,j] = mutual_information(x=df[col_i], y=df[col_j], normalized='additive')
            MI_M[i,j] = mutual_information(x=df[col_i], y=df[col_j], normalized='multiplicative')

    # Print out
    print('>> Raw Mutual Information')
    print(pd.DataFrame(MI,   columns = df.columns, index = df.columns))
    print('')
    print('>> Additively Normalized Mutual Information')
    print(pd.DataFrame(MI_A, columns = df.columns, index = df.columns))
    print('')
    print('>> Multiplicatively Normalized Mutual Information')
    print(pd.DataFrame(MI_M, columns = df.columns, index = df.columns))
"""
