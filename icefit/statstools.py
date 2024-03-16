# Statistical tests and tools
#
# pytest icefit/statstools.py -rP
#
# m.mieskolainen@imperial.ac.uk, 2023

import numpy as np
import numba
import copy
import scipy.stats as stats


def covmat2corrmat(C, EPS=1e-12):
    """
    Covariance matrix to correlation matrix

    Args:
        C: Covariance matrix

    Returns:
        Correlation matrix
    """
    v       = np.sqrt(np.diag(C))
    outer_v = np.outer(v, v)
    corr    = C / (outer_v + EPS)
    corr[C == 0] = np.nan # Not defined

    return corr

def correlation_matrix(X, weights=None, return_cov=False):
    """
    Compute a (sample weighted) correlation or covariance matrix
    
    Args:
        X:           Data matrix (rows as sample points, columns as variables)
        weights:     Weights per sample
        return_cov:  Return covariance matrix (not correlation)
    
    Returns:
        correlation matrix
    """
    
    if weights is None:
        C = np.cov(X, rowvar=False)

        if return_cov: # Return covariance
            return C

        # Turn into a correlation matrix
        C = covmat2corrmat(C=C)

    else:
        # Zero-mean each column (variable), repeat weights
        Q = copy.deepcopy(X)
        W = np.zeros(X.shape)
        for j in range(Q.shape[1]):
            Q[:,j] = Q[:,j] - np.average(X[:,j], weights=weights)
            W[:,j] = weights
        
        # Compute a weighted covariance matrix
        QW = Q*W
        C  = QW.T.dot(QW) / W.T.dot(W)

        if return_cov: # Return covariance
            return C

        # Turn into a correlation matrix
        C = covmat2corrmat(C=C)

    return C


def error_on_mu(sigma, n):
    """ Standard error estimate on a sample mean.
    https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf
    
    Args:
        sigma : sample standard deviation
           n  : sample size
    """
    return sigma / np.sqrt(n)


def error_on_std(sigma, n):
    """
    Standard error estimate on a sample standard deviation.
    https://web.eecs.umich.edu/~fessler/papers/files/tr/stderr.pdf
    Args:
        sigma : sample standard deviation
        n     : sample size
    """
    Kn = np.sqrt((n-1)/2) * np.exp(special.loggamma((n-1) / 2) - special.loggamma(n/2))
    Vn = 2 * ((n-1)/2 - (special.gamma(n/2) / special.gamma((n-1)/2))**2)
    return sigma * Kn * np.sqrt(Vn / (n-1))


def geom_mean_2D(x, y, x_err, y_err, flip_vertical=False):
  """
  Geometric 2D mean of x and y (understood e.g. as 1D histograms) element wise
  and its error propagation based uncertainty (x,y taken also independent elem. wise)
  
  Math note: outer(u,v)_ij = u_i v_j, with outer(u,v) = uv^T
  
  Args:
    x, y         : values        (numpy arrays)
    x_err, y_err : uncertainties (numpy arrays)
    flip_vertical: flip the values over the vertical (y) dimension
  
  Returns:
    z, z_err     : 2D-matrix (with rank 1) and its uncertainty
  """
  z     = np.sqrt(np.outer(y, x))
  z_err = 0.5 * np.sqrt(np.outer(y, x_err**2/x) + np.outer(y_err**2/y, x))
  
  if flip_vertical:
    z     = np.flip(z,0)
    z_err = np.flip(z_err,0)
  
  return z, z_err


def tpratio_taylor(x,y, x_err,y_err, xy_err=0.0):
    """
    Error propagation on a ratio

    r = x/(x+y)
    
    If x and y are e.g. counts from stat. independent histograms,
    then xy_err = 0 (~ tag & probe style ratios).
    
    Args:
        x,y         : values
        x_err,y_err : uncertainties
        xy_err      : covariance (err_x * err_y * rho_xy)
    
    Returns:
        Error propagated ratio error
    """
    dx =  y/(x + y)**2
    dy = -x/(x + y)**2

    r2 = dx**2*x_err**2 + dy**2*y_err**2 + 2*dx*dy*xy_err
    
    return np.sqrt(r2)


def prod_eprop(A, B, sigmaA, sigmaB, sigmaAB=0):
    """
    Error propagation (Taylor expansion) of a product A*B

    Args:
        A       : Value A 
        B       : Value B
        sigmaA  : 1 sigma uncertainty on A
        sigmaB  : 1 sigma uncertainty on B
        sigmaAB : Covariance of A,B

    Returns:
        Uncertainty
    """
    f = A * B
    return np.sqrt(f**2*((sigmaA/A)**2 + (sigmaB/B)**2 + 2*sigmaAB/(A*B)))


def ratio_eprop(A, B, sigmaA, sigmaB, sigmaAB=0):
    """
    Error propagation (Taylor expansion) of a ratio A/B

    Args:
        A       : Value A 
        B       : Value B
        sigmaA  : 1 sigma uncertainty on A
        sigmaB  : 1 sigma uncertainty on B
        sigmaAB : Covariance of A,B
    
    Returns:
        Uncertainty
    """
    f = A / B
    return np.sqrt(f**2*((sigmaA/A)**2 + (sigmaB/B)**2 - 2*sigmaAB/(A*B)))


def weighted_binomial_err(b1: float, b2: float, e1: float, e2: float):
    """
    Binomial counting efficiency error with weights
    
    For the reference, see:
    https://root.cern/doc/master/TH1_8cxx_source.html#l02962
    
    If e1 ~ sqrt(b1) and e2 ~ sqrt(b2), one should recover results of this
    function by using the proper covariance with:
    
    prodratio_eprop(A=b1, B=b2, sigma_A=e1, sigma_B=e2, sigmaAB=b1, mode='ratio)
    
    N.B. If b1 == b2, will return 0
    
    Args:
        b1: binomial numerator (e.g. count or sum of weights)
        b2: binomial denominator
        e1: error on the numerator (e.g. \sqrt(\sum_i(w_i^2))
        e2: error on the denominator
    """
    if b1 == b2:
        return 0.0
    
    r   = b1 / b2
    err = np.sqrt(np.abs( ((1.0 - 2.0*r))*e1**2 + (r*e2)**2) / (b2**2))
    
    return err


def columnar_mask_efficiency(num_mask: np.ndarray, den_mask: np.ndarray,
                             y_true: np.ndarray, y_target: int, weights: np.ndarray):
    """
    Masked columnar selection efficiency with event weights and classes
    
    Args:
        num_mask: numerator mask vector (Nx1, bool)
        den_mask: denominator mask      (Nx1, bool)
        y_true:   label vector          (Nx1, int)
        y_target: target label          (int)
        weights:  event weights         (Nx1, float)
        
    Returns:
        binomial efficiency, weighted binomial stat. error
    """
     
    w1 = weights[y_true == y_target]*num_mask[y_true == y_target]
    w2 = weights[y_true == y_target]*den_mask[y_true == y_target]
    
    b1 = np.sum(w1)
    b2 = np.sum(w2)
    e1 = np.sqrt(np.sum(w1**2))
    e2 = np.sqrt(np.sum(w2**2))
    
    eff     = b1 / b2
    eff_err = weighted_binomial_err(b1=b1, b2=b2, e1=e1, e2=e2)
    
    return eff, eff_err


def ADS(s, b, sigma_b, EPS=1e-6):
    """
    Asimov discovery significance with background uncertainty
    https://indico.cern.ch/event/708041/papers/3272129/files/9437-acat19DirectOpti.pdf
    
    when sigma_b --> 0, this gives np.sqrt(2*((s+b)*np.log(1.0 + s/b) - s))
    
    Args:
        s:        expected signal count
        b:        expected background count
        sigma_b:  error on the background estimate
        EPS:      regularization constant
    """
    if s + b == 0:
        return 0
    if b == 0:
        b = EPS
    if sigma_b == 0:
        sigma_b = EPS

    T1 = ((s+b)*(b+sigma_b**2)) / (b**2+(s+b)*sigma_b**2)
    T2 = 1 + (sigma_b**2*s) / (b*(b+sigma_b**2))
    
    return np.sqrt(2*((s+b)*np.log(T1) - b**2/(sigma_b**2) * np.log(T2)))


def welch_ttest(X1, X2, s1, s2, n1, n2):
    """
    Welch's two sample t-test for normally distributed variables.
    https://en.wikipedia.org/wiki/Welch%27s_t-test
    
    Args:
        X1 : Mean of sample 1
        X2 : Mean of sample 2
        s1 : Standard deviation of sample 1
        s2 : Standard deviation of sample 2
        n1 : Counts in sample 1
        n2 : Counts in sample 2 

    Returns:
        t : test statistic
        p : p-value from t-distribution
    """

    # Satterthwaite Approximation
    sdelta = np.sqrt(s1**2/n1 + s2**2/n2)

    # Test statistic
    t = (X1 - X2) / sdelta

    # Degrees of freedom
    nu1 = n1-1 
    nu2 = n2-1
    df  = np.ceil((s1**2/n1 + s2**2/n2)**2 / (s1**4/(n1**2*nu1) + s2**4/(n2**2*nu2)))

    # p-value, two-tailed test gives factor 2
    p = (1 - stats.t.cdf(np.abs(t), df=df)) * 2

    return t,p


def p2zscore(p):
    """
    p-value to z-score (standard normal sigmas)
    
    Example:
        p-value is divided by two such that
        p = 0.05 gives z_{1-\\alpha/2} = z_{1-0.025} = 1.96
    """
    z = np.abs( stats.norm.ppf(p / 2) )

    if np.isinf(z): # overflow
        z = 99
    return z


def clopper_pearson_err(k, n, CL=[0.025, 0.975]):
    """ Clopper-Pearson binomial proportion confidence interval.
    Below, beta.ppf (percent point functions) returns inverse CDF for quantiles.
    
    Args:
        k  : observed success counts
        n  : number of trials
        CL : confidence levels [lower, upper]
    Returns:
        confidence interval [lower, upper]
    """
    # Special case must be treated separately
    if   k == 0:
        lower = 0
        upper = 1 - (1-CL[1])**(1/n)

    # Special case must be treated separately
    elif k == n:
        lower = CL[0]**(1/n)
        upper = 1

    # Normal case
    else:
        lower = stats.beta.ppf(q=CL[0], a=k, b=n-k+1)
        upper = stats.beta.ppf(q=CL[1], a=k+1, b=n-k)

    return np.array([lower, upper])


def poisson_ratio(k1,k2, CL=np.array([0.025, 0.975])):
    """
    Poisson ratio uncertainty via conditional ratio
    theta = lambda1 / (lambda1 + lambda2)
    
    See e.g. "Confidence limits on the ratio of two Poisson variables, 1974"
    
    E[k1] = lambda1 (Poisson mean)
    E[k2] = lambda2 (Poisson mean)
    
    Args:   
        k1 : observations k1
        k2 : observations k2
        CL : 2-array of lower and upper quantiles

    Returns:
        ratio confidence interval endpoints
    """

    theta = clopper_pearson_err(k=k1, n=k1+k2, CL=CL)

    if (1-theta).all() > 0:
        R = theta / (1 - theta)
    else:
        R = np.array([0,0])

    return R


def poisson_tail(k1, k2):
    """
    Single Poisson CDF tail integral. Counts 2 works as the reference.
    Args:
        k1 : poisson counts 1
        k2 : poisson counts 2 (reference sample)
    Returns:
        p value
    """
    P = stats.poisson.cdf(k1, mu=k2)
    if k1 > k2:
        return 1 - P
    else:
        return P


def mc_extreme_npdf(x, N, mu=0, sigma=1, trials=int(1e6)):
    """
    Extreme value distribution for normal pdf via Monte Carlo
    
    Args:
        x      : value to evaluate the extreme value pdf upper tail integral
        mu     : mean of normal pdf
        sigma  : std of normal pdf
        N      : sample size
        trials : number of Monte Carlo samples, as large as possible CPU wise
    
    Returns:
        p_value : probability for the value x or more extreme
        maxvals : full MC distribution of extreme values obtained
    """

    sample  = np.random.normal(mu, sigma, size=(trials, N))
    maxvals = np.amax(sample, 1) # max per each row (trial)
    p_value = np.sum(maxvals >= x) / trials

    return p_value, maxvals


def mc_extreme_multivariate_npdf(x, mu, cov, trials=int(1e6)):
    """
    Extreme value distribution for a multivariate normal pdf via Monte Carlo,
    e.g. for "correlated measurements or bins" -- casted as a univariate problem
    such that maximum of components is taken (not a "vector maximum")
    
    Args:
        x       : scalar value to evaluate the extreme value pdf upper tail integral
        mu      : mean vector (N)
        cov     : covariance matrix (NxN)
        trials  : number of Monte Carlo samples, as large as possible CPU wise
    
    Returns:
        p_value : probability for the value x or more extreme
        maxvals : full MC distribution of extreme values obtained
    """

    sample  = np.random.multivariate_normal(mean=mu, cov=cov, size=trials)
    maxvals = np.amax(sample, 1) # max per each row (trial)
    p_value = np.sum(maxvals >= x) / trials

    return p_value, maxvals


def analytical_extreme_npdf(N, mu=0, sigma=1):
    """
    Analytical expectation (mean) value approximation (based on an expansion)
    for a normal pdf max (extreme) values with a sample size of N.
    
    References:
        https://en.wikipedia.org/wiki/Extreme_value_theory
        https://en.wikipedia.org/wiki/Gumbel_distribution
    """

    # Ref: https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution
    # y = 0.5772156649 # Euler–Mascheroni constant
    # return np.sqrt(np.log(N**2/(2*np.pi*np.log(N**2/(2*np.pi)) ) ) ) *(1+y/np.log(N))

    # Ref: Youtube, <From one extreme to another: the statistics of extreme events, Jon Keating, Oxford>
    return mu + sigma*np.sqrt(2*np.log(N)) - 0.5*sigma*(np.log(np.log(N))) / (np.sqrt(2*np.log(N)))


def cormat2covmat(R,s):
    """ Convert a correlation matrix to a covariance matrix.
    
    Args:
        R : correlation matrix (NxN)
        s : vector of standard deviations (N)
    Returns:
        covariance matrix
    """

    if np.any(s <= 0):
        raise Exception('cormat2covmat: s vector elements must be positive')

    if np.any(np.abs(R) > 1):
        raise Exception('cormat2covmat: R matrix elements must be between [-1,1]')

    return np.diag(s) @ R @ np.diag(s)


def test_extreme_npdf():

    mu    = 0 # standard normal
    sigma = 1

    x = 3.0   # we test for "x sigma deviation"
    N = 100   # data sample size

    # -------
    print('Independent standard normal PDF')

    p_value, maxvals = mc_extreme_npdf(x=x, N=N, mu=mu, sigma=sigma)

    print(f' Extreme value p-value for x >= {x} is {p_value:0.2E} [with N = {N}]')
    print(f' Monte Carlo expectation: <extreme> = {np.mean(maxvals):0.3f}')
    print(f' Analytical expectation:  <extreme> = {analytical_extreme_npdf(N=N, mu=mu, sigma=sigma):0.3f}')
    
    # -------
    # Uncorrelated case, will return the same as independent Gaussians
    print('\nIndependent standard N-dim normal PDF')

    mu  = np.zeros(N) # Mean vector
    cov = np.eye(N)   # Covariance matrix

    p_value, maxvals = mc_extreme_multivariate_npdf(x=x, mu=mu, cov=cov)

    print(f' Extreme value p-value for x >= {x} is {p_value:0.2E}')
    print(f' Monte Carlo expectation: <extreme> = {np.mean(maxvals):0.3f}')

    # -------
    # Correlated bins
    ccoef = 0.7
    print(f'\nCorrelated N-dim normal PDF [ccoef = {ccoef:0.3f}]')

    mu  = np.zeros(N)             # Mean vector

    R   = np.ones((N,N)) * ccoef  # Correlation matrix
    np.fill_diagonal(R, 1.0)
    s   = np.ones(N)              # Vector of standard deviations
    cov = cormat2covmat(R=R,s=s)  # --> Covariance matrix

    p_value, maxvals = mc_extreme_multivariate_npdf(x=x, mu=mu, cov=cov)
    
    print(f' Extreme value p-value for x >= {x} is {p_value:0.2E}')
    print(f' Monte Carlo expectation: <extreme> = {np.mean(maxvals):0.3f}')


def test_efficiency_ratio(EPS=1e-6):

    import pytest
    
    # Ratio error, where b1 is the numerator (number of successes)
    # in a binomial ratio and b2 is the denominator (total number of trials)
    # where we allow both numerator and denominator to fluctuate aka take
    # here Poisson errors for them.
    
    b1 = 5
    b2 = 20
    e1 = np.sqrt(b1)
    e2 = np.sqrt(b2)

    eff       = b1 / b2
    eff_err_A = weighted_binomial_err(b1=b1, b2=b2, e1=e1, e2=e2)
    eff_err_B = ratio_eprop(A=b1, B=b2, sigmaA=e1, sigmaB=e2, sigmaAB=b1)
    eff_err_C = tpratio_taylor(x=b1, y=b2-b1, x_err=e1, y_err=np.sqrt(b2 - b1))
    
    print(f'eff = {eff}')
    print(f'err_eff = {eff_err_A:0.6g} (A)')
    print(f'err_eff = {eff_err_B:0.6g} (B)')
    print(f'err_eff = {eff_err_C:0.6g} (C)')

    assert   eff_err_A == pytest.approx(eff_err_B, abs=EPS)
    assert   eff_err_B == pytest.approx(eff_err_C, abs=EPS)
    assert   eff_err_A == pytest.approx(eff_err_C, abs=EPS)


def test_ratios():
    """
    Test function
    """

    # Quantile values
    q68 = np.array([0.16, 0.84])
    q95 = np.array([0.025, 0.975])

    # --------------------------------------------
    # INPUT TEST DATA

    # Observed Poisson counts
    k_obs = np.array([17, 10, 100, 20, 100, 30, 400, 400])

    # Predicted Poisson counts
    k_pre = np.array([7, 15, 101, 25, 105, 10, 380, 200])

    # Poisson uncertainties
    s_obs = np.sqrt(k_obs)
    s_pre = np.sqrt(k_pre)
    # --------------------------------------------

    for i in range(len(k_obs)):

        print(f'>> Case {i} pre/obs = {k_pre[i]} / {k_obs[i]} = {k_pre[i] / k_obs[i]:0.3f}')

        # Conditional Poisson ratios    
        ppois68 = poisson_ratio(k1=k_pre[i], k2=k_obs[i], CL=q68)
        ppois95 = poisson_ratio(k1=k_pre[i], k2=k_obs[i], CL=q95)

        print(f'Poisson ratio uncertainty CI68: {ppois68}')
        print(f'Poisson ratio uncertainty CI95: {ppois95}')
        
        # Systematic residual from target ratio = 1 at CI95 level
        covers = lambda n,a,b : (n - a)*(n - b) <= 0
        res = 0 if covers(1, ppois95[0], ppois95[1]) else np.min(np.abs(ppois95 - 1))
        print(f'Systematic residual: {res:0.2f}')
        
        # Single sample Poisson tail integral
        ppois = poisson_tail(k1=k_pre[i], k2=k_obs[i])
        print(f'Poisson tail p = {ppois:0.3E} ({p2zscore(ppois):0.2f} sigmas)')

        # Error propagation
        errAB = ratio_eprop(A=k_pre[i], sigmaA=s_pre[i], B=k_obs[i], sigmaB=s_obs[i])
        PRE_over_OBS = k_pre[i] / k_obs[i]
        print(f'  error propagated rel.uncertainty on the ratio: err / (pre/obs) = {errAB / PRE_over_OBS:0.3f}')

        print('')
