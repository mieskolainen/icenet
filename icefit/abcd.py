# Minimal "ABCD" estimator and its confidence intervals
#
# Run tests with: pytest icefit/abcd.py -rP
#
# m.mieskolainen@imperial.ac.uk, 2022

import copy
import numpy as np
import scipy
import scipy.stats as stats
import numba

from icefit.cortools import prc_CI


def ABCD_eq(b,c,d):
    """
    Basic estimator formula for count in 'A' (signal domain)
        
    Independent (X,Y) gives us the definition A = B x C / D
    
    Y
    |C | A
    |-----
    |D | B
    0------> X
    
    """
    return b * c / d


def ABCD_err(b,c,d, method='errorprop', N=int(1e5), alpha=0.32, lrange=5):
    """
    ABCD uncertainty (confidence interval) methods
    
    DEFINITION: A = B x C / D

    Args:
        b,c,d  : input value  
        method : 'errorprop', 'poisson', 'bootstrap', 'likelihood'
        N      : number of random samples for sampling based methods
        alpha  : confidence level ('errorprop' fixed at CL68)
        lrange : likelihood scan range
    
    Returns:
        confidence interval array (lower,upper)
    """
    
    # Analytical error propagation (1st order Taylor expansion)
    if   method == 'errorprop':
        sigma = np.sqrt( (c/d)**2 * b + (b*c/d**2)**2 * d + (b/d)**2 * c)
        A = ABCD_eq(b=b,c=c,d=d)

        # Return symmetric (Gaussian like) 1 sigma (68%) confidence interval
        return np.array([A - sigma, A + sigma])

    # Poisson MC error propagation
    elif method == 'poisson_prc':

        B_new = np.random.poisson(lam=b, size=N)
        C_new = np.random.poisson(lam=c, size=N)
        D_new = np.random.poisson(lam=d, size=N)
        A     = ABCD_eq(b=B_new, c=C_new, d=D_new)

        return prc_CI(x=A, alpha=alpha)

    # Efron's percentile bootstrap
    elif method == 'bootstrap_prc':

        # Re-generate input data (N.B. this could be generalized to weighted data)
        T = [1,2,3] # integer labels ~ B,C,D
        data = np.concatenate((T[0]*np.ones(int(b)), T[1]*np.ones(int(c)), T[2]*np.ones(int(d))), axis=None)

        # Generate bootstrap samples
        A_new    = np.zeros(N)
        for i in range(N):
            ind      = np.random.choice(range(len(data)), size=len(data), replace=True)
            bs       = data[ind]
            A_new[i] = ABCD_eq(b=np.sum(bs==T[0]), c=np.sum(bs==T[1]), d=np.sum(bs==T[2]))

        return prc_CI(x=A_new, alpha=alpha)

    elif method == 'likelihood':

        def optfunc(theta):
            return ABCD_2NLL(B=b, C=c, D=d, mu=theta[0], mu_B=theta[1], mu_D=theta[2])

        # Initial guess and optimize (reservation for more general models)
        theta0 = np.array([ABCD_eq(b=b, c=c, d=d) / b, b, d])
        res    = scipy.optimize.minimize(optfunc, theta0, method='Nelder-Mead', tol=1e-6)
        
        # ----------------------------------------------------------------
        # Likelihood scan over A = mu * mu_B product plane
        x0 = np.linspace(res['x'][0]/lrange, res['x'][0]*lrange, int(1e3))
        x1 = np.linspace(res['x'][1]/lrange, res['x'][1]*lrange, int(1e3))

        # keep (mu_D = theta[2]) to its optimal value (~profiled)       
        theta = np.array(copy.deepcopy(res['x']))
        chi2cut = stats.chi2.ppf(1 - alpha, df=1) # note NDF=1 (we have a product)

        values = []
        for i in range(len(x0)):
            for j in range(len(x1)):
                theta[0] = x0[i]
                theta[1] = x1[j]
                if optfunc(theta) < res['fun'] + chi2cut:
                    values.append(theta[0] * theta[1]) # A = mu * mu_B

        values = np.array(values)

        return np.array([np.min(values), np.max(values)])

    else:
        raise Exception(f'ABCD_err: Unknown method {method}')   


@numba.njit
def ABCD_2NLL(B,C,D, mu, mu_B, mu_D, EPS=1e-32):
    """
    ABCD estimators negative log-likelihood function
    
    DEFINITION: A = B x C / D
    
    Args:
        B,C,D          : Measured event counts
        mu, mu_B, mu_D : Free parameters of the likelihood function
        
        (N.B. here number of measurements == number of free parameters)
    
    Model relation:
        Ntot = mu*mu_B + mu_B + mu*mu_D + mu_D
             = (A) + (B) + (C) + (D)
    
    See e.g. https://twiki.cern.ch/twiki/pub/Main/ABCDMethod/ABCDGuide_draft18Oct18.pdf    
    
    Returns:
        -2logL
    """

    N_blindtot = mu_B + mu*mu_D + mu_D

    if N_blindtot < EPS:
        N_blindtot = EPS

    LL = (B + C + D) * np.log(N_blindtot) - N_blindtot \
        + B * np.log(mu_B / N_blindtot) \
        + C * np.log((mu*mu_D) / N_blindtot) \
        + D * np.log(mu_D / N_blindtot)

    return -2*LL


def test_abcd():
    
    import pytest

    EPS = 1.5


    # ---------------------------
    # INPUT DATA
    B = 100
    C = 5
    D = 50
    # ---------------------------

    A = ABCD_eq(b=B, c=C, d=D)
    
    # Uncertainty estimates
    alpha  = 1 - 0.68 # confidence level


    print(f'INPUT')
    print(f'  B = {B}, C = {C}, D = {D}')
    print(f'')
    print(f'ALGEBRAIC ESTIMATE')
    print(f'  A = B x C / D = {A:0.1f}')
    print(f'')
    print(f'CONFIDENCE INTERVAL on A (alpha = {alpha:0.2f})')
    
    methods = ['errorprop', 'poisson_prc', 'bootstrap_prc', 'likelihood']
    for method in methods:
        
        CI = ABCD_err(b=B, c=C, d=D, method=method, alpha=alpha)
        print(f'  {method:15s}  CI = [{CI[0]:0.2f}, {CI[1]:0.2f}]')
        
        # Test
        assert  5  == pytest.approx(CI[0], abs=EPS)
        assert  15 == pytest.approx(CI[1], abs=EPS)

