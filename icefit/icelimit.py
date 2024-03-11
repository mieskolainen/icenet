# Standalone signal upper limit calculus (MC and Asymptotic)
# 
# Run with: pytest icefit/icelimit.py -rP
#
# m.mieskolainen@imperial.ac.uk, 2023

import numpy as np
import scipy.optimize
from scipy import stats
from tqdm import tqdm
from scipy.stats import ncx2,norm
import matplotlib.pyplot as plt

#import mystic

from icefit import lognormal


def convolve_systematics(n, syst_error, num_toys, error_type='log-normal-mean'):
    """
    Produce an array of distribution convoluted values
    
    Args:
        n:           expected count value
        syst_error:  error size (e.g. standard deviation)
        num_toys:    number of MC samples
        error_type:  type of distribution

    Returns:
        array of counts
    """
    if   error_type == 'flat':
        arr = np.random.uniform(low=n-syst_error, high=n+syst_error, size=num_toys)
    elif error_type == 'normal':
        arr = np.random.normal(loc=n, scale=syst_error, size=num_toys)
    elif error_type == 'log-normal-mean':  
        arr = lognormal.rand_lognormal(m=n, v=syst_error**2, N=num_toys, inmode='mean')
    elif error_type == 'log-normal-median':  
        arr = lognormal.rand_lognormal(m=n, v=syst_error**2, N=num_toys, inmode='median')
    else:
        raise Exception(f'convolve_systematics: unknown error_type "{error_type}"')
    
    return arr


def LL_splusb(n, s,b, EPS=1e-15):
    """
    Single channel Poisson log-likelihood
    
    Args:
        n: counts
        s: expected signal count (mean)
        b: expected background count (mean)
    """
    a = s + b
    
    # - np.log(np.math.factorial(n)) is cancelled in the log-likelihood ratio
    return -a + np.log(np.clip(a, EPS, None))*n


def CL_single(mu, s_hypothesis, bg_expected, observed, num_toys_pdf=int(1E4),
    s_syst_error=None,  s_syst_type='flat',
    bg_syst_error=None, bg_syst_type='flat', use_CLs=True):
    """
    One tailed signal upper limits using "toy" Monte Carlo.
    
    A single channel counting experiment case with "quasi-Bayesian" style systematics
    convoluted in via MC.
    
    Args:
        mu:                Signal strength parameter
        s_hypothesis:      Number of predicted signal events (with mu = 1)
        bg_expected:       Number of expected background events
        observed:          Number of observed events
        num_toys_pdf:      Number of MC samples to construct PDFs
        use_CLs:           Return "CLs" type limits
        
        (s)bg_syst_error:  Systematic error on estimated (signal) background
        (s)bg_error_type:  Systematic uncertainty perturbation type (flat, normal, log-normal)
    
    Return:
        p-value
    
    References:
        For the "CLs" approach motivation, see: A. Read, "Presentation of search results: the CLs technique"
        https://en.wikipedia.org/wiki/CLs_method_(particle_physics)
    """

    if observed < 0: observed = 0 # Protect unphysical input (e.g. NLO MC)

    # Systematics convoluted in here with MC
    if bg_syst_error != None:
        b_arr = convolve_systematics(n=bg_expected, syst_error=bg_syst_error,
            error_type=bg_syst_type, num_toys=num_toys_pdf)
    else:
        b_arr = np.ones(num_toys_pdf) * bg_expected

    if s_syst_error != None:
        s_arr = convolve_systematics(n=s_hypothesis, syst_error=s_syst_error,
            error_type=s_syst_type, num_toys=num_toys_pdf)
    else:
        s_arr = np.ones(num_toys_pdf) * (mu * s_hypothesis)
    # -----------------------------------------------

    s_arr[s_arr < 0] = 0 # Protection for Poisson sampling
    b_arr[b_arr < 0] = 0

    s_toys   = np.random.poisson(s_arr) # Sample according to Poisson mean given by arr
    b_toys   = np.random.poisson(b_arr) # --|--

    # -------------------------------
    # We use pure MC PDF-based "ordering principle" here (no likelihood ratio involved)

    ## Prob{s+b}(X <= X_obs), Prob{b}(X <= X_obs)
    # [important to have less and EQUAL to, e.g. for the zero case]
    p_splusb = np.sum(s_toys+b_toys <= observed) / num_toys_pdf
    p_b      = np.sum(b_toys        <= observed) / num_toys_pdf

    if use_CLs:
        return (p_splusb / p_b) if p_b > 0 else p_splusb
    else:
        return p_splusb


def CL_single_asymptotic(mu, s_hypothesis, bg_expected, observed):
    """
    One tailed signal upper limits using asymptotic (Gaussian) approximation.
    
    A single channel counting experiment case (without systematics --> requires profile likelihood)
    
    Args:
        mu:              Signal strength parameter
        s_hypothesis:    Number of predicted signal events (with mu = 1)
        bg_expected:     Number of expected background events
        observed:        Number of observed events
    
    Return:
        p-value
    
    References:
        Cowan et al, "Asymptotic formulae for likelihood-based tests of new physics",
        https://arxiv.org/abs/1007.1727

        https://indico.cern.ch/event/126652/contributions/1343592/
            attachments/80222/115004/Frequentist_Limit_Recommendation.pdf

        https://arxiv.org/abs/2102.04275
    """

    if observed < 0: observed = 0 # Protect unphysical input (e.g. NLO MC)

    # Maximum likelihood estimate
    shat = observed - bg_expected

    # Likelihood-ratio test statistic
    if shat < (mu * s_hypothesis):
        LL_mu_numer = LL_splusb(n=observed, s=(mu * s_hypothesis), b=bg_expected)
        LL_mu_denom = LL_splusb(n=observed, s=shat, b=bg_expected)

        q_mu = -2 * (LL_mu_numer - LL_mu_denom)
    else:
        q_mu = 0
    
    # Asymptotic p-value
    p_mu = 1 - norm.cdf(np.sqrt(q_mu))

    return p_mu, q_mu


def asymptotic_uncertainty_CLs(mu_up, q, alpha, N):
    """
    Asymptotic (Asimov) +-N sigma values around median expected
    N = 0 gives the median.

    Type "CLs", keeps the minus sigma tail on positive side of mu.
    See reference: "Frequentist_Limit_Recommendation.pdf"
    
    Args:
        mu_up:  iteratively solved median expected upper limit value
        q:      test statistic value corresponding to mu_up
        alpha:  confidence level at which mu_up was extracted
        N:      array of sigmas e.g. [-2,-1,0,1,2]
    """
    sigma = np.sqrt(mu_up**2 / q)
    out   = sigma * (norm.ppf(1 - alpha * norm.cdf(N)) + N)

    #print(f'{mu_up} || {out[2]}')

    return out


def CL_single_compute(s_hypothesis, bg_expected, observed=None, s_syst_error=None, bg_syst_error=None,
    method='toys', alpha=0.05, num_toys_pdf=int(1E4), num_toys_obs=int(1E3), mu_bounds=(1e-5, 1E4),
    mu_init=1.0, max_restart=1000, func_tol=1E-5, optimize_method='golden', maxiter=10000):
    """
    Main solver function for the upper limit computation on the signal strength parameter mu.
    
    Args:
        s_hypothesis:     expected signal count (with mu = 1)
        bg_expected:      expected background count
        observed:         observed event count, if None then observed = bg_expected
        method:           'toys' or 'asymptotic'
        num_toys:         number of MC samples for toys
        alpha:            confidence level for the upper limit (e.g. 0.05 ~ CL95 upper limits)
        s_syst_error:     systematic error size on signal
        bg_syst_error:    systematic error size on background
    
    Optimization args:
        mu_bounds:        bounds on mu-value to obey
        mu_init:          initial value
        max_restart:      maximum number of restarts
        func_tol:         quadratic loss tolerance
        optimize_method:  method to use for the optimization, 'brent' or 'golden'
        maxiter:          maximum number of iterations

    Returns:
        upper limits [-2 sigma, 1 sigma, median, +1 sigma, +2 sigma] at the level alpha
    """

    # To find the expected limit, we insert in the Asimov data which is the expected
    # background (with no fluctuations) with the signal = 0.
    if   observed == None:
        X = bg_expected

    # Observed
    else:
        X = observed

    ## Cost function for the limit inversion (i.e. finding out value of r which corresponds to alpha)
    def func(mu):
        mu_value = mu[0] if type(mu) is list else mu # if the optimization function uses lists
        
        if   method == 'toys':
            p = CL_single(mu=mu_value, s_hypothesis=s_hypothesis, observed=X_input,
                        bg_expected=bg_expected, s_syst_error=s_syst_error, bg_syst_error=bg_syst_error,
                        num_toys_pdf=num_toys_pdf)

        elif method == 'asymptotic':
            p, q_mu = CL_single_asymptotic(mu=mu_value, s_hypothesis=s_hypothesis,
                bg_expected=bg_expected, observed=X_input)  
        else:
            raise Exception(__name__ + '.CL_single_compute: Unknown computation method')

        return (p - alpha)**2 # quadratic loss

    optimizer_param = {'method': optimize_method, 'tol': 1E-8, 'options': {'maxiter': maxiter}}

    ## Compute limit inversion
    repeat = num_toys_obs if (method == 'toys' and observed == None) else 1
    mu_up  = np.nan * np.ones(repeat)

    i = 0
    trials = 0
    while i < repeat:

        # Generate toy data for the expected limits
        X_input = np.random.poisson(X) if (method == 'toys' and observed == None) else X

        # Minimize
        out = scipy.optimize.minimize_scalar(fun=func, **optimizer_param)

        if out['x'] > mu_bounds[0] and out['x'] < mu_bounds[1] and out['fun'] < func_tol:
            mu_up[i] = out['x']
            i += 1
        else:
            trials += 1
        
        if trials > max_restart:
            mu_up[i] = np.nan
            i += 1
            trials = 0
            print(__name__ + '.CL_single_compute: Minimizer problem; set NaN (check parameter and input feasibility)')

    # Compute final results for mu_up(at the level alpha) +- 1(2) sigma uncertainties
    if observed is None:
        if method == 'asymptotic':
            _, q   = CL_single_asymptotic(mu=mu_up[0], s_hypothesis=s_hypothesis, bg_expected=bg_expected, observed=X)  
            output = asymptotic_uncertainty_CLs(mu_up=mu_up[0], q=q, alpha=alpha, N=np.array([-2,-1,0,1,2]))
        else:
            output = np.array([np.percentile(mu_up, quant) for quant in [2.5, 16, 50, 84, 97.5]])
    else:
        output = mu_up

    return output


def test_limits_unit_test(EPS=0.05):
    """
    Simple unit tests
    """

    # Test counts
    s_hypothesis = 500.0
    bg_expected  = 1500.0
    observed     = bg_expected + s_hypothesis + 10

    expected_results = {'asymptotic': None, 'toys': None} 
    observed_results = {'asymptotic': None, 'toys': None}

    for method in expected_results.keys():
        
        print(f'\n[Method: {method}]')

        expected_results[method] = CL_single_compute(method=method, observed=None, s_hypothesis=s_hypothesis, bg_expected=bg_expected)
        print('Expected upper 95CL limit on xs/xs_{benchmark} (H0: background only scenario):')
        print(np.round(expected_results[method],4))

        print('')

        observed_results[method] = CL_single_compute(method=method, observed=observed, s_hypothesis=s_hypothesis, bg_expected=bg_expected)
        print('Simulated (observed) 95CL upper limit on xs/xs_{benchmark} (H1: nominal signal (mu=1) + background scenario):')
        print(np.round(observed_results[method],4))

    assert np.linalg.norm(expected_results['asymptotic'] - expected_results['toys']) < EPS
    assert np.linalg.norm(observed_results['asymptotic'] - observed_results['toys']) < EPS

#else:
#    raise Exception('Minimization failed')

"""
rvalues = np.logspace(np.log10(r_bounds[0]), np.log10(r_bounds[1]), 10000)
import matplotlib.pyplot as plt
cc = np.zeros(len(rvalues))
for kk in range(len(rvalues)):
    cc[kk] = CL_single_asymptotic(s_hypothesis=rvalues[kk] * s_hypothesis, bg_expected=bg_expected, observed=X)

fig, ax = plt.subplots()
plt.plot(rvalues, cc)
plt.xscale('log')
plt.xlabel('$r$'); plt.ylabel('CL'); plt.title(f'$\\alpha = {alpha[i]:0.3f}$')
plt.savefig(f'tmp/method={method}_alpha={alpha[i]}_trials={trials}.pdf')
plt.close()

# --------------

# Slower but more global search via mystic solver
monitor = mystic.monitors.VerboseMonitor(0)
if   mystic_solver == 'diffev2':
    solver = mystic.solvers.diffev2
elif mystic_solver == 'fmin':
    solver = mystic.solvers.fmin
elif mystic_solver == 'fmin_powell':
    solver = mystic.solvers.fmin_powell

result  = solver(func, x0=[start], bounds=[r_bounds], penalty=None, ftol=func_tol, \
                 disp=False, full_output=True, itermon=monitor, maxiter=mystic_maxiter)
mystic_maxiter += 50

r_up.append( result[0] )
fun  = result[1]
"""