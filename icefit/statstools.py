# Statistical tests on ratios
#
# m.mieskolainen@imperial.ac.uk

import numpy as np
import numba
import copy
import scipy.stats as stats


def prodratio_eprop(A, B, sigmaA, sigmaB, sigmaAB=0, mode='ratio'):
	"""
	Error propagation (Taylor expansion) of product A*B or ratio A/B

	Args:
		A : Value A 
		B : Value B
		sigmaA  : 1 sigma uncertainty on A
		sigmaB  : 1 sigma uncertainty on B
		sigmaAB : Covariance of A,B
		mode    : 'prod' or 'ratio'

	Returns:
		Uncertainty on A/B 
	"""
	sign = 1 if mode == 'prod' else -1
	return np.sqrt((A/B)**2*((sigmaA/A)**2 + (sigmaB/B)**2 + sign*2*sigmaAB/(A*B)))


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
	R     = theta / (1 - theta)

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

		# Single sample Poisson tail integral
		ppois = poisson_tail(k1=k_pre[i], k2=k_obs[i])
		print(f'Poisson tail p = {ppois:0.3E} ({p2zscore(ppois):0.2f} sigmas)')

		# Error propagation
		errAB = prodratio_eprop(A=k_pre[i], sigmaA=s_pre[i], B=k_obs[i], sigmaB=s_obs[i])
		PRE_over_OBS = k_pre[i] / k_obs[i]
		print(f'  error propagated rel.uncertainty on the ratio: err / (pre/obs) = {errAB / PRE_over_OBS:0.3f}')

		print('')
