# Expectation-Maximization and mixture density algorithms
#
# m.mieskolainen@imperial.ac.uk, 2020

import numpy as np
import numba
import copy

@numba.njit
def gausspdf(x,mu,sigma):
	""" Gaussian pdf """	
	return 1/np.sqrt(2*np.pi*sigma**2) * np.exp(-0.5 * ((mu-x)/sigma)**2)

@numba.njit
def mixture_nll(pdf, frac):
	""" Mixture model negative Log-Likelihood
	
	Args:
		pdf  : (n x K) density (pdf) values for n measurements, K classes
		frac : (K) class fractions with sum 1
	Returns:
		nll  : Negative Log-Likelihood
	"""
	LL = 0

	# Likelihood over all measurements
	for i in range(pdf.shape[0]):
		f = np.sum(pdf[i,:]*frac)
		LL += np.log(f)
	
	return (-1)*LL

#@numba.njit
def EM_frac(pdf, iters=30, EPS=1E-12, verbose=True):
	""" EM-algorithm for unknown integrated class fractions
	
	Args:
		pdf   : (n x K) density (pdf) values for n measurements, K classes
		iter  : Number of iterations
	Returns:
		frac  : Integrated class fractions
	"""
	n = pdf.shape[0]
	K = pdf.shape[1]

	P    = np.zeros((n,K))
	frac = np.ones(K) / K

	for k in range(iters):
		# Loop over observations
		for i in range(n):
			
			# E-step, obtain normalized probabilities
			P[i,:] = pdf[i,:] * frac[:]
			P[i,:] /= (np.sum(P[i,:]) + EPS)

		# M-step, update fractions by averaging over observations
		frac = np.sum(P,axis=0) / n

		if verbose:
			print(f'EM_frac: iter {k:4}, NLL = {mixture_nll(pdf,frac):.3f}, frac = {frac}')

	return frac

def test_EM():
	""" Test function to test EM iteration """
	
	# Toy Gaussian problem
	mu    = np.array([0.0,   3.0])
	sigma = np.array([1.0,   2.0])
	N     = np.array([1000, 2000]) # Number of samples

	# Generate random samples for each class
	S = []
	for i in range(len(N)):
		S.append( np.array( np.random.randn(N[i])*sigma[i] + mu[i] ))

	# Concatenate into one full sample
	SS = np.zeros(0)
	for i in range(len(S)):
		SS = np.concatenate((SS, np.array(S[i])))

	# Evaluate pdf values for each class
	pdf = np.zeros((len(SS), len(mu)))
	for i in range(len(SS)):
		for j in range(len(mu)):
			pdf[i,j] = gausspdf(SS[i], mu[j], sigma[j])

	# Solve the unknown fractions
	frac = EM_frac(pdf)

	print(f'True fractions = {N / np.sum(N)}')
