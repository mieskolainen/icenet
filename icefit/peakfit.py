# Binned histogram chi2/likelihood fits with iminuit (minuit from python)
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

import os
import pickle
import iminuit
import matplotlib.pyplot as plt
import uproot
import scipy.special as special
import scipy.integrate as integrate

# iceplot
import sys
sys.path.append(".")

from iceplot import iceplot
import statstools

# numpy
import numpy as onp # original numpy
from numpy.random import default_rng

sgn_pind = None
bgk_pind = None


def TH1_to_numpy(hist):
	"""
	Convert TH1 (ROOT) histogram to numpy array
	
	Args:
		hist: TH1 object (from uproot)
	"""

	#for n, v in hist.__dict__.items(): # class generated on the fly
	#	print(f'{n} {v}')

	hh         = hist.to_numpy()
	counts     = np.array(hist.values())
	errors     = np.array(hist.errors())

	bin_edges  = np.array(hh[1])
	bin_center = np.array((bin_edges[1:] + bin_edges[:-1]) / 2)

	return {'counts': counts, 'errors': errors, 'bin_edges': bin_edges, 'bin_center': bin_center}


def gauss_pdf(x, par):
	"""
	Normal (Gaussian) density
	
	Args:
		par: parameters
	"""
	mu, sigma = par
	y = 1.0 / (sigma * np.sqrt(2*np.pi)) * np.exp(- 0.5 * ((x - mu)/sigma)**2)
	return y


def CB_pdf(x, par):
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

	return y


def CB_G_conv_pdf(x, par, norm=True):
	"""
	Crystall Ball (*) Gaussian, with the same center value as CB,
	where (*) is a convolution product.
	
	Args:
		par: CB parameters (4), Gaussian width (1)
	"""
	mu     = par[0]
	reso   = par[-1]

	CB_y   = CB_pdf_(x=x, par=par[:-1])
	kernel = gauss_pdf(x=x, par=np.array([mu, reso]))

	y = np.convolve(a=CB_y, v=kernel, mode='same')

	# Normalize to density over the range of x
	y = y / integrate.simpson(y=y, x=x)

	return y


def cauchy_pdf(x, par):
	"""
	Cauchy pdf (non-relativistic fixed width Breit-Wigner)
	"""
	M0, W0 = par

	return 1 / (np.pi*W0) * (W0**2 / ((x - M0)**2 + W0**2))


def RBW_pdf(x, par):
	"""
	Relativistic Breit-Wigner pdf
	https://en.wikipedia.org/wiki/Relativistic_Breit%E2%80%93Wigner_distribution
	"""
	M0, W0 = par

	# Normalization
	gamma = np.sqrt(M0**2 * (M0**2 + W0**2))
	k     = (2*np.sqrt(2)*M0*W0*gamma) / (np.pi * np.sqrt(M0**2 + gamma))

	return k / ((x**2 - M0**2)**2 + M0**2 * W0**2)


def asym_RBW_pdf(x, par):
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

	return k / ((x**2 - M0**2)**2 + M0**2 * W**2)


def asym_BW_pdf(x, par):
	"""
	Breit-Wigner with asymmetric tail shape

	Param: a < 0 gives right hand tail, a == 0 without, a > 0 left hand tail
	"""
	M0, W0, a = par

	# Asymmetric running width
	W = 2*W0 / (1 + np.exp(a * (x - M0)))

	return 1 / (np.pi*W0) * (W**2 / ((x - M0)**2 + W**2))


def exp_pdf(x, par):
	"""
	Exponential density
	
	Args:
		par: mean parameter
	"""
	mu = par[0]
	return mu * np.exp(-mu * x)


def CB_RBW_conv_pdf(x, par, norm=True):
	"""
	Crystall Ball (*) Relativistic Breit-Wigner, with the same center values,
	where (*) is a convolution product.
	
	Args:
		par: RBW parameters (4), Breit-Wigner width and asymmetry
	"""

	CB_param   = par[0],par[1],par[2],par[3]
	aRBW_param = par[0],par[4],par[5]

	f1 = CB_pdf(x=x, par=CB_param)
	f2 = asym_RBW_pdf(x=x, par=aRBW_param)
	y  = np.convolve(a=f1, v=f2, mode='same')

	# Normalize to density over the range of x
	if norm:
		y = y / integrate.simpson(y=y, x=x)

	return y


def binned_1D_fit(hist, fitfunc, param, losstype='chi2', \
	ncall_gradient=10000, ncall_simplex=10000, ncall_brute=10000, max_trials=10, max_chi2=100):
	"""
	Main fitting function
	
	Args:
		hist:     TH1 histogram object (from uproot)
		fitfunc:  Fitting function
		param:    Parameters dict
		losstype: 'chi2' or 'nll'
	"""

	global sgn_pind
	global bgk_pind

	# -------------------------------------------------------------------------------
	# Histogram data

	h = TH1_to_numpy(hist)

	counts = h['counts']
	errs   = h['errors']
	cbins  = h['bin_center']

	# Limit the fit range
	fit_range_ind = (cbins >= param['fitrange'][0]) & (cbins <= param['fitrange'][1])

	### Chi2 loss function definition
	#@jit
	def chi2_loss(par):

		posdef = (errs > 0) # Check do we have non-zero bins
		if np.sum(posdef) == 0:
			return 1e9

		yhat = fitfunc(cbins[fit_range_ind & posdef], par)
		xx   = (yhat - counts[fit_range_ind & posdef])**2 / (errs[fit_range_ind & posdef])**2
		
		return onp.sum(xx)

	### Poissonian negative log-likelihood loss function definition
	#@jit
	def poiss_nll_loss(par):

		posdef = (errs > 0) # Check do we have non-zero bins
		if np.sum(posdef) == 0:
			return 1e9

		yhat  = fitfunc(cbins[fit_range_ind & posdef], par)

		T1 = counts[fit_range_ind & posdef] * np.log(yhat)
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

	trials = 1

	while True:

		if trials == 1:
			start_values = param['start_values']
		else:
			start_values = np.random.rand(len(param['start_values']))

		# ------------------------------------------------------------
		from scipy.optimize import minimize

		# Nelder-Mead search (from scipy)
		res = minimize(loss, param['start_values'], method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
		print(res)
		param['start_values'] = res.x
		# --------------------------------------------------------------------

		## Initialize Minuit
		m1 = iminuit.Minuit(loss, start_values, name=param['name'])

		if   losstype == 'chi2':
			m1.errordef = iminuit.Minuit.LEAST_SQUARES
		elif losstype == 'nll':
			m1.errordef = iminuit.Minuit.LIKELIHOOD


		m1.limits   = param['limits']
		m1.strategy = 0
		#m1.tol      = 1e-9

		# Brute force 1D-scan per dimension
		m1.scan(ncall=ncall_brute)
		print(m1.fmin)

		# Simplex (Nelder-Mead search)
		m1.simplex(ncall=ncall_simplex)
		print(m1.fmin)

		# Gradient search
		m1.migrad(ncall=ncall_gradient)
		print(m1.fmin)

		# Finalize with error analysis [migrad << hesse << minos (best)]
		m1.hesse()
		try:
			m1.minos()
		except:
			print(f'binned_1D_fit: Error occured with MINOS uncertainty estimation')

		### Output
		par     = m1.values
		cov     = m1.covariance
		var2pos = m1.var2pos
		chi2    = chi2_loss(par)
		ndof    = len(counts[fit_range_ind]) - len(par) - 1

		if (chi2 / ndof < max_chi2):
			break
		elif trials > max_trials:
			break
		else:
			trials += 1

	print(f'Parameters: {par}')
	print(f'Covariance: {cov}')

	if cov is None:
		print('** Uncertainty estimation failed! **')
		cov = -1 * np.ones((len(par), len(par)))

	print(f"chi2 / ndf = {chi2:.2f} / {ndof} = {chi2/ndof:.2f}")

	return par, cov, var2pos, chi2, ndof


def analyze_1D_fit(hist, fitfunc, sigfunc, bgkfunc, par, cov, var2pos, chi2, ndof, param):
	"""
	Analyze and visualize fit results
	
	Args:
		hist:    TH1 histogram object (from uproot)
		fitfunc, sigfunc, bgkfunc: Fitfunctions
		par:     Parameters obtained from the fit
		cov:     Covariance matrix obtained from the fit
		var2pos: Variable name to position index
		chi2:    Chi2 value of the fit
		ndof:    Number of dof
		param:   Input parameters of the fit
	
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

	x   = np.linspace(param['fitrange'][0], param['fitrange'][1], int(1e3))
	y_S = par[var2pos['S']] * sigfunc(x, par[sgn_pind])
	y_B = par[var2pos['B']] * bgkfunc(x, par[bgk_pind])

	print(f'Input bin count sum: {np.sum(counts):0.1f} (full range)')
	print(f'Input bin count sum: {np.sum(counts[fitind]):0.1f} (fit range)')	


	# --------------------------------------------------------------------
	# Compute count value integrals inside fitrange

	N          = {}
	N_err      = {}

	# Normalize integral measures to event counts
	# [normalization given by the data histogram binning because we fitted against it]
	deltaX     = np.mean(cbins[fitind][1:] - cbins[fitind][:-1])
	N['S']     = integrate.simpson(y_S, x) / deltaX
	N['B']     = integrate.simpson(y_B, x) / deltaX

	# Use the scale error as the leading uncertainty
	# [neglect the functional shape uncertanties affecting the integral]
	
	for key in ['S', 'B']:
		ind        = var2pos[key]
		N_err[key] = N[key] * np.sqrt(cov[ind][ind]) / par[ind]


	# --------------------------------------------------------------------
	# Print out

	for key in N.keys():
		print(f"N_{key}: {N[key]:0.1f} +- {N_err[key]:0.1f}")


	# --------------------------------------------------------------------
	# Plot it

	obs_M = {

	# Axis limits
	'xlim'    : (2.75, 3.50),
	'ylim'    : None,
	'xlabel'  : r'$M$',
	'ylabel'  : r'Counts',
	'units'   : {'x': 'GeV', 'y': '1'},
	'label'   : r'Invariant mass',
	'figsize' : (5,4),
	'density' : False,

	# Function to calculate
	'func'    : None
	}

	fig, ax = iceplot.create_axes(**obs_M, ratio_plot=True)
	
	## UPPER PLOT
	ax[0].errorbar(x=cbins, y=counts, yerr=errs, color=(0,0,0), label=f'Data, $N = {np.sum(counts):0.1f}$', **iceplot.errorbar_style)
	ax[0].legend(frameon=False)
	ax[0].set_ylabel('Counts / bin')

	## Plot fits
	plt.sca(ax[0])
	plt.plot(x, fitfunc(x, par), label="total fit: $Sf_S + Bf_B$", color=(0.5,0.5,0.5), lw=2)
	plt.plot(x, y_S, label=f"sig: $N_S = {N['S']:.1f} \\pm {N_err['S']:.1f}$", color=(0.85,0.85,0.85), linestyle='-')
	plt.plot(x, y_B, label=f"bkg: $N_B = {N['B']:.1f} \\pm {N_err['B']:.1f}$", color=(0.7,0,0), linestyle='--')
	plt.ylim(bottom=0)
	plt.legend(fontsize=7)

	# chi2 / ndf
	title = f"$\\chi^2 / n_\\mathrm{{dof}} = {chi2:.2f} / {ndof} = {chi2/ndof:.2f}$"
	plt.title(title)

	## LOWER PLOT
	plt.sca(ax[1])
	iceplot.plot_horizontal_line(ax[1], ypos=1.0)

	ax[1].errorbar(x=cbins, y=np.ones(len(cbins)),                            yerr=errs / np.maximum(1e-9, counts),        color=(0,0,0), label=f'Data', **iceplot.errorbar_style)
	ax[1].errorbar(x=cbins, y=fitfunc(cbins, par) / np.maximum(1e-9, counts), yerr=np.zeros(len(cbins)), color=(0.5,0.5,0.5), label=f'Fit', **iceplot.errorbar_line_style)

	ax[1].set_ylabel('Ratio')

	return fig,ax,N,N_err


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


def test_jpsi_fitpeak(MAINPATH = '/home/user/fitdata/flat/muon/generalTracks/JPsi', savepath='./output/peakfit'):
	"""
	J/psi peak fitting
	"""

	import pytest

	global sgn_pind
	global bgk_pind


	if not os.path.exists(savepath):
		os.makedirs(savepath)

	rng = default_rng(seed=1)


	# ====================================================================
	# Fit parametrization setup

	# Parameter indices for signal and background functions
	sgn_pind = np.array([2,3,4,5,6,7])
	bgk_pind = np.array([8])

	### Fit functions
	sigfunc = CB_RBW_conv_pdf
	bgkfunc = exp_pdf

	def fitfunc(x, par):
		return par[0]*sigfunc(x, par[sgn_pind]) + par[1]*bgkfunc(x, par[bgk_pind])

	# Parameter start values
	start_values = (10,
		            1,

		            3.1,
		            0.05,
		            1.001,
		            0.5,

		            9.32e-05,
		            0.0,

		            0.5)
	
	# Parameter (min,max) constraints
	limits = ((0.1, 1e8),
		      (0.1, 1e8),

		      (3.07, 3.13),
		      (1e-3, 0.3),
		      (1.0001, 10.0),
		      (0.1, 3.0),

		      (1e-9, 1e-1),
		      (-8.0, 0.0),
		  	  
		      (0.01, 2.5))
	
	# Parameter names
	name   = ('S',
		      'B',

		      'M0',
		      'sigma',
		      'n',
		      'alpha',

		      'width',
		      'asym',

		      'lambda')

	# Fit range limits
	fitrange = np.array([2.9, 3.3])

	# Finally collect all
	param  = {'start_values': start_values,
			  'limits': 	  limits,
			  'name': 		  name,
			  'fitrange': 	  fitrange}

	### Loss function type
	losstype = 'chi2'
	#losstype = 'nll'

	# ====================================================================
	#np.seterr(all='print') # Numpy floating point error treatment


	### Loop over datasets

	for YEAR     in [2016, 2017, 2018]:
		for TYPE in [f'Run{YEAR}', 'JPsi_pythia8']: # Data or MC

			for BIN1 in [1,2,3]:
				for BIN2 in [1,2,3,4,5]:
					for PASS in ['Pass', 'Fail']:

						### Uproot input
						rootfile = f'{MAINPATH}/Run{YEAR}/{TYPE}/Nominal/NUM_LooseID_DEN_TrackerMuons_absdxy_pt.root'
						tree     = f'NUM_LooseID_DEN_TrackerMuons_absdxy_{BIN1}_pt_{BIN2}_{PASS}'
						hist     = uproot.open(rootfile)[tree]

						# Fit and analyze
						par,cov,var2pos,chi2,ndof = binned_1D_fit(hist=hist, fitfunc=fitfunc, param=param, losstype=losstype)
						fig,ax,N,N_err            = analyze_1D_fit(hist=hist, fitfunc=fitfunc, sigfunc=sigfunc, bgkfunc=bgkfunc, \
																par=par, cov=cov, chi2=chi2, var2pos=var2pos, ndof=ndof, param=param)

						# Create savepath
						total_savepath = f'{savepath}/Run{YEAR}/{TYPE}/Nominal'
						if not os.path.exists(total_savepath):
							os.makedirs(total_savepath)

						# Save the fit plot
						plt.savefig(f'{total_savepath}/{tree}.pdf')
						plt.close('all')

						# Save the fit numerical data
						par_dict, cov_arr = iminuit2python(par=par, cov=cov, var2pos=var2pos)
						outdict  = {'par': par_dict, 'cov': cov_arr, 'var2pos': var2pos, 'chi2': chi2, 'ndof': ndof, 'N': N, 'N_err': N_err}
						filename = f"{total_savepath}/{tree}.pkl"
						pickle.dump(outdict, open(filename, "wb"))
						print(f'Fit results saved to: {filename} (pickle) \n\n')


def test_jpsi_tagprobe(savepath='./output/peakfit'):
	"""
	Tag & Probe efficiency (& scale factors)
	"""

	import pytest
	
	def tagprobe(treename, total_savepath):

		N      = {}
		N_err  = {}

		for PASS in ['Pass', 'Fail']:

			tree     = f'{treename}_{PASS}'
			filename = f"{total_savepath}/{tree}.pkl"
			print(f'Reading fit results from: {filename} (pickle)')			
			outdict  = pickle.load(open(filename, "rb"))
			#pprint(outdict)

			# Read out signal peak fit event count yield and its uncertainty
			N[PASS]     = outdict['N']['S']
			N_err[PASS] = outdict['N_err']['S']

		return N, N_err


	### Loop over datasets
	for YEAR     in [2016, 2017, 2018]:

		data_tag = f'Run{YEAR}'
		mc_tag   = 'JPsi_pythia8'

		# Create savepath
		total_savepath = f'{savepath}/Run{YEAR}/Efficiency'
		if not os.path.exists(total_savepath):
			os.makedirs(total_savepath)
		
		for BIN1 in [1,2,3]:
			for BIN2 in [1,2,3,4,5]:

				print(f'------------------ YEAR = {YEAR} | BIN1 = {BIN1} | BIN2 = {BIN2} ------------------')

				eff     = {}
				eff_err = {}

				treename = f'NUM_LooseID_DEN_TrackerMuons_absdxy_{BIN1}_pt_{BIN2}'

				for TYPE in [data_tag, mc_tag]:

					### Compute Tag & Probe efficiency
					N,N_err       = tagprobe(treename=treename, total_savepath=f'{savepath}/Run{YEAR}/{TYPE}/Nominal')
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
				filename = f"{total_savepath}/{treename}.pkl"
				pickle.dump(outdict, open(filename, "wb"))
				print(f'Efficiency and scale factor results saved to: {filename} (pickle)')


if __name__ == "__main__":
    test_jpsi_fitpeak()
    test_jpsi_tagprobe()
