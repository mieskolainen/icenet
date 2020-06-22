# Binned chi2 histogram fit with iminuit (minuit from python) and jax (autograds)
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk


# !pip install iminuit jax jaxlib

# JAX for autograd
from jax.config import config
config.update("jax_enable_x64", True) # enable float64 precision
from jax import numpy as np           # jax replacement for normal numpy
from jax.scipy.special import erf
from jax import jit, grad
from iminuit import Minuit

import matplotlib.pyplot as plt

# iceplot
import sys
sys.path.append(".")
import _icepaths_
import iceplot

# numpy
import numpy as onp # original numpy
from numpy.random import default_rng


# -------------------------------------------------------------------------------
# Generate some toy data
rng = default_rng(seed=1)

S_samples = rng.normal(3.3, 0.15, 1000)
B_samples = rng.exponential(1.5, 10000)
data = onp.concatenate([S_samples, B_samples])


# -------------------------------------------------------------------------------
# Histogram data

obs_M = {

# Axis limits
'xlim'    : (0, 8.0),
'ylim'    : None,
'xlabel'  : r'$M$',
'ylabel'  : r'Counts',
'units'   : r'GeV',
'label'   : r'Invariant mass',
'figsize' : (5,4),

# Histogramming
'bins'    : np.linspace(0, 8, 120),
'density' : False,

# Function to calculate
'func'    : None
}

counts, errs, bins, cbins = iceplot.hist(data, bins=obs_M['bins'], density=False)


# -------------------------------------------------------------------------------
# FIT functions

# Normal density
def gauss_pdf(x, par) :
	mu, sigma = par
	y = 1.0 / (sigma * np.sqrt(2*np.pi)) * np.exp(- 0.5 * ((x - mu)/sigma)**2)
	return y

# Exponential density
def exp_pdf(x, par) :
	return par * np.exp(-par * x)

# Total function
def fitfunc(x, par) :
	return par[0]*gauss_pdf(x, par[2:4]) + par[1]*exp_pdf(x, par[4])

# Chi2 loss
@jit
def chi2_loss(par) :
	EPS = 1E-15
	print(par)
	yhat = fitfunc(cbins, par)
	chi2 = onp.sum( (yhat - counts)**2 / (errs + EPS)**2 )
	return chi2


# -------------------------------------------------------------------------------
# MINUIT fit

# Parameter start values and (min,max) constraints
start_values = (100, 100, 3.0, 0.1, 0.5)
limits = ((0, 1e6), (0, 1e6), (2.5, 3.5), (0.1, 0.2), (0.5, 2.5))


m1 = Minuit.from_array_func( jit(chi2_loss), start_values, limit=limits, pedantic=False)
m1.strategy = 0

m1.migrad()
par = m1.np_values()

m1.hesse()
cov = m1.np_covariance()

print(f'Parameters: {par}')
print(f'Covariance: {cov}')


# -------------------------------------------------------------------------------
# Plots

fig, ax = iceplot.create_axes(**obs_M, ratio_plot=False)

ax[0].errorbar(x=cbins, y=counts, yerr=errs, color=(0,0,0), label='Data', **iceplot.errorbar_style)
#ax.hist(x=cbins, bins=bins, weights=counts, color=(0,0,0), label='Data', **iceplot.hist_style_step)
ax[0].legend(frameon=False)

x = np.linspace(bins[0], bins[-1], 1000)

plt.plot(x, fitfunc(x, par), label="total fit: $Sf_S + Bf_B$", color=(0.5,0.5,0.5))
plt.plot(x, par[1] * exp_pdf(x, par[4]), label=f"bkg fit: $B = {par[1]:.1f} \\pm {np.sqrt(cov[1][1]):.1f}$", color=(0.7,0,0), linestyle='--')
plt.plot(x, par[0] * gauss_pdf(x, par[2:4]), label=f"signal fit: $S = {par[0]:.1f} \\pm {np.sqrt(cov[0][0]):.1f}$", color=(0.85,0.85,0.85), linestyle='-')
plt.ylim(bottom=0)

plt.legend()

# chi2 / ndf
chi2 = m1.fval
ndof = len(counts) - len(par) - 1
plt.title(f"$\\chi^2 / n_\\mathrm{{dof}} = {chi2:.2f} / {ndof} = {chi2/ndof:.2f}$");

plt.show()
