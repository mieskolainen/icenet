# Floating point mantissa bit reduction precision impact study
# 
# run with: python icefit/mantissa.py
#
# m.mieskolainen@imperial.ac.uk, 2023

import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg') # matplotlib thread problems

import math
import numba
import copy
import math
import os

@numba.njit
def round_sig(x: float, sig: int=2):
    """
    Round to significant figures
    """
    return round(x, sig-int(math.floor(math.log10(abs(x))))-1)

@numba.njit(parallel=True)
def reduce_mantissa_digits(x: np.ndarray, mandigits: int=3):
    """
    Args:
        x         : floating point numbers in an array
        mandigits : number of digits precision in the mantissa
    """
    y = np.zeros(len(x))
    
    for i in numba.prange(len(x)):
        
        # Get mantissa and exponent
        m,e  = math.frexp(x[i])
        # Round mantissa, frexp returns 0.5 <= |m| < 1.0
        m    = np.round(m, mandigits)
        # Map back
        y[i] = math.ldexp(m,e)

    return y

@numba.njit(parallel=True)
def reduce_mantissa_bits(x: np.ndarray, manbits: int=10):
    """
    Args:
        x       : floating point numbers in an array
        manbits : number of leading bits for the mantissa
    """
    y = np.zeros(len(x))
    
    for i in numba.prange(len(x)):
        
        # Get mantissa and exponent
        m,e  = math.frexp(x[i])
        # Scale mantissa with ldexp, then round to nearest integer
        mhat = np.rint(math.ldexp(m, manbits))
        # Scale back and recover the exponent
        y[i] = math.ldexp(mhat, e - manbits)
    
    return y

def digits2bits(digits: float) -> float:
    return digits * np.log(10) / np.log(2)

def bits2digits(bits: int) -> float:
    return np.log10(2**bits)

def plots(x, xhat, b, error_bins=300, dquant_label='none', mantissa_title='None', label='default', savepath='output'):

    # Error distribution plots
    fig,ax = plt.subplots(1,2, figsize=(10,14))
    fig.tight_layout(pad=1.0) # Give space

    error     = x - xhat
    rel_error = (x - xhat) / x
    
    plt.sca(ax[0])
    plt.hist(error, error_bins, histtype='step', label=f'$\\mu = {np.mean(error):0.1e}, \\sigma = {np.std(error):0.1e}$')
    plt.xlabel('Error: $x - \\tilde{x}$')
    plt.legend(fontsize=9, loc='upper right')
    plt.title(f'{label}')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) # scientific notation
    
    plt.sca(ax[1])
    plt.hist(rel_error, error_bins, histtype='step', label=f'$\\mu = {np.mean(rel_error):0.1e}, \\sigma = {np.std(rel_error):0.1e}$')
    plt.xlabel('Relative error: $(x - \\tilde{x}) / x$')
    plt.legend(fontsize=9, loc='upper right')
    plt.title(f'')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    for i in range(len(ax)):
        ax[i].set_aspect(1.0/ax[i].get_data_ratio(), adjustable='box')
    
    filename = f'{savepath}/hist_error_{label}.pdf'
    print(f'Saving figure to: {filename}')
    fig.savefig(filename, bbox_inches='tight')
    plt.close()
    
    # Histograms
    fig,ax = plt.subplots(3, len(b), figsize=(16,14))
    fig.tight_layout(pad=6.0) # Give space
    
    for i in range(len(b)):
        
        bins  = b[i]
        cbins = (bins[1:] + bins[:-1])/2
        
        ## Original
        plt.sca(ax[0][i])
        n_x, _, _ = plt.hist(x, bins=bins, histtype='step')
        
        plt.errorbar(cbins, y=n_x, yerr=np.sqrt(n_x), color='black', linestyle='')
        plt.title('IEEE 64b floats')
        plt.ylabel(f'$N$  (counts / {bins[1]-bins[0]:0.2g})')
        plt.xlabel('$x$')
        
        ## Approximate
        plt.sca(ax[1][i])
        n_xhat, _, _ = plt.hist(xhat, bins=bins, histtype='step')
        
        plt.errorbar(cbins, y=n_xhat, yerr=np.sqrt(n_xhat), color='black', linestyle='')
        plt.title(mantissa_title)
        plt.ylabel(f'$\\tilde{{N}}$  (counts / {bins[1]-bins[0]:0.2g})')
        plt.xlabel('$\\tilde{x}$')
        
        
        ## Ratios
        plt.sca(ax[2][i])
        
        non_zero     = (n_x != 0)
        poisson_err  = np.zeros(len(n_x))
        relative_err = np.zeros(len(n_x))
        
        poisson_err[non_zero]  = 100 * 1 / np.sqrt(n_x[non_zero])
        relative_err[non_zero] = 100 * (n_xhat[non_zero] - n_x[non_zero]) / n_x[non_zero]
        
        plt.hist(x=cbins, bins=bins, weights= poisson_err, color='red', alpha=0.25, label='$1 \, / \, \sqrt{N}$')
        plt.hist(x=cbins, bins=bins, weights=-poisson_err, color='red', alpha=0.25)
        
        plt.hist(x=cbins, bins=bins, weights=relative_err, label='$(\\tilde{N}\, - \, N) \, / \, N$', alpha=0.6)
        
        plt.plot(bins, np.zeros(len(bins)), color=(0.5,0.5,0.5)) # Horizontal line
        plt.title('Ratios $\\times \, 100$')
        plt.ylabel('[%]')
        plt.legend(fontsize=8, loc='upper left')
        
    for i in range(len(ax)):
        for j in range(len(ax[i])):
            ax[i][j].set_aspect(1.0/ax[i][j].get_data_ratio(), adjustable='box')

    filename = f'{savepath}/hist_counts_{label}.pdf'
    print(f'Saving figure to: {filename}')
    fig.savefig(filename, bbox_inches='tight')
    plt.close()


def main():

    savepath = os.getcwd() + '/output/mantissa'
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    
    # Number of samples
    N = int(1e5)

    ## Histogram binning
    min_edge = 1.8
    max_edge = 2.2

    b = [np.linspace(min_edge, max_edge, 501),
         np.linspace(min_edge, max_edge, 101),
         np.linspace(min_edge, max_edge, 26),
         np.linspace(min_edge, max_edge, 11)]

    # Dequantize strength parameter
    d_sigma = 1e-3

    # MC simulation input
    for input in ['flat', 'resonance']:
        
        if input == 'flat':
            # Generate flat distribution
            # Larger range --> more "steps" in the error distribution
            # (dynamic precision of floats)
            xmin = 1
            xmax = 3
            x = xmin + (xmax - xmin) * np.random.rand(N)

        elif input == 'resonance':    
            # Generate Gaussian like resonance
            mu    = 2
            sigma = 0.05
            x     = mu + sigma * np.random.randn(N)

        # Number of mantissa bits to preserve
        for manbits in [10, 15, 20]:
            
            mandigits = bits2digits(manbits)
            mantissa_title = f'mantissa {manbits} bits ~ {mandigits:0.1f} digits'

            print(mantissa_title)
            
            # Reduce floating point precision
            print(f'Reducing precision to {manbits} mantissa bits')
            xhat_raw = reduce_mantissa_bits(x, manbits)

            # Post-Dequantization
            for dquant in ['none', 'relG']:
                
                if dquant == 'none':
                    xhat = copy.deepcopy(xhat_raw)
                
                # Add relative Gaussian dithering noise to dequantize
                if dquant == 'relG':
                    xhat = xhat_raw * (1 + d_sigma * np.random.randn(N))
                
                # Make plots
                plots(x=x, xhat=xhat, b=b, dquant_label=dquant, \
                    mantissa_title=mantissa_title, label=f'MC-input_{input}_manbits_{manbits}_dquant_{dquant}', \
                    savepath=savepath)

if __name__ == "__main__":
    main()
