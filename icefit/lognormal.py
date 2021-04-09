# Log-normal density parameters & generation
#
# Run tests with: pytest lognormal.py -rP
#
# m.mieskolainen@imperial.ac.uk, 2021

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def lognormal_param(m, v, inmode='mean'):
    """
    Compute log-normal distribution \mu and \sigma parameters from
    the target mean (or median) m and variance v

    \mu and \sigma follow the Wikipedia convention:
    https://en.wikipedia.org/wiki/Log-normal_distribution

    Args:
        m      : mean (or median or mode) value of the target log-normal pdf
        v      : variance value of the target log-normal pdf
        inmode : input m as 'mean', 'median', or 'mode'
    """
    if   inmode == 'mean':
        """ CLOSED FORM SOLUTION """
        mu    = np.log(m**2/np.sqrt(v + m**2))
        sigma = np.sqrt(np.log(v/m**2 + 1))

    elif inmode == 'median':
        """ CLOSED FORM SOLUTION """
        mu    = np.log(m)
        sigma = np.sqrt(np.log((np.sqrt(m**2 + 4*v)/m + 1)/2))

    elif inmode == 'mode':
        """ NUMERICAL SOLUTION """
        def fun(x):
            d = np.array([m - np.exp(x[0] - x[1]), v - (np.exp(x[1]) - 1)*np.exp(2*x[0] + x[1])])
            return np.sum(d**2)

        sol   = optimize.minimize(fun=fun, x0=[1, 1], method='nelder-mead')
        mu    = sol['x'][0]
        sigma = np.sqrt(sol['x'][1])

    else:
        raise Except('lognormal_param: Unknown inmode')

    print(f'lognormal_param:')
    print(f'  - Input:     m = {m:0.5f}, v = {v:0.5f}, target = {inmode}')
    print(f'  - Obtained parameters: mu = {mu:0.5f}, sigma = {sigma:0.5f}')

    return mu, sigma


def rand_lognormal(m, v, N, inmode='mean'):
    """
    Generate random numbers from log-normal density with
    the distribution target mean (or median) m and variance v

    Args:
        m      : mean (or median or mode) value of the target log-normal pdf
        v      : variance value of the target log-normal pdf
        N      : number of samples
        inmode : input m as 'mean', 'median', or 'mode'
    """
    mu,sigma = lognormal_param(m=m, v=v, inmode=inmode)

    # Standard transform
    y = np.exp(mu + sigma * np.random.randn(N))

    return y


def test_lognormal():

    import pytest

    N     = int(1e7)
    
    # Nominal value
    b0    = 3
    
    
    # Different relative uncertainties
    delta_val = np.array([0.1, 0.5, 1.0])

    # Different input modes
    modes = ['mean', 'median', 'mode']

    fig, ax = plt.subplots(nrows=len(delta_val), ncols=len(modes), figsize=(12,6))
    for i in range(len(delta_val)):
        delta = delta_val[i]
        for j in range(len(modes)):

            # Generate log-normally distributed variables
            b   = rand_lognormal(m=b0, v=(delta*b0)**2, N=N, inmode=modes[j])

            # Computete statistics
            mean = np.mean(b)
            med  = np.median(b)
            std  = np.std(b)

            # Find mode
            bins = np.linspace(0, 5*b0, 300)
            counts, xval = np.histogram(b, bins)
            mode  = xval[np.argmax(counts)]

            label = f'b_0 = {b0:0.1f}, \delta = {delta:0.2f} :: (mean,med,mode) = ({mean:0.2f}, {med:0.2f}, {mode:0.2f}), std = {std:0.2f}, std/mean = {std/mean:0.2f}, std/med = {std/med:0.2f}, std/mode = {std/mode:0.2f}'
            print(f'{label} \n')

            if (i == 0):
                ax[i,j].set_title(f'<{modes[j]}> as the target')

            ax[i,j].hist(b, bins, label=label)
            #ax[i,j].legend(fontsize=7)
            ax[i,j].set_xlim([0, b0*3])
            ax[i,j].set_xlabel('$b$')

        print('----------------------------------------------------------')
    plt.show()

