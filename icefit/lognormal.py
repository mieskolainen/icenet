# Log-normal density parameters & generation
#
# Run tests with: pytest lognormal.py -rP
#
# m.mieskolainen@imperial.ac.uk, 2021

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def lognormal_param(m, v, inmode='mean', verbose=False):
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

    if verbose:
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


def rand_powexp(b, sigma, N):
    """
    Power+exp type parametrization (positive definite)
    
    Args:
        sigma : standard deviation
        N     : number of samples
    """
    theta = np.random.randn(N) # theta ~ exp(-0.5 * sigma^2)

    return b*(1 + sigma)**theta


def test_lognormal():

    import pytest

    N     = int(1e6)
    
    # Nominal value
    b0    = 1.5
    
    # Different relative uncertainties
    delta_val = np.array([0.1, 0.4, 1.5])

    # Different input modes
    modes = ['mean', 'median', 'mode']

    fig, ax = plt.subplots(nrows=len(delta_val), ncols=len(modes), figsize=(14,9))
    for i in range(len(delta_val)):
        delta = delta_val[i]
        for j in range(len(modes)):

            # 1. Generate log-normally distributed variables
            sigma = delta*b0
            b   = rand_lognormal(m=b0, v=sigma**2, N=N, inmode=modes[j])

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
                ax[i,j].set_title(f'<{modes[j]}> as the target $m$')

            # 2. Generate power-exp distributed variables
            c  = rand_powexp(b=b0, sigma=sigma, N=N)

            ax[i,j].hist(b, bins, label=f'exact log-N: $m = {b0:0.2f}, \\sqrt{{v}} = {delta:0.2f}$', histtype='step')
            ax[i,j].hist(c, bins, label=f'approx power: $s = {delta:0.2f}$', histtype='step')
            
            ax[i,j].legend(fontsize=9, loc='lower right')
            ax[i,j].set_xlim([0, b0*3])
            ax[i,j].set_xlabel('$x$')

        print('----------------------------------------------------------')

    plt.savefig('sampling.pdf', bbox_inches='tight')
    plt.show()


def test_accuracy():
    """
    Test accuracy made in the power-exp approximation with median = 1
    """
    def func_s(v, order=1):

        A = np.sqrt(v)

        if order >= 2:
            A += v/2
        if order >= 3:
            A += -7*v**(3/2)/12
        if order >= 4:
            A += -17*v**2/24
        if order >= 5:
            A += 163*v**(5/2)/160
        if order >= 6:
            A += 1111*v**3/720
        if order >= 7:
            A += -96487*v**(7/2)/40320

        return A / (np.exp(np.sqrt(np.log(1/2 * (1 + np.sqrt(1 + 4*v))))) - 1)


    v = np.linspace(1e-9, 1.5**2, 10000)
    fig, ax = plt.subplots()
    plt.plot(np.sqrt(v), np.ones(len(v)), color=(0.5,0.5,0.5), ls='--')

    for i in range(1,7+1):
        if i % 2:
            txt = f'$O(v^{{{i:.0f}/2}})$'
        else:
            txt = f'$O(v^{{{i/2:.0f}}})$'
        plt.plot(np.sqrt(v), func_s(v=v, order=i), label=f'{txt}')

    plt.ylabel('[Expansion of $s$ $|_{near \\;\\; v=0}$] / Exact $s$')
    plt.xlabel('$\\sqrt{v}$')

    plt.legend()
    plt.xlim([0, 1.5])
    plt.ylim([0.8, 1.5])

    plt.savefig('series_approx.pdf', bbox_inches='tight')
    plt.show()



#test_lognormal()
#test_accuracy()