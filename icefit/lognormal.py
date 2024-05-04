# Log-normal density parameters & generation
#
# Run with: python lognormal.py
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def lognormal_param(m, v, match_mode='mean', verbose=True):
    """
    Compute log-normal distribution \mu and \sigma parameters from
    the target mean (or median) m and variance v

    \mu and \sigma follow the Wikipedia convention:
    https://en.wikipedia.org/wiki/Log-normal_distribution

    Args:
        m      : mean (or median or mode) value of the target log-normal pdf
        v      : variance value of the target log-normal pdf
        match_mode : input m as 'mean', 'median', or 'mode'
    """
    if   match_mode == 'mean':
        """ CLOSED FORM SOLUTION """
        mu    = np.log(m**2/np.sqrt(v + m**2))
        sigma = np.sqrt(np.log(v/m**2 + 1))

    elif match_mode == 'median':
        """ CLOSED FORM SOLUTION """
        mu    = np.log(m)
        sigma = np.sqrt(np.log((np.sqrt(m**2 + 4*v)/m + 1)/2))

    elif match_mode == 'mode':
        """ NUMERICAL SOLUTION """
        
        def mode_func(x):
            d = np.array([m - np.exp(x[0] - x[1]), v - (np.exp(x[1]) - 1)*np.exp(2*x[0] + x[1])])
            return d
        
        init_guess = (np.log(m), np.log(1 + np.sqrt(v)/m))
        sol   = fsolve(mode_func, init_guess)
        mu    = sol[0]
        sigma = np.sqrt(sol[1])
    
    elif match_mode == 'geometric':
        """ CLOSED FORM SOLUTION """
        mu    = np.log(m) # m is median and also geometric mean
        sigma = np.log(1 + np.sqrt(v)/m) # from log(s_G), where s_G = exp(STD(log(X)))
    
    else:
        raise Exception(f'lognormal_param: Unknown match_mode {match_mode}')

    if verbose:
        print(f'lognormal_param:')
        print(f'  - Input:     m = {m:0.3f}, v = {v:0.3f}, sqrt[v] = {np.sqrt(v):0.3f}, sqrt[v] / m = {np.sqrt(v)/m:0.3f}, target mode = {match_mode}')
        print(f'  - Obtained parameters: mu = {mu:0.3f}, sigma = {sigma:0.3f}, exp(sigma) = {np.exp(sigma):0.3f}')
    
    return mu, sigma


def rand_lognormal(m, v, N, match_mode='mean'):
    """
    Generate random numbers from log-normal density with
    the distribution target mean (or median) m and variance v

    Args:
        m      : mean (or median or mode) value of the target log-normal pdf
        v      : variance value of the target log-normal pdf
        N      : number of samples
        match_mode : input m as 'mean', 'median', or 'mode'
    """
    mu,sigma = lognormal_param(m=m, v=v, match_mode=match_mode)

    # Standard transform
    y = np.exp(mu + sigma * np.random.randn(N))

    return y

def rand_powexp(m, std, N):
    """
    Power+exp type parametrization for median matching
    
    Args:
        m     : median target
        std   : standard deviation target
        N     : number of samples
    """
    theta = np.random.randn(N) # theta ~ exp(-0.5 * sigma^2)

    return m*(1 + std/m)**theta

def create_label(X):
    mean  = np.mean(X)
    med   = np.median(X)
    std   = np.std(X)
    gsd   = np.exp(np.std(np.log(X))) # Geometric standard deviation
    
    # Find histogram mode
    bins = np.linspace(0, 5*med, 300)
    counts, xval = np.histogram(X, bins)
    mode = xval[np.argmax(counts)]

    return f'(mean,med,mode) = ({mean:0.1f}, {med:0.1f}, {mode:0.1f}), STD[X] = {std:0.2f}, GSD[X] = {gsd:0.2f}, std/mean = {std/mean:0.1f}, std/med = {std/med:0.1f}, std/mode = {std/mode:0.1f}'

def test_lognormal():

    import pytest

    N     = int(1e7)

    # Nominal value
    m0    = 1.5
    
    # Different relative uncertainties
    delta_val = np.array([0.1, 0.4, 0.8, 1.6])
    
    # Different input modes
    modes  = ['mean', 'median', 'mode', 'geometric']
    symbol = ['\\bar{{m}}', 'm', '\\tilde{{m}}', 'm']
    
    fig, ax = plt.subplots(nrows=len(delta_val), ncols=len(modes), figsize=(20,15))
    for i in range(len(delta_val)):
        
        # log-normal pdf targets
        delta = delta_val[i] 
        std   = delta*m0
        
        # 2. Generate power-approx formula distributed variables
        Y  = rand_powexp(m=m0, std=std, N=N)
        Y_label = create_label(Y)
        
        for j in range(len(modes)):

            # 1. Generate log-normally distributed variables
            
            X       = rand_lognormal(m=m0, v=std**2, N=N, match_mode=modes[j])
            X_label = create_label(X)

            # Compute sample point statistics
            print(f'** {modes[j]} matching: m_0 = {m0:0.1f}, \delta = {delta:0.2f} **')
            print(f'{X_label} \n')

            title = f'$({symbol[j]} = {m0:0.1f}, \sqrt{{v}}/{symbol[j]} = {delta:0.1f})$'
            ax[i,j].set_title(f'<{modes[j]}> matching: {title}', fontsize=9)

            # ---------------
            
            bins = np.linspace(0, 5*np.median(X), 200)
            
            ax[i,j].hist(X, bins, label=f'E: {X_label}', histtype='step', density=True)
            ax[i,j].hist(Y, bins, label=f'A: {Y_label}', histtype='step', density=True)
            
            ax[i,j].legend(fontsize=4, loc='lower right')
            ax[i,j].set_xlim([0, m0*2.5])
            ax[i,j].set_ylim([0, None])
            
            if i == len(delta_val) - 1:
                ax[i,j].set_xlabel('$x$')

        print('----------------------------------------------------------')

    figname = 'sampling.pdf'
    print(f'Saving figure: {figname}')
    plt.savefig(figname, bbox_inches='tight')
    plt.close()

def test_accuracy():
    """
    Test accuracy made in the power-exp approximation with median = 1
    """
    def expanded_s(v, m, order=1):

        if order >= 1:
            A = np.sqrt(v)/m
        if order >= 2:
            A +=  (1/2)*(v/m**2)
        if order >= 3:
            A += -(7/12)*(v**(3/2)/m**3)
        if order >= 4:
            A += -(17/24)*(v**2/m**4)
        if order >= 5:
            A +=  (163/160)*(v**(5/2)/m**5)
        if order >= 6:
            A +=  (1111/720)*(v**3/m**6)
        if order >= 7:
            A += -(96487/40320)*(v**(7/2)/m**7)

        return A
    
    def exact_s(v, m):
        return np.exp(np.sqrt(np.log(1/2 * (np.sqrt(4*v/m**2 + 1) + 1)))) - 1
    
    # Fixed m value
    m       = 1.5
    v       = np.linspace(1e-9, m*4, 10000)
    fig, ax = plt.subplots()
    plt.plot(np.sqrt(v) / m, np.ones(len(v)), color=(0.5,0.5,0.5), ls='--')

    for i in range(1,7+1):
        if i % 2:
            txt = f'$O(v^{{{i:.0f}/2}} / m^{{{i:.0f}}})$'
        else:
            txt = f'$O(v^{{{i/2:.0f}}} / m^{{{i:.0f}}})$'
        plt.plot(np.sqrt(v) / m, expanded_s(v=v, m=m, order=i) / exact_s(v=v, m=m), label=f'{txt}')

    plt.ylabel('[Expansion of $s$ $|_{near \\;\\; \\sqrt{v}/m=0}$] / Exact $s$')
    plt.xlabel('$\\sqrt{v} / m$')

    plt.legend()
    plt.xlim([0, 1.5])
    plt.ylim([0.8, 1.5])

    figname = 'series_approx.pdf'
    print(f'Saving figure: {figname}')
    plt.savefig(figname, bbox_inches='tight')
    plt.close()

def main():
    test_accuracy()
    test_lognormal()
    

if __name__ == "__main__":
    main()
