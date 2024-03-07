# Inverse CDF based dequantizer of integer valued (countably finite) amplitudes
#
# m.mieskolainen@imperial.ac.uk, 2023

import numpy as np
import matplotlib.pyplot as plt
import scipy
import numba
from tqdm import tqdm

def U(n, a, b):
    """
    Uniform random numbers between [a,b]
    """
    return a + (b-a)*np.random.rand(n)

@numba.njit
def compute_ind(x, theta):
    """
    Index association helper
    """
    ind = np.zeros(len(x))
    for i in range(len(x)):
        ind[i] = np.argmin(np.abs(x[i] - theta))
    return ind

def construct_icdf(theta, pvals, n_interp=int(1e4), kind='cubic'):
    """
    Construct inverse CDF
    """
    # Finely sampled discrete PDF via interpolation
    f  = scipy.interpolate.interp1d(x=theta, y=pvals, kind=kind)
    xx = np.linspace(np.min(theta), np.max(theta), n_interp)
    y  = f(xx)
    py = y / np.sum(y)

    # Discrete CDF via cumulative sum
    cdfy = np.cumsum(py)

    # Inverse discrete CDF
    boundary_shift = [-(theta[1]-theta[0])/2, (theta[-1]-theta[-2])/2] # Symmetric down/up
    xx   = np.linspace(np.min(theta)+boundary_shift[0], np.max(theta)+boundary_shift[1], n_interp)
    icdf = scipy.interpolate.interp1d(x=cdfy, y=xx, kind=kind, fill_value="extrapolate")
    
    return icdf

@numba.njit
def fast_loop(i:int, k:int, N_buffer:int, x:np.ndarray,
              deq:np.ndarray, theta=np.ndarray, theta_ind=np.ndarray):
    """
    Fast loop function used by iDQF
    """
    while True:
        if k >= N_buffer:
            return -1
        else:
            if np.argmin(np.abs(x[k] - theta)) == theta_ind[i]:
                deq[i] = x[k]
                k += 1
                break           
            k += 1
    return k

def iDQF(x, theta=None, pvals=None, N_buffer=int(1e4), n_interp=int(1e4), kind='cubic'):
    """
    Dequantization via inverse CDF
    
    Args:
        x:        data sample array (1D)
        theta:    (theoretical) quantized pdf x-values, if None, discrete pdf is estimated from x
        pvals:    (theoretical) quantized pdf y-values, -||-
        N_buffer: technical buffer size
        n_interp: number of interpolation points
        kind:     interpolation type, e.g. 'linear', 'cubic'
    
    Returns:
        dequantized x values
    """
    
    if theta is None:   # PDF not given, construct empirical PDF
        theta, counts = np.unique(x, return_counts=True)
        pvals = counts / np.sum(counts)
    
    if len(theta) == 1: # No need to dequantize, only one value
        return x
    
    icdf = construct_icdf(theta=theta, pvals=pvals, n_interp=n_interp, kind=kind)
    
    def generate():
        u    = np.random.rand(N_buffer) # Inverse CDF sampling
        return icdf(u), 0
    
    # Index association
    theta_ind = compute_ind(x=x, theta=theta)
    deq       = np.zeros(len(theta_ind))
    x,k       = generate()
    
    # Dequantization loop
    for i in tqdm(range(len(deq))):
        while True:
            # Arrays are updated via reference
            k = fast_loop(i=i, k=k, N_buffer=N_buffer,
                            x=x, deq=deq, theta=theta, theta_ind=theta_ind)
            if k == -1:
                x,k = generate()
            else:
                break
    
    return deq

def main():
    
    # Number of MC samples
    N     = int(1e5)

    # Theoretical discrete 1D PDF (could be an empirical histogram driven)
    theta = np.array([200, 220, 240, 260, 280, 300])  # observable values
    pvals = np.array([0.1, 0.15, 0.3, 0.6, 1.2, 2.4]) # prob
    pvals = pvals / np.sum(pvals)
    
    # Generate MC sample
    x         = np.random.choice(a=theta, size=N, p=pvals)
    
    # iCDF based dequantizer
    deq       = iDQF(x=x)

    # Naive flat dequantization
    noise     = U(len(x), -(theta[1]-theta[0])/2, (theta[-1]-theta[-2])/2)
    deq_naive = x + noise

    # Visualize MC samples
    fig,ax = plt.subplots(1,2,figsize=(10,3))
    
    plt.sca(ax[0])
    bins = np.linspace(np.min(deq), np.max(deq), 75)
    plt.hist(x,         bins=bins, color='red',   label='original')
    plt.hist(deq,       bins=bins, color='green', label='dequantized (iDQF)',  histtype='stepfilled', alpha=0.5)
    plt.hist(deq_naive, bins=bins, color='black', label='dequantized (naive)', histtype='step')
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('count')
    
    plt.sca(ax[1])
    bins = np.linspace(np.min(x-deq), np.max(x-deq), 75)
    plt.hist(x-deq,       bins=bins, color='green', label='orig - iDQF',  histtype='stepfilled', alpha=0.5)
    plt.hist(x-deq_naive, bins=bins, color='black', label='orig - naive', histtype='step', alpha=0.8)
    plt.legend()
    plt.xlabel('$x - \\tilde{{x}}$')
    plt.ylabel('count')
    plt.show()
    
    plt.savefig('dequantize.pdf', bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    main()
