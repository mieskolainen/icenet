# Linear (correlation) and non-linear dependency tools
#
# m.mieskolainen@imperial.ac.uk, 2021

import numpy as np
import numba
import copy
import scipy
import scipy.special as special
import scipy.stats as stats

# Needed for tests only
import pandas as pd


def hacine_entropy_bin(x, rho, mode="nbins", alpha=0.01):
    """
    Hacine-Gharbi et al. 
    “Low Bias Histogram-Based Estimation of Mutual Information for Feature Selection.”
    Pattern Recognition Letters, 2012.
    
    Args:
        See scott_bin()
    """

    N  = len(x)
    xi = (8 + 324*N + 12*np.sqrt(36*N + 729*N**2))**(1/3)
    nb = np.round(xi/6 + 2/(3*xi) + 1/3)

    if mode == "width":
        return (np.percentile(x, 100*(1-alpha)) - np.percentile(x, 100*alpha)) / nb
    else:
        return int(nb)

def hacine_joint_entropy_bin(x, rho, mode="nbins", alpha=0.01):
    """
    Hacine-Gharbi, Ravier. "A Binning Formula of Bi-histogram
    for Joint Entropy Estimation Using Mean Square Error Minimization.”
    Pattern Recognition Letters, 2018.

    Args:
        See scott_bin()
    """

    N = len(x)

    # BX=BY
    nb = np.round(1/np.sqrt(2) * np.sqrt(1 + np.sqrt(1 + 24*N/(1-rho**2))))

    if mode == "width":
        return (np.percentile(x, 100*(1-alpha)) - np.percentile(x, 100*alpha)) / nb
    else:
        return int(nb)


def freedman_diaconis_bin(x, mode="nbins", alpha=0.01):
    """
    Freedman-Diaconis rule for a 1D-histogram bin width
    
    D. Freedman & P. Diaconis (1981)
    “On the histogram as a density estimator: L2 theory”.
    
    ~ N**(-1/3)

    Args:
        x     : array of 1D data
        mode  : return 'width' or 'nbins'
        alpha : outlier percentile

    """
    IQR  = stats.iqr(x, rng=(25, 75), scale=1.0, nan_policy="omit")
    N    = len(x)
    bw   = (2 * IQR) / N**(1.0/3.0)

    if mode == "width":
        return bw
    else:
        return int(np.ceil((np.percentile(x, 100*(1-alpha)) - np.percentile(x, 100*alpha)) / bw))


def scott_bin(x, rho, mode="nbins", alpha=0.01):

    """ 
    Scott rule for a 2D-histogram bin widths
    
    Scott, D.W. (1992),
    Multivariate Density Estimation: Theory, Practice, and Visualization -- 2D-Gaussian case
    
    ~ N**(-1/4)

    Args:
        x     : array of 1D data (one dimension of the bivariate distribution)
        rho   : Linear correlation coefficient
        mode  : return 'width' or 'nbins'
        alpha : outlier percentile
    """
    N  = len(x)
    bw = 3.504*np.std(x)*(1 - rho**2)**(3.0/8.0)/len(x)**(1.0/4.0)

    if mode == "width":
        return bw
    else:
        return int(np.ceil((np.percentile(x, 100*(1-alpha)) - np.percentile(x, 100*alpha)) / bw))


def H_score(p):
    """
    Shannon Entropy (log_e ~ nats units)

    Args:
        p : probability vector
    Returns:
        entropy
    """
    # Make sure it is normalized
    p_ = (p[p>0]/np.sum(p[p>0])).astype(np.float64)

    return -np.sum(p_*np.log(p_))


def I_score(C, normalized=None, EPS=1E-15):
    """
    Mutual information score (log_e ~ nats units)

    Args:
        C : (X,Y) 2D-histogram array with event counts
        normalized : return normalized version (None, 'additive', 'multiplicative')
    
    Returns:
        mutual information score
    """
    def Pnorm(x):
        return np.maximum(x / np.sum(x.flatten()), EPS)

    nX, nY   = np.nonzero(C)
    Pi       = np.ravel(np.sum(C,axis=1))
    Pj       = np.ravel(np.sum(C,axis=0))
    
    # Joint 2D
    P_ij     = Pnorm(C[nX, nY]).astype(np.float64)
    log_P_ij = np.log(P_ij)
    
    # Factorized 1D x 1D
    Pi_Pj = Pi.take(nX).astype(np.float64) * Pj.take(nY).astype(np.float64)
    Pi_Pj = Pi_Pj / np.maximum(np.sum(Pi) * np.sum(Pj), EPS)

    # Definition
    I = np.sum(P_ij * (log_P_ij - np.log(Pi_Pj) ))
    I = np.clip(I, 0.0, None)

    # Normalization
    if   normalized == None:
        return I
    elif normalized == 'additive':
        return 2*I/(H_score(Pi) + H_score(Pj))
    elif normalized == 'multiplicative':
        return I/np.sqrt(H_score(Pi) * H_score(Pj))
    else:
        raise Exception(f'I_score: Error with unknown normalization parameter "{normalized}"')


def mutual_information(x, y, weights = None, bins_x=None, bins_y=None, normalized=None, automethod='Scott2D', minbins=4, alpha=0.01):
    """
    Mutual information entropy (non-linear measure of dependency)
    between x and y variables
    Args:
        x          : array of values
        y          : array of values
        w          : weights (default None)
        bins_x     : x binning array  If None, then automatic.
        bins_y     : y binning array.
        normalized : normalize the mutual information (see I_score() function)
    
    Autobinning args:    
        automethod : 'Hacine2D', 'Scott2D'
        minbins    : minimum number of bins per dimension
        alpha      : outlier protection percentile
        gamma      : FD ad-hoc scale parameter
    
    Returns:
        mutual information
    """
    rho,_ = pearson_corr(x,y)

    def autobinwrap(data):
        if   automethod == 'Scott2D':
            NB = scott_bin(x=data,rho=rho, mode='nbins',alpha=alpha)
        elif automethod == 'Hacine2D':
            NB = hacine_joint_entropy_bin(x=data, rho=rho, mode='nbins', alpha=alpha)
        else:
            raise Exception(f'mutual_information: Unknown autobinning parameter <{automethod}>')

        NB = int(np.maximum(minbins, NB))
        return np.linspace(np.percentile(data, alpha*100), np.percentile(data, 100*(1-alpha)), NB + 1)

    if bins_x is None:
        bins_x = autobinwrap(x)
    if bins_y is None:
        bins_y = autobinwrap(y)

    XY = np.histogram2d(x=x, y=y, bins=[bins_x,bins_y], weights=weights)[0]
    mi = I_score(C=XY, normalized=normalized)

    return mi


def gaussian_mutual_information(rho):
    """
    Analytical 2D-Gaussian mutual information
    using a correlation coefficient rho.
    
    I(X1,X2) = H(X1) + H(X2) - H(X1,X2)
    Args:
        rho : correlation coefficient between (-1,1)
    Returns:
        mutual information
    """
    return -0.5*np.log(1-rho**2)


def pearson_corr(x, y):
    """
    Pearson Correlation Coefficient
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    Args:    
        x,y : arrays of values
    Returns: 
        correlation coefficient [-1,1], p-value
    """

    if len(x) != len(y):
        raise Exception('pearson_corr: x and y with different size.')

    x     = np.asarray(x)
    y     = np.asarray(y)
    dtype = type(1.0 + x[0] + y[0]) # Should be at least float64

    # Astype guarantees precision. Loss of precision might happen here.
    x_ = x.astype(dtype) - x.mean(dtype=dtype)
    y_ = y.astype(dtype) - y.mean(dtype=dtype)

    # Correlation coefficient
    r  = np.dot(x_,y_) / (np.linalg.norm(x_) * np.linalg.norm(y_))
    r  = np.clip(r, -1.0, 1.0) # Safety

    # 2-sided p-value from the Beta-distribution
    ab   = len(x)/2 - 1
    dist = scipy.stats.beta(ab, ab, loc=-1, scale=2)
    prob = 2*dist.cdf(-abs(r))

    return r, prob


def optbins(x, maxM=150, mode="nbins", alpha=0.025):
    """

    NOTE: Weak performance, study the method !!
    
    Optimal 1D-histogram binning via Bayesian Brute Force search algorithm.
    
    K.H Knuth, 2012, Entropy.
    https://arxiv.org/abs/physics/0605197
    
    Args:
        x     : data points
        maxM  : maximum number of bins
        mode  : "nbins" or "width"
        alpha : outlier protection percentile

    Returns:
        optimal number of bins
    """
    N = len(x)

    # Outlier protection
    lo,hi  = np.percentile(x, alpha*100), np.percentile(x, 100*(1-alpha))
    ind = (x > lo) & (x < hi)

    # Loop over number of bins and compute (relative) posterior probability
    logp = np.ones(maxM)*(-1E32) # keep it negative for 0-bin

    for M in range(1,maxM):
        n       = np.histogram(x[ind], bins=M)[0]
        part1   = N*np.log(M) + special.gammaln(M/2) - special.gammaln(N+M/2)
        part2   = -M*special.gammaln(1/2) + np.sum(special.gammaln(n+0.5))
        logp[M] = part1 + part2;

    optM = np.argmax(logp)

    if mode == "width":
        return (np.max(x[ind]) - np.min(x[ind])) / optM
    else:
        return optM


def optbins2d(x,y, maxM=(40,40), mode="nbins", alpha=0.025):
    """
    
    NOTE: Weak performance, study the method !!

    Optimal 2D-histogram binning via Bayesian Brute Force search algorithm.
    
    K.H Knuth, 2012, Entropy.
    https://arxiv.org/abs/physics/0605197
    
    Args:
        x     : data points
        maxM  : maximum number of bins per dimension
        mode  : "nbins" or "width"
        alpha : outlier protection percentile
    
    Returns:
        optimal number of bins
    """
    N = len(x)

    if len(x) != len(y):
        raise Exception('optbins2d: len(x) != len(y)')

    # Outlier protection
    x_lo,x_hi  = np.percentile(x, alpha*100), np.percentile(x, 100*(1-alpha))
    y_lo,y_hi  = np.percentile(y, alpha*100), np.percentile(y, 100*(1-alpha))

    ind = (x > x_lo) & (x < x_hi) & (y > y_lo) & (y < y_hi)

    # Loop over number of bins and compute (relative) posterior probability
    logp = np.ones(maxM)*(-1E32) # keep it negative for 0-bin

    for Mx in range(1,maxM[0]):
        for My in range(1,maxM[1]):
            n         = np.histogram2d(x=x[ind],y=y[ind], bins=(Mx,My))[0].flatten()
            M         = Mx*My
            part1     = N*np.log(M) + special.gammaln(M/2) - special.gammaln(N+M/2)
            part2     = -M*special.gammaln(1/2) + np.sum(special.gammaln(n+0.5))
            logp[Mx,My] = part1 + part2;

    # Find optimal number of (x,y) bins
    optM = np.unravel_index(logp.argmax(), logp.shape)

    if mode == "width":
        return ((np.max(x[ind]) - np.min(x[ind])) / optM[0],
                (np.max(y[ind]) - np.min(y[ind])) / optM[1])
    else:
        return optM


def test_gaussian():
    """
    Gaussian unit test of the dependence estimators.
    """

    ## Create synthetic Gaussian data
    for N in [int(1e2), int(1e3), int(1e4)]:

        print(f'*************** statistics N = {N} ***************')

        for rho in np.linspace(-0.99, 0.99, 11):
        
            # Create correlation via 2D-Cholesky
            z1  = np.random.randn(N)
            z2  = np.random.randn(N)
            
            x1  = z1
            x2  = rho*z1 + np.sqrt(1-rho**2)*z2
            
            # ---------------------------------------------------------------

            print(f'<rho = {rho:.3f}>')

            r,prob = pearson_corr(x=x1, y=x2)
            MI_G   = gaussian_mutual_information(rho)
            MI     = mutual_information(x=x1, y=x2)

            print(f'Pearson corrcoeff = {r:.3f} (p-value = {prob:0.3E})')
            print(f'Gaussian exact MI = {MI_G:.3f}')
            print(f'Numerical      MI = {MI:.3f}')
            print('')


def test_data():

    # Read toy dataset
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00451/dataR2.csv")
    #target = df['Classification']

    df.drop(['Classification'], axis=1, inplace=True)

    # Build MI matrix for each pair of features
    D = df.shape[1]
    MI   = np.zeros((D,D))
    MI_A = np.zeros((D,D))
    MI_M = np.zeros((D,D))

    for i,col_i in enumerate(df):
        for j,col_j in enumerate(df):

            MI[i,j]   = mutual_information(x=df[col_i], y=df[col_j], normalized=None)
            MI_A[i,j] = mutual_information(x=df[col_i], y=df[col_j], normalized='additive')
            MI_M[i,j] = mutual_information(x=df[col_i], y=df[col_j], normalized='multiplicative')

    # Print out
    print('>> Raw Mutual Information')
    print(pd.DataFrame(MI,   columns = df.columns, index = df.columns))
    print('')
    print('>> Additively Normalized Mutual Information')
    print(pd.DataFrame(MI_A, columns = df.columns, index = df.columns))
    print('')
    print('>> Multiplicatively Normalized Mutual Information')
    print(pd.DataFrame(MI_M, columns = df.columns, index = df.columns))


# Run tests
#test_gaussian()
#test_data()

