# Auxialary functions
# 
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

import math
import numpy as np
import awkward as ak

import numba
import copy
from tqdm import tqdm
import os

import sklearn
from sklearn import metrics
from scipy import stats
import scipy.special as special

import icenet.tools.prints as prints
import icenet.tools.stx as stx


def parse_vars(items):
    """
    Parse a series of key-value pairs and return a dictionary
    """
    d = {}

    if items:
        for item in items:
            key, value = parse_var(item)
            d[key] = value
    return d


def ak2numpy(x, fields, null_value=float(999.0)):
    """
    Unzip awkward array to numpy array per column (awkward Record)
    
    Args:
        x:      awkward array
        fields: record field names to extract
    Returns:
        numpy array with columns ordered as 'fields' parameter
    """
    out = null_value * np.ones((len(x), len(fields)), dtype=float)
    for i in range(len(fields)):
        out[:,i] = ak.flatten(ak.unzip(x[fields[i]]))
    return out

#@numba.njit
def jagged2matrix(arr, scalar_vars, jagged_vars, jagged_maxdim, jagged_totdim, null_value=float(-999.0)):
    """
    Transform a "jagged" event container to a matrix (rows ~ event, columns ~ variables)
    
    Args:
        arr:           Awkward array type input for N events
        scalar_vars:   Scalar variables to pick (list of strings)
        jagged_vars:   Jagged variables to pick (list of strings)
        jagged_maxdim: Maximum dimension per jagged variable (integer array)
        jagged_totdim: Total dimension of jagged
        library:       Input type 'ak' or 'np'
        null_value:    Default
    
    Returns:
        mat:     Fixed dimensional 2D-numpy matrix (N x [# scalar var x {#jagged var x maxdim}_i])
    """

    jagged_maxdim = np.asarray(jagged_maxdim, dtype=int)

    N   = int(len(arr))
    D   = int(len(scalar_vars)) + int(jagged_totdim)
    
    print(__name__ + f'.jagged2matrix: Creating a matrix with dimension [{N} x {D}]')

    # Pre-processing of jagged vars
    jvname = []
    for j in range(len(jagged_vars)):
        # Awkward groups jagged variables, e.g. 'sv_x' to sv.x
        jvname.append(jagged_vars[j].split('_'))
    
    # Loop over events
    mat = null_value * np.ones((N,D), dtype=float)
    
    for ev in tqdm(range(N)):

        # First scalar vars
        k = 0
        for j in range(len(scalar_vars)):
            mat[ev,k] = ak.to_numpy(arr[ev][scalar_vars[j]])
            k += 1
        
        # Then jagged vars (can be empty)
        for j in range(len(jagged_vars)):

            x  = ak.to_numpy(arr[ev][jvname[j][0]][jvname[j][1]])
            if len(x) > 0:
                d_this = np.min([len(x), jagged_maxdim[j]])
                mat[ev, k:k+d_this] = x[0:d_this]
            
            # Increase block counter
            k += jagged_maxdim[j]

    return mat


def jagged2tensor(X, ids, xyz, x_binedges, y_binedges):
    """
    Args:
        
        X          : input data (samples x dimensions) with jagged structure
        ids        : all variable names
        xyz        : array of (x,y,z) channel triplet strings such as [['image_clu_eta', 'image_clu_phi', 'image_clu_e']]
        x_binedges
        y_binedges : arrays of bin edges
    
    Returns:
        T : tensor of size (samples x channels x rows x columns)
    """

    # Samples x Channels x Rows x Columns
    T = np.zeros((X.shape[0], len(xyz), len(x_binedges)-1, len(y_binedges)-1), dtype=float)

    # Choose targets
    for c in range(len(xyz)):
        ind = [ids.index(x) for x in xyz[c]]

        # Loop over all events
        for i in tqdm(range(X.shape[0])):
            T[i,c,:,:] = arrays2matrix(x_arr=X[i,ind[0]], y_arr=X[i,ind[1]], z_arr=X[i,ind[2]], 
                x_binedges=x_binedges, y_binedges=y_binedges)

    print(__name__ + f'.jagged2tensor: Returning tensor with shape {T.shape}')

    return T


def arrays2matrix(x_arr, y_arr, z_arr, x_binedges, y_binedges):
    """
    Array representation summed to matrix.
    
    Args:
        x_arr :      array of [x values]
        y_arr :      array of [y values]
        z_arr :      array of [z values]
        x_binedges : array of binedges
        y_binedges : array of binedges
        
    Returns:
        Matrix output
    """

    x_ind = x2ind(x=x_arr, binedges=x_binedges)
    y_ind = x2ind(x=y_arr, binedges=y_binedges)

    # Loop and sum (accumulate)
    A  = np.zeros((len(x_binedges)-1, len(y_binedges)-1), dtype=float)
    try:
        for i in range(len(x_ind)):
            A[x_ind[i], y_ind[i]] += z_arr[i]
    except:
        print(__name__ + f'.arrays2matrix: not valid input (returning 0-matrix)')

    return A


def x2ind(x, binedges) :
    """ Return histogram bin indices for data in x, which needs to be an array [].
    Args:
        x:        data to be classified between bin edges
        binedges: histogram bin edges
    Returns:
        inds:     histogram bin indices
    """
    nbins = len(binedges) - 1
    inds = np.digitize(x, binedges, right=True) - 1

    if len(x) > 1:
        inds[inds >= nbins] = nbins-1
        inds[inds < 0] = 0
    else:
        if inds < 0:
            inds = 0
        if inds >= nbins:
            inds = nbins - 1

    return inds


def makedir(targetdir, exist_ok=True):
    """
    Make directory
    """
    os.makedirs(targetdir, exist_ok = exist_ok)
    return targetdir


def split(a, n):
    """
    Generator which returns approx equally sized chunks.
    Args:
        a : Total number
        n : Number of chunks
    Example:
        list(split(10, 3))
    """
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def split_start_end(a, n):
    """
    Returns approx equally sized chunks.
    Args:
        a : Total number
        n : Number of chunks
    Example:
        list(split(10, 3))
    """
    ll = list(split(a,n))
    out = []
    for i in range(len(ll)):
        out.append([ll[i][0], ll[i][-1]])

    return out


def count_targets(events, ids, entry_start=0, entry_stop=None, new=False, library='np'):
    """ Targets statistics printout

    Args:
        events :     uproot object
        ids    :     list of branch identifiers
        entrystart:  uproot starting point
        entrystop :  uproot ending point
    
    Returns:
        Printout on stdout
    """
    K   = len(ids)
    if new:
        vec = events.arrays(ids, library=library, how=list, entry_start=entry_start, entry_stop=entry_stop)
        vec = np.asarray(vec)
    else:
        vec = np.array([events.array(name, entry_start=entry_start, entrystop=entry_stop) for name in ids])
    vec = vec.T
    
    print(__name__ + f'.count_targets: vec.shape = {vec.shape}')

    intmat = binaryvec2int(vec)
    BMAT   = generatebinary(K)
    print(__name__ + f'.count_targets: {ids}')
    for i in range(BMAT.shape[0]):
        print(f'{BMAT[i,:]} : {np.sum(intmat == i):>10} ({np.sum(intmat == i) / len(intmat):.4f})')
    
    return


def longvec2matrix(X, M, D, order='F'):
    """
    A matrix representation / dimension converter function
    useful e.g. for DeepSets and similar neural architectures.
    
    Args:
        X:        Numpy input matrix (2-dim) (N x [MD])
        M:        Number of set elements
        D:        Feature dimension
        order:    Reshape direction
    
    Returns:
        Output matrix (3-dim) (N x M x D)
    """

    Y = np.zeros((X.shape[0], M, D))
    for i in range(X.shape[0]):
        Y[i,:,:] = np.reshape(X[i,:], (M,D), order)

    return Y


@numba.njit
def number_of_set_bits(i):
    """ Return how many bits are active of an integer in a standard binary representation.
    """
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    return (((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24


@numba.njit
def binvec_are_equal(a,b):
    """ Compare equality of two binary vectors a and b.

    Args:
        a,b : binary vectors
    Returns
        true or false
    """
    if (np.sum(np.abs(a - b)) == 0):
        return True
    else:
        return False


@numba.njit
def binvec2powersetindex(X, B):
    """ 
    Binary vector to powerset index.

    Args:
        X : matrix of binary vectors [# number of vectors x dimension]
        B : the powerset matrix
    Returns:
        y : array of powerset indices
    """
    y = np.zeros(X.shape[0])

    # Over all vectors
    for i in range(X.shape[0]):

        # Find corresponding powerset index
        for j in range(B.shape[0]):
            if binvec_are_equal(X[i,:], B[j,:]):
                y[i] = j
                break
    return y


def to_graph(l):
    """ Turn the list into a graph.
    """
    G = networkx.Graph()
    for part in l:
        # Each sublist is a set of nodes
        G.add_nodes_from(part)
        # It also gives the number of edges
        G.add_edges_from(to_edges(part))
    return G


def to_edges(l):
    """ treat `l` as a Graph and returns it's edges 

    Examples:
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current    


def merge_connected(lists):
    """ Merge sets with common elements (find connected graphs problem).
    
    Examples:
        Input:  [{0, 1}, {0, 1}, {2, 3}, {2, 3}, {4, 5}, {4, 5}, {6, 7}, {6, 7}, {8, 9}, {8, 9}, {10}, {11}]
        Output: [{0, 1}, {2, 3}, {4, 5}, {6, 7}, {8, 9}, {10}, {11}]
    """

    sets = [set(lst) for lst in lists if lst]
    merged = True
    while merged:
        merged = False
        results = []

        while sets:
            common, rest = sets[0], sets[1:]
            sets = []

            for x in rest:

                # Two sets are said to be disjoint sets if they have no common elements
                if x.isdisjoint(common):
                    sets.append(x)
                else:
                    merged = True
                    common |= x
            results.append(common)
        sets = results
    return sets


def los2lol(listOsets):
    """ Convert a list of sets [{},{},..,{}] to a list of of lists [[], [], ..., []].
    """
    lists = []
    for i in listOsets:
        lists.append(list(i))
    return lists


def bin_array(num, N):
    """ Convert a positive integer num into an N-bit bit vector.
    """
    return np.array(list(np.binary_repr(num).zfill(N))).astype(dtype=np.uint8)


def binomial(n,k):
    """ Binomial coefficient C(n,k).  
    """
    return np.int64(math.factorial(n) / (math.factorial(k) * math.factorial(n-k)))


def generatebinary_fixed(n,k):
    """ Generate all combinations of n bits with fixed k ones.
    """

    # Initialize
    c = [0] * (n - k) + [1] * k

    X = np.zeros(shape=(binomial(n,k), n), dtype=np.uint8)
    X[0,:] = c

    z = 1
    while True:

        # Find the right-most [0,1] AND keep count of ones
        i = n - 2
        ones = 0
        while i >= 0 and c[i:i+2] != [0,1]:
            if c[i+1] == 1:
                ones += 1
            i -= 1
        if i < 0:
            break

        # Change the 01 to 10 and reset the suffix to the smallest
        # lexicographic string with the right number of ones and zeros
        c[i:] = [1] + [0] * (n - i - ones - 1) + [1] * ones
        
        # Save it
        X[z,:] = c
        z += 1

    return X


def generatebinary(N, M=None, verbose=False):
    """ Function to generate all 2**N binary vectors (as boolean matrix rows)
        with 1 <= M <= N number of ones (hot bits) (default N)
    """

    if M is None: M = N
    if (M < 1) | (M > N): 
        raise Exception(f'generatebinary: M = {M} cannot be less than 1 or greater than N = {N}')

    # Count the number of vectors (rows) needed using binomial coefficients
    K = 1
    for k in range(1,M+1):
        K += binomial(N,k)

    if verbose:
        print(__name__ + f'.generatebinary: Binary matrix dimension {K} x {N}')

    X = np.zeros((K, N), dtype=np.uint8)
    ivals = np.zeros(K, dtype = np.double)

    # Generate up to each m separately here, then sort
    i = 0
    for m in range(0,M+1):
        Y = generatebinary_fixed(N,m)
        for z in range(Y.shape[0]):
            X[i,:] = Y[z,:]
            ivals[i] = bin2int(X[i,:])
            i += 1

    # Sort them to lexicographic order
    lexind = np.argsort(ivals)
    return X[lexind,:]


def bin2int(b):
    """ Binary vector to integer.
    """
    base = int(2)
    if len(b) > 63: # Doubles for large number of bits
        base = np.double(base)

    return b.dot(base**np.arange(b.size)[::-1])


def binom_coeff_all(N, MAX = None):
    """ Sum all all binomial coefficients up to MAX.
    """
    B = generatebinary(N, MAX)
    s = np.sum(B, axis=1)
    c = np.zeros(N+1, dtype=np.int64)

    for i in range(N+1):
        c[i] = np.sum(s == i)
    return c


def binaryvec2int(X):
    """ Turn a matrix of binary vectors row-by-row into integer reps.
    """

    if X.shape[1] > 63:
        # double because we may have over 63 bits
        Y = np.zeros(X.shape[0], dtype=np.double)
    else:
        Y = np.zeros(X.shape[0], dtype=np.int)

    for i in range(len(Y)):
        Y[i] = bin2int(X[i,:])
    return Y


def int2onehot(Y, num_classes):
    """ Integer class vector to class "one-hot encoding"

    Args:
        Y:         Class indices (# samples)
        num_classes: Number of classes

    Returns:
        onehot:    Onehot representation
    """
    onehot = np.zeros(shape=(len(Y), num_classes), dtype=np.bool_)
    for i in range(onehot.shape[0]):
        onehot[i, int(Y[i])] = 1
    return onehot


@numba.njit
def deltaphi(phi1, phi2):
    """ Deltaphi measure. """
    return np.mod(phi1 - phi2 + np.pi, 2*np.pi) - np.pi


@numba.njit
def deltar(eta1,eta2, phi1,phi2):
    """ DeltaR measure. """
    return np.sqrt((eta1 - eta2)**2 + deltaphi(phi1,phi2)**2)


def create_model_filename(path, label, epoch, filetype):

    def createfilename(i):
        return path + '/' + label + '_' + str(i) + filetype

    # Loop over epochs
    i = 0
    last_found  = -1
    while True:
        filename = createfilename(i)
        if os.path.exists(filename):
            last_found = i
        else:
            break
        i += 1

    epoch = last_found if epoch == -1 else epoch
    return createfilename(epoch)


def pick_ind(x, minmax):
    """ Return indices between minmax[0] <= x < minmax[1], i.e. [a,b)
    
    Args:
        x :      Input vector
        minmax : Minimum and maximum values
    Returns:
        indices
    """
    return (minmax[0] <= x) & (x < minmax[1])


def multiclass_roc_auc_score(y_true, y_pred, weights=None, average="macro"):
    """ Multiclass AUC (area under the curve).

    Args:
        y_true : True classifications
        y_pred : Soft probabilities per class
        weights: Sample weights 
        average: Averaging strategy
    Returns:
        auc    : Area under the curve via averaging
    """

    lb = sklearn.preprocessing.LabelBinarizer()
    lb.fit(y_true)
    y_true = lb.transform(y_true)
    y_pred = lb.transform(y_pred)
    
    auc = sklearn.metrics.roc_auc_score(y_true, y_pred, sample_weights=weights, average=average)
    return auc


class Metric:
    """
    Classifier performance evaluation metrics.
    """
    def __init__(self, y_true, y_pred, weights=None, num_classes=2, hist=True, valrange='prob', N_mva_bins=30, verbose=True):
        """
        Args:
            y_true     : true classifications
            y_pred     : predicted probabilities per class (N x 1), (N x 2) or (N x K) dimensional array
            weights    : event weights
            num_classes: number of classses
            
            hist       : histogram soft decision values
            valrange   : histogram range selection type
            N_mva_bins : number of bins
    
        Returns:
            metrics, see the source code for details
        """

        self.num_classes = num_classes

        # Transform N x 2 to N x 1 (pick class[1] probabilities as the signal)
        if (num_classes == 2) and (np.squeeze(y_pred).ndim == 2):
            y_pred = y_pred[:,-1]

        # Make sure the weights array is 1-dimensional, not sparse array of (events N) x (num class K)
        if (weights is not None) and (np.squeeze(weights).ndim > 1):
            weights = np.sum(weights, axis=1)
        
        # Check numerical validity
        if (np.squeeze(y_pred).ndim > 1):
            ok = np.isfinite(np.sum(y_pred,axis=1))
        else:
            ok = np.isfinite(y_pred)

        y_true = y_true[ok]
        y_pred = y_pred[ok]
        if weights is not None:
            weights = weights[ok]
        
        # Invalid input
        if len(np.unique(y_true)) <= 1:
            if verbose:
                print(__name__ + f'.Metric: only one class present in y_true, cannot evaluate metrics (return -1)')
            self.acc = -1
            self.auc = -1
            self.fpr = -1
            self.tpr = -1
            self.thresholds = -1

            self.mva_bins = []
            self.mva_hist = []

            return # Return None
            
        if hist is True:
            
            # Bin the soft prediction values
            if   valrange == 'prob':
                valrange = [0.0, 1.0]
            elif valrange == 'auto':
                valrange = [np.percentile(y_pred, 1), np.percentile(y_pred, 99)]
            else:
                raise Exception('Metric: Unknown valrange parameter')

            self.mva_bins = np.linspace(valrange[0], valrange[1], N_mva_bins)
            self.mva_hist = []
            
            for c in range(num_classes):
                ind    = (y_true == c)
                counts = []
                
                if np.sum(ind) != 0:
                    w = weights[ind] if weights is not None else None
                    x = y_pred[ind] if num_classes == 2 else y_pred[ind,c]
                    counts, edges = np.histogram(x, weights=w, bins=self.mva_bins)
                self.mva_hist.append(counts)
        else:
            self.mva_bins = None
            self.mva_hist = None

        # ------------------------------------
        # Compute Metrics
        
        self.acc = None
        self.auc = None
        self.fpr = None
        self.tpr = None
        self.thresholds = None
        
        if  num_classes == 2:
            self.fpr, self.tpr, self.thresholds = metrics.roc_curve(y_true=y_true, y_score=y_pred, sample_weight=weights)
            self.auc = metrics.roc_auc_score(y_true=y_true,  y_score=y_pred, sample_weight=weights)
            self.acc = metrics.accuracy_score(y_true=y_true, y_pred=np.round(y_pred), sample_weight=weights)
        else:
            self.auc = metrics.roc_auc_score(y_true=y_true, y_score=y_pred, sample_weight=None, \
                        average="weighted", multi_class='ovo', labels=np.arange(num_classes))
            self.acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred.argmax(axis=1), sample_weight=weights)
