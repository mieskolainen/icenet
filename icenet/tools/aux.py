# Auxialary functions
# 
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import copy
from tqdm import tqdm

from sklearn import metrics
from scipy import stats
import scipy.special as special

import icenet.tools.prints as prints
import numba


def apply_cutflow(cut, names, xcorr_flow=True):
    """ Apply cutflow

    Args:
        cut        : list of pre-calculated cuts, each is a boolean array
        names      : list of names (description of each cut, for printout only)
        xcorr_flow : compute full N-point correlations
    
    Returns:
        ind : list of indices, 1 = pass, 0 = fail
    """
    print(__name__ + '.apply_cutflow: \n')

    # Print out "serial flow"
    N   = len(cut[0])
    ind = np.ones(N, dtype=np.uint8)
    for i in range(len(cut)):
        ind = np.logical_and(ind, cut[i])
        print(f'cut[{i}][{names[i]:>25}]: pass {np.sum(cut[i]):>10}/{N} = {np.sum(cut[i])/N:.4f} | total = {np.sum(ind):>10}/{N} = {np.sum(ind)/N:0.4f}')
    
    # Print out "parallel flow"
    if xcorr_flow:
        print('\n')
        print(__name__ + '.apply_cutflow: Computing N-point correlations <xcorr_flow = True>')
        vec = np.zeros((len(cut[0]), len(cut)))
        for j in range(vec.shape[1]):
            vec[:,j] = np.array(cut[j])

        intmat = binaryvec2int(vec)
        BMAT   = generatebinary(vec.shape[1])
        print(f'Boolean combinations for {names}: \n')
        for i in range(BMAT.shape[0]):
            print(f'{BMAT[i,:]} : {np.sum(intmat == i):>10} ({np.sum(intmat == i) / len(intmat):.4f})')
        print('\n')
    
    return ind


def count_targets(events, names, entrystart=0, entrystop=None):
    """ Targets statistics printout

    Args:
        events :     uproot object
        names  :     list of branch names
        entrystart : uproot starting point
        entrystop  : uproot ending point

    Returns:
        Printout on stdout
    """
    K   = len(names)
    vec = np.array([events.array(name, entrystart=entrystart, entrystop=entrystop) for name in names])
    vec = vec.T

    intmat = binaryvec2int(vec)
    BMAT   = generatebinary(K)
    print(__name__ + f'.count_targets: {names}')
    for i in range(BMAT.shape[0]):
        print(f'{BMAT[i,:]} : {np.sum(intmat == i):>10} ({np.sum(intmat == i) / len(intmat):.4f})')
    
    return


def longvec2matrix(X, M, D, order='F'):
    """ A matrix representation / dimension converter function.
    
    Args:
        X:     Input matrix
        M:     Number of set elements
        D:     Feature dimension
        order: Reshape direction

    Returns:
        Y:     Output matrix

    Examples:
        X = [# number of samples N ] x [# M x D long feature vectors]
        -->
        Y = [# number of samples N ] x [# number of set elements M] x [# vector dimension D]
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


def weight2onehot(weights, Y, N_classes):
    """
    Weights into one-hot encoding.
    Args:
        weights   :
        Y         : targets
        N_classes : number of classes

    """
    one_hot_weights = np.zeros((len(weights), N_classes))
    for i in range(N_classes):
        one_hot_weights[Y == i, i] = weights[Y == i]

    return one_hot_weights


def int2onehot(Y, N_classes):
    """ Integer class vector to class "one-hot encoding"

    Args:
        Y:         Class indices (# samples)
        N_classes: Number of classes

    Returns:
        onehot:    Onehot representation
    """
    onehot = np.zeros(shape=(len(Y), N_classes), dtype=np.bool_)
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


def load_torch_checkpoint(filepath) :
    """ Load pytorch checkpoint
    """
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

def save_torch_model(model, optimizer, epoch, path):
    """ PyTorch model saver
    """
    def f():
        print('Saving model..')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, (path))
    
    return f


def load_torch_model(model, optimizer, param, path, load_start_epoch = False):
    """ PyTorch model loader
    """
    def f():
        print('Loading model..')
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        if load_start_epoch:
            param.start_epoch = checkpoint['epoch']

    return f


def reweight_aux(X, y, binedges, shape_reference = 'signal', max_reg = 1E3, EPS=1E-12) :
    """ Compute reweighting coefficients for 2-classes.
    Args:
        X  :              Input data (# samples)
        y  :              Class target data (# samples)
        binedges :        One dimensional histogram edges
        shape_reference : Target class of re-weighting
    Returns:
        weights_doublet:  Re-weight coefficients for two classes (# vectors x 2)
    """

    # Re-weighting weights
    weights_doublet = np.zeros((X.shape[0], 2)) # Init with zeros!!

    # Take re-weighting variables
    pdf0, bins, patches = plt.hist(x = X[y == 0], bins = binedges)
    pdf1, bins, patches = plt.hist(x = X[y == 1], bins = binedges)

    # Make them densities
    pdf0 = pdf0 / np.sum(pdf0)
    pdf1 = pdf1 / np.sum(pdf1)

    # Indexing
    inds = x2ind(X[y == 0], binedges)
    C0 = np.ones((inds.shape[0]))
    if (shape_reference == 'signal'):
        C0 = pdf1[inds] / (pdf0[inds] + EPS)

    inds = x2ind(X[y == 1], binedges)
    C1 = np.ones((inds.shape[0]))
    if (shape_reference == 'background'):
        C1 = pdf0[inds] / (pdf1[inds] + EPS)

    if (shape_reference == 'none'):
        C0 = C1 = 1

    # Maximum threshold regularization
    weights_doublet[weights_doublet > max_reg] = max_reg

    # Save weights
    weights_doublet[y == 0, 0] = C0
    weights_doublet[y == 1, 1] = C1

    return weights_doublet


@numba.njit
def balanceweights(weights_doublet, y, EPS=1e-12):
    """ Balance class weights to sum to equal counts.
    """
    EQ = np.sum(weights_doublet[y == 0, 0]) / (np.sum(weights_doublet[y == 1, 1]) + EPS)
    weights_doublet[y == 1,1] *= EQ

    return weights_doublet


def reweightcoeff1D(X, y, binedges, shape_reference = 'signal', equalize_classes = True, max_reg = 1e3) :
    """ Compute TRAINING re-weighting coefficients for each vector.
    
    Args:
        X: Observable of interest (N x 1)
        y: Signal (1) and background (0) labels
        shape_reference: 'signal' or 'background' or 'none'

    Returns:
        weights
    """
    weights_doublet = reweight_aux(X, y, binedges, shape_reference, max_reg)

    # Apply class balance equalizing weight
    if (equalize_classes == True):
        weights_doublet = balanceweights(weights_doublet, y)

    # Get 1D array
    weights = np.sum(weights_doublet, axis=1)

    return weights


def reweightcoeff2DFP(X_A, X_B, y, binedges_A, binedges_B, shape_reference = 'signal', equalize_classes = True, max_reg = 1e3) :
    """ Compute TRAINING re-weighting coefficients for each vector.
    
    Operates in 2D with factorized marginal 1D distributions.
    
    Args:
        X_A : Observable of interest (N x 1)
        X_B : Observable of interest (N x 1)
        y   : Signal (1) and background (0) labels
        shape_reference : 'signal' or 'background' or 'none'

    Returns:
        w   :  array of weights
    """

    weights_doublet_A = reweight_aux(X_A, y, binedges_A, shape_reference, max_reg)
    weights_doublet_B = reweight_aux(X_B, y, binedges_B, shape_reference, max_reg)

    weights_doublet   = weights_doublet_A * weights_doublet_B

    # Apply class balance equalizing weight
    if (equalize_classes == True):
        weights_doublet = balanceweights(weights_doublet, y)

    # Get 1D array
    weights = np.sum(weights_doublet, axis=1)

    return weights


def reweightcoeff2D(X_A, X_B, y, binedges_A, binedges_B, shape_reference = 'signal', equalize_classes = True, max_reg = 1e3, EPS=1E-12) :
    """ Compute TRAINING re-weighting coefficients for each vector.

    Operates in full 2D without factorization.

    Args:
        X_A : Observable of interest (N x 1)
        X_B : Observable of interest (N x 1)
        y   : Signal (1) and background (0) labels
        shape_reference : 'signal' or 'background' or 'none'

    Returns:
        w   :  array of weights
    """
    
    # Re-weighting weights
    weights_doublet = np.zeros((X_A.shape[0], 2)) # Init with zeros!!

    # Take re-weighting variables
    pdf0, foo, bar, zoo = plt.hist2d(x = X_A[y == 0], y = X_B[y == 0], bins = [binedges_A, binedges_B])
    pdf1, foo, bar, zoo = plt.hist2d(x = X_A[y == 1], y = X_B[y == 1], bins = [binedges_A, binedges_B])

    # Make them densities
    pdf0 = pdf0 / np.sum(pdf0)
    pdf1 = pdf1 / np.sum(pdf1)
    
    # Indexing
    inds_A = x2ind(X_A[y == 0], binedges_A)
    inds_B = x2ind(X_B[y == 0], binedges_B)

    C0 = np.ones((inds_A.shape[0]))
    if (shape_reference == 'signal'):
        C0 = pdf1[inds_A, inds_B] / (pdf0[inds_A, inds_B] + EPS)

    inds_A = x2ind(X_A[y == 1], binedges_A)
    inds_B = x2ind(X_B[y == 1], binedges_B)

    C1 = np.ones((inds_A.shape[0]))
    if (shape_reference == 'background'):
        C1 = pdf0[inds_A, inds_B] / (pdf1[inds_A, inds_B] + EPS)

    if (shape_reference == 'none'):
        C0 = C1 = 1

    # Save weights
    weights_doublet[y == 0, 0] = C0
    weights_doublet[y == 1, 1] = C1

    # Maximum threshold regularization
    weights_doublet[weights_doublet > max_reg] = max_reg

    # Apply class balance equalizing weight
    if (equalize_classes == True):
        weights_doublet = balanceweights(weights_doublet, y)

    # Get 1D array
    weights = np.sum(weights_doublet, axis=1)

    return weights


def pick_ind(x, minmax):
    """ Return indices between minmax[0] and minmax[1].

    Args:
        x :      Input vector
        minmax : Minimum and maximum values
    Returns:
        indices
    """
    return (x >= minmax[0]) & (x <= minmax[1])

def jagged2tensor(X, VARS, xyz, x_binedges, y_binedges):
    """
    Args:
        
        X          : input data (samples x dimensions) with jagged structure
        VARS       : all variable names
        xyz        : array of (x,y,z) channel triplet strings such as [['image_clu_eta', 'image_clu_phi', 'image_clu_e']]
        x_binedges
        y_binedges : arrays of bin edges
    
    Returns:
        T : tensor of size (samples x channels x rows x columns)
    """

    # Samples x Channels x Rows x Columns
    T = np.zeros((X.shape[0], len(xyz), len(x_binedges)-1, len(y_binedges)-1), dtype=np.float)

    # Choose targets
    for c in range(len(xyz)):
        ind = [VARS.index(x) for x in xyz[c]]

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

    # Loop and sum
    A  = np.zeros((len(x_binedges)-1, len(y_binedges)-1), dtype=np.float)
    try:
        for i in range(len(x_ind)):
            A[x_ind[i], y_ind[i]] += z_arr[i]
    except:
        print(__name__ + f'.arrays2matrix: not valid input')

    return A

def x2ind(x, binedges) :
    """ Return histogram bin indices for data in x, which needs to be an array [].
    Args:
        x:        data to be classified between bin edges
        binedges: histogram bin edges
    Returns:
        inds:     histogram bin indices
    """
    NBINS = len(binedges) - 1
    inds = np.digitize(x, binedges, right=True) - 1

    if len(x) > 1:
        inds[inds >= NBINS] = NBINS-1
        inds[inds < 0] = 0
    else:
        if inds < 0:
            inds = 0
        if inds >= NBINS:
            inds = NBINS - 1

    return inds


def hardclass(y_soft, valrange = [0,1]):
    """ Soft decision to hard decision at point (valrange[1] - valrange[0]) / 2
    
    Args:
        y_soft : probabilities for two classes
    Returns:
        y_out  : classification results
    """

    y_out = copy.deepcopy(y_soft)

    boundary = (valrange[1] - valrange[0]) / 2
    y_out[y_out  > boundary] = 1
    y_out[y_out <= boundary] = 0

    return y_out

def multiclass_roc_auc_score(y_true, y_soft, average="macro"):
    """ Multiclass AUC (area under the curve).

    Args:
        y_true : True classifications
        y_soft : Soft probabilities
        average: Averaging strategy
    Returns:
        auc    : Area under the curve via averaging
    """

    lb = sklearn.preprocessing.LabelBinarizer()
    lb.fit(y_true)
    y_true = lb.transform(y_true)
    y_soft = lb.transform(y_soft)
    
    auc = sklearn.metrics.roc_auc_score(y_true, y_soft, average=average)
    return auc


class Metric:
    """ Classifier performance evaluation metrics.
    """
    def __init__(self, y_true, y_soft, valrange = [0,1]) :
        """
        Args:
            y_true   : true classifications
            y_soft   : probabilities for two classes
            valrange : range of probabilities / soft scores
        """
        ok = np.isfinite(y_true) & np.isfinite(y_soft)
        
        lhs = len(y_true) 
        rhs = (ok == True).sum()
        if (lhs != rhs) :
            print('Metric: input length = {} with not-finite values = {}'.format(lhs, lhs-rhs))
            print(y_soft)

        # invalid input
        if (np.sum(y_true == 0) == 0) | (np.sum(y_true == 1) == 0):
            print('Metric: only one class present in y_true, cannot evaluate metrics (set all == -1)')
            self.fpr = -1
            self.tpr = -1
            self.thresholds = -1
            self.auc = -1
            self.acc = -1
            return

        self.fpr, self.tpr, self.thresholds = metrics.roc_curve(y_true = y_true[ok], y_score = y_soft[ok])
        self.auc = metrics.roc_auc_score(y_true  = y_true[ok], y_score = y_soft[ok])
        self.acc = metrics.accuracy_score(y_true = y_true[ok], y_pred = hardclass(y_soft = y_soft[ok], valrange = valrange))


