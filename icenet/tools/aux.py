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

import os

from sklearn import metrics
from scipy import stats
import scipy.special as special

import icenet.tools.prints as prints
import numba


def apply_algebra_operator(a, ope, b, ope_lhs = ''):
    """
    Algebraic operators applied
    
    Args:
        a       : left hand side (string)
        ope     : algebraic operator (string)
        b       : right hand side (bool, float)
        ope_lhs : operator applied on left hand side first (e.g. 'abs')
    """

    # Left hand side
    if ope_lhs == 'abs':
        f = lambda x : np.abs(x)
    else:
        f = lambda x : x

    # Algebra
    if   ope == '<':
        g = lambda x,y : x < y
    elif ope == '>':
        g = lambda x,y : x > y
    elif ope == '!=':
        g = lambda x,y : x != y
    elif ope == '==':
        g = lambda x,y : x == y
    elif ope == '<=':
        g = lambda x,y : x <= y
    elif ope == '>=':
        g = lambda x,y : x >= y
    else:
        raise Exception(f'Unknown algebraic operator "{ope}"')

    return g(f(a), b)


def construct_cut_tuplets(cutlist):
    """
    Construct cuts 4-tuplets from a list of strings.
    
    Args:
        cutlist : For example ['var_y < 0.5', 'var_x == True']

    Returns:
        list of 4-tuplets of cuts (var, operator, value, lhs_operator)
    """
    tuplets = []

    for s in cutlist:
            
        # Split into [a <operator> b]
        splitted = s.split()

        if len(splitted) != 3:
            raise Except(__name__ + f'.construct_cut_triplets: Problem parsing cut string {s} [len(s) != 3]')

        var   = splitted[0]

        # Construct (possible) left hand side operators
        if var[0] == '|' and var[-1] == '|':
            var     = var[1:-1]
            lhs_ope = 'abs'
        else:
            var     = var
            lhs_ope = ''

        # Middle operator
        ope   = splitted[1]

        # RHS value
        value = splitted[2]
        if   (value == 'True')  or (value == 'true'):
            value = True
        elif (value == 'False') or (value == 'false'):
            value = False
        else:
            value = float(value)

        trp = (var, ope, value, lhs_ope)
        tuplets.append(trp)

    return tuplets


def construct_columnar_cuts(X, VARS, cutlist):
    """
    Construct cuts and corresponding names.

    Args:
        X       : Input columnar data matrix
        VARS    : Variable names for each column of X
        cutlist : Selection cuts as strings, such as ['|eta| < 0.5', 'trigger0 == True']
    
    Returns:
        cuts, names
    """
    cuts    = []
    names   = []
    tuplets = construct_cut_tuplets(cutlist)

    for tup in tuplets:

        if len(tup) != 4:
            raise Exception(__name__ + f'.construct_columnar_cuts: Problem with the input structure.')

        # Apply Algebraic Operators
        var     = tup[0]
        ope     = tup[1]
        value   = tup[2]
        ope_lhs = tup[3]

        cuts.append( apply_algebra_operator(a=X[:, VARS.index(var)], ope=ope, b=value, ope_lhs=ope_lhs))
        names.append(f'{ope_lhs}({var}) {ope} {value}')

    return cuts,names


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


def count_targets(events, names, entrystart=0, entrystop=None, new=False):
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
    if new:
        vec = events.arrays(names, library="np", how=list, entry_start=entrystart, entry_stop=entrystop)
        vec = np.asarray(vec)
    else:
        vec = np.array([events.array(name, entrystart=entrystart, entrystop=entrystop) for name in names])
    vec = vec.T
    
    print(__name__ + f'.count_targets: vec.shape = {vec.shape}')

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
        weights   : array of weights
        Y         : targets
        N_classes : number of classes

    """
    one_hot_weights = np.zeros((len(weights), N_classes))
    for i in range(N_classes):
        try:
            one_hot_weights[Y == i, i] = weights[Y == i]
        except:
            print(__name__ + f'weight2onehot: Failed with class = {i} (zero samples)')
    
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


def load_torch_checkpoint(path='/', label='mynet', epoch=-1):
    """ Load pytorch checkpoint

    Args:
        path  : folder path
        label : model label name
        epoch : epoch index. Use -1 for the last epoch
    
    Returns:
        pytorch model
    """

    filename = create_model_filename(path=path, label=label, epoch=epoch, filetype='.pth')
    print(__name__ + f'.load_torch_checkpoint: Loading checkpoint {filename}')

    # Load the model
    checkpoint = torch.load(filename)
    model      = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval() # Set it in the evaluation mode
    return model


def save_torch_model(model, optimizer, epoch, filename):
    """ PyTorch model saver
    """
    def f():
        print('Saving model..')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, (filename))
    return f


def load_torch_model(model, optimizer, filename, load_start_epoch = False):
    """ PyTorch model loader
    """
    def f():
        print('Loading model..')
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        if load_start_epoch:
            param.start_epoch = checkpoint['epoch']
    return f


def reweight_1D(X, pdf, y, N_class=2, reference_class = 0, max_reg = 1E3, EPS=1E-12) :
    """ Compute N-class density reweighting coefficients.
    Args:
        X   :             Input data (# samples)
        pdf :             Dictionary of pdfs for each class
        y   :             Class target data (# samples)
        N_class :         Number of classes
        reference_class : Target class of re-weighting
    
    Returns:
        weights for each event
    """

    # Re-weighting weights
    weights_doublet = np.zeros((X.shape[0], N_class)) # Init with zeros!!

    # Weight each class against the reference class
    for c in range(N_class):
        inds = x2ind(X[y == c], pdf['binedges'])
        if c is not reference_class:
            weights_doublet[y == c, c] = pdf[reference_class][inds] / (pdf[c][inds] + EPS)
        else:
            weights_doublet[y == c, c] = 1 # Reference class stays intact

    # Maximum weight cut-off regularization
    weights_doublet[weights_doublet > max_reg] = max_reg

    # Save weights
    weights_doublet[y == 0, 0] = C0
    weights_doublet[y == 1, 1] = C1

    return weights_doublet


def reweightcoeff1D(X, y, pdf, N_class=2, reference_class = 0, equal_frac = True, max_reg = 1e3) :
    """ Compute N-class density reweighting coefficients.
    
    Args:
        X  :   Observable of interest (N x 1)
        y  :   Class labels (0,1,...) (N x 1)
        pdf:   PDF for each class
        N_class : Number of classes
        equal_frac:  equalize class fractions
        reference_class : e.g. 0 (background) or 1 (signal)
    
    Returns:
        weights for each event
    """
    weights_doublet = reweight_1D(X=X, pdf=pdf, y=y, N_class=N_class, reference_class=reference_class, max_reg=max_reg)

    # Apply class balance equalizing weight
    if (equal_frac == True):
        weights_doublet = balanceweights(weights_doublet=weights_doublet, y=y)

    # Get 1D array
    weights = np.sum(weights_doublet, axis=1)

    return weights


def reweightcoeff2DFP(X_A, X_B, y, pdf_A, pdf_B, N_class=2, reference_class = 0,
    equal_frac = True, max_reg = 1e3) :
    """ Compute N-class density reweighting coefficients.
    
    Operates in 2D with FACTORIZED PRODUCT marginal 1D distributions.
    
    Args:
        X_A   :  Observable of interest (N x 1)
        X_B   :  Observable of interest (N x 1)
        y     :  Signal (1) and background (0) targets
        pdf_A :  Density of observable A
        pdf_B :  Density of observable B
        N_class: Number of classes
        reference_class: e.g. 0 (background) or 1 (signal)
        equal_frac:      Equalize integrated class fractions
        max_reg:         Maximum weight regularization
    
    Returns:
        weights for each event
    """

    weights_doublet_A = reweight_1D(X=X_A, pdf=pdf_A, N_class=N_class, y=y, reference_class=reference_class, max_reg=max_reg)
    weights_doublet_B = reweight_1D(X=X_B, pdf=pdf_B, N_class=N_class, y=y, reference_class=reference_class, max_reg=max_reg)

    # Factorized product
    weights_doublet   = weights_doublet_A * weights_doublet_B

    # Apply class balance equalizing weight
    if (equal_frac == True):
        weights_doublet = balanceweights(weights_doublet=weights_doublet, reference_class=reference_class, y=y)

    # Get 1D array
    weights = np.sum(weights_doublet, axis=1)

    return weights


def reweightcoeff2D(X_A, X_B, y, pdf, N_class=2, reference_class = 0, equal_frac = True, max_reg = 1e3, EPS=1E-12) :
    """ Compute N-class density reweighting coefficients.
    
    Operates in full 2D without factorization.

    Args:
        X_A : Observable A of interest (N x 1)
        X_B : Observable B of interest (N x 1)
        y   : Signal (1) and background (0) labels (N x 1)
        pdf : Density histograms for each class
        N_class :         Number of classes
        reference_class : e.g. Background (0) or signal (1)
        equal_frac :      Equalize class fractions
        max_reg :         Regularize the maximum reweight coefficient
    
    Returns:
        weights for each event
    """
    
    # Re-weighting weights
    weights_doublet = np.zeros((X_A.shape[0], N_class)) # Init with zeros!!

    # Weight each class against the reference class
    for c in range(N_class):
        inds_A = x2ind(X_A[y == c], pdf['binedges_A'])
        inds_B = x2ind(X_B[y == c], pdf['binedges_B'])
        if c is not reference_class:
            weights_doublet[y == c, c] = pdf[reference_class][inds_A, inds_B] / (pdf[c][inds_A, inds_B] + EPS)
        else:
            weights_doublet[y == c, c] = 1 # Reference class stays intact

    # Maximum weight cut-off regularization
    weights_doublet[weights_doublet > max_reg] = max_reg

    # Apply class balance equalizing weight
    if (equal_frac == True):
        weights_doublet = balanceweights(weights_doublet=weights_doublet, reference_class=reference_class, y=y)

    # Get 1D array
    weights = np.sum(weights_doublet, axis=1)
    return weights


def pdf_1D_hist(X, binedges):
    """ 
    Compute re-weighting 1D pdfs.
    """

    # Take re-weighting variables
    pdf,_,_ = plt.hist(x = X, bins = binedges)

    # Make them densities
    pdf  /= np.sum(pdf.flatten())
    return pdf


def pdf_2D_hist(X_A, X_B, binedges_A, binedges_B):
    """
    Compute re-weighting 2D pdfs.
    """

    # Take re-weighting variables
    pdf,_,_,_ = plt.hist2d(x = X_A, y = X_B, bins = [binedges_A, binedges_B])

    # Make them densities
    pdf  /= np.sum(pdf.flatten())
    return pdf


@numba.njit
def balanceweights(weights_doublet, reference_class, y, EPS=1e-12):
    """ Balance N-class weights to sum to equal counts.
    
    Args:
        weights_doublet: N-class event weights (events x classes)
        reference_class: which class gives the reference (integer)
        y : class targets
    Returns:
        weights doublet with new weights per event
    """
    N = weights_doublet.shape[1]
    ref_sum = np.sum(weights_doublet[(y == reference_class), reference_class])

    for i in range(N):
        if i is not reference_class:
            EQ = ref_sum / (np.sum(weights_doublet[y == i, i]) + EPS)
            weights_doublet[y == i, i] *= EQ

    return weights_doublet


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


