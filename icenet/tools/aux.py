# Auxialary functions
# 
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

import math
import numpy as np
import awkward as ak
import re
import time

import numba
import copy
from tqdm import tqdm
import os
from termcolor import cprint

import sklearn
from sklearn import metrics
from scipy import stats
import scipy.special as special

import multiprocessing
import ctypes


from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support


import icenet.tools.prints as prints
import icenet.tools.stx as stx


def slice_range(start, stop, N):
    """
    Python slice type processor function

    Args:
        start:   first index
        stop:    end index + 1
        N:       total number of indices

    Returns:
        a,b,b-a: processed indices and total length
    """

    if start is None:
        a = 0
    elif start < 0: # count from the end
        a = N - start
    else:
        a = start

    if stop is None:
        b = N
    elif stop < 0: # count from the end
        b = N - stop
    else:
        b = stop
    
    return a, b, b-a


def red(X, ids, param, mode='X', exclude_tag='exclude_MVA_vars', include_tag='include_MVA_vars'):
    """
    Reduce the input set variables of X
     
    Args:
        X:           data matrix
        ids:         names of columns
        param:       parameter dictionary (from yaml)
        mode:        return mode 'X' or 'ids'
        exclude_tag: key in param
        include_tag: key in param
    """
    mask = np.ones(len(ids), dtype=np.bool_)

    for boolean in [0,1]:

        tag = exclude_tag if boolean == 0 else include_tag

        if tag in param:
            # Compile regexp
            var_names = process_regexp_ids(all_ids=ids, ids=param[tag])
            
            # Boolean flag each variable found
            for var in var_names:
                for i in range(len(ids)):
                    if ids[i] == var:
                        mask[i] = boolean
    
    # Variable set is reduced
    if np.sum(mask) != len(ids):
        cprint(__name__ + f'.red: Using the reduced set of variables: {np.array(ids)[mask].tolist()}', 'red')
    else:
        cprint(__name__ + f'.red: Using the full set of input variables', 'red')
    
    if mode == 'X':
        return X[:, mask]
    else:
        return np.array(ids)[mask].tolist()


def unroll_ak_fields(x, order='first'):
    """
    Unroll field names in a (nested) awkward array

    Args:
        x   : awkward array
        type: return first order and second order field names 
    Returns:
        field names as a list
    """
    all_fields = []
    for key in x.fields: # x.fields returns only the first order fields
        
        fields = x[key].fields

        # Non-nested entry
        if order == 'first' and fields == []:
            all_fields += [key]

        # Contains second order sub-fields
        if order == 'second' and fields != []:
            
            # Add of type "first_second"
            extended = []
            for sub_field in fields:
                extended += [f'{key}_{sub_field}']
            all_fields += extended

    return all_fields


def process_regexp_ids(all_ids, ids=None):
    """
    Process regular expressions for variable names

    Args:
        all_ids: all keys in a tree
        ids:     keys to pick, if None, use all keys

    Returns:
        ids matching regular expressions
    """

    if ids is None:
        load_ids = all_ids
    else:
        load_ids = []
        chosen   = np.zeros(len(all_ids))

        # Loop over our input
        for string in ids:

            # Compile regular expression
            reg = re.compile(string)
            
            # Loop over all keys in the tree
            for i in range(len(all_ids)):
                if re.fullmatch(reg, all_ids[i]) and not chosen[i]:
                    load_ids.append(all_ids[i])
                    chosen[i] = 1

    return load_ids


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


def jagged_ak_to_numpy(data, scalar_vars, jagged_vars, jagged_maxdim, entry_start=None, entry_stop=None, null_value=float(-999.0)):
    """
    Transform jagged awkward array to fixed dimensional numpy data
    
    Args:
        data:          awkward arrays in a XYW object
        scalar_vars:   Scalar variable names
        jagged_vars:   Jagged variable names
        jagged_maxdim: Maximum dimension per jagged category
        null_value:    Fill null value
    Returns:
        data object
    """
    # Create tuplet expanded jagged variable names
    jagged_dim      = []
    all_jagged_vars = []
        
    for i in range(len(jagged_vars)):
        
        # Split by the first "_" occuring
        dim = int(jagged_maxdim[jagged_vars[i].split('_', 1)[0]])
        jagged_dim.append(dim)
        
        # Create names of type 'varname[j]'
        for j in range(dim):
            all_jagged_vars.append(f'{jagged_vars[i]}[{j}]')

    # Parameters
    arg = {
        'entry_start': entry_start,
        'entry_stop':  entry_stop,
        'scalar_vars': scalar_vars,
        'jagged_vars': jagged_vars,
        'jagged_dim':  jagged_dim,
        'null_value':  null_value
    }

    #matrix = jagged2matrix(data.x, **arg)
    matrix = jagged2matrix_multiprocess(data.x, **arg)
    
    # Cast to numpy arrays
    new_data = copy.deepcopy(data)

    new_data.x   = ak.to_numpy(matrix)
    new_data.y   = ak.to_numpy(data.y)
    new_data.ids = scalar_vars + all_jagged_vars # First scalar, then jagged !

    return new_data


# ------------------------------------------------------------------------
# Multiprocessing

# Global
parfunc = None

def my_func(index):#, def_param=shared_array):
    """
    Helper function for jagged2matrix_multiprocessor()
        
    Args:
        index: Pair of (index for the new array, event number)
    """

    global parfunc

    arr          = parfunc['arr']
    jagged_dim   = parfunc['jagged_dim']
    scalar_vars  = parfunc['scalar_vars']
    jagged_vars  = parfunc['jagged_vars']
    jvname       = parfunc['jvname']
    shared_array = parfunc['shared_array']

    k  = len(scalar_vars)
    i  = index[0]
    ev = index[1]

    ## Jagged variables looped over
    for j in range(len(jagged_vars)):

        x  = ak.to_numpy(arr[ev][jvname[j][0]][jvname[j][1]])
        if len(x) > 0:
            d_this = np.min([len(x), jagged_dim[j]])
            shared_array[i, k:k+d_this] = x[0:d_this]
        
        # Increase block counter
        k += jagged_dim[j]

    # -----------------------------------
    # For printing
    if ev % 10000 == 0:
        print(f'Event: {ev}')
    # -----------------------------------


def jagged2matrix_multiprocess(arr, scalar_vars, jagged_vars, jagged_dim, entry_start=None, entry_stop=None, null_value=float(-999.0)):
    """
    Transform a "jagged" event container to a matrix (rows ~ event, columns ~ variables)
    
    Args:
        arr:           Awkward array type input for N events
        scalar_vars:   Scalar variables to pick (list of strings)
        jagged_vars:   Jagged variables to pick (list of strings)
        jagged_dim:    Maximum dimension per jagged variable (integer array)
        null_value:    Default value for empty ([]) jagged entries
    
    Returns:
        Fixed dimensional 2D-numpy matrix (N x [# scalar var x {#jagged var x maxdim}_i])
    """

    if len(jagged_vars) != len(jagged_dim):
        raise Exception(__name__ + f'.jagged2matrix_multiprocess: len(jagged_vars) != len(jagged_maxdim) {len(jagged_vars)} != {len(jagged_dim)}')

    entry_start, entry_stop, N = slice_range(start=entry_start, stop=entry_stop, N=len(arr))
    D = len(scalar_vars) + int(np.sum(np.array(jagged_dim)))

    print(__name__ + f'.jagged2matrix_multiprocess: Creating a matrix with dimensions [{N} x {D}] ({N*D*64/8/1024**3:0.3f} GB)')

    # Pre-processing of jagged vars
    jvname = []
    for j in range(len(jagged_vars)):
        # Awkward groups jagged variables, e.g. 'sv_x' to sv.x
        jvname.append(jagged_vars[j].split('_', 1)) # argument 1 takes the first '_' occurance

    # Return matrix
    shared_array_base = multiprocessing.Array(ctypes.c_double, N*D)
    shared_array      = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array      = shared_array.reshape(N, D)
    np.copyto(shared_array, np.full((N,D), null_value)) # Initialize values

    ## 1. Pure scalar vars without multiprocessing (very fast)
    for j in range(len(scalar_vars)):
        shared_array[:,j] = ak.to_numpy(ak.ravel(arr[scalar_vars[j]])[entry_start:entry_stop])

    ## 2. Jagged variables with multiprocessing per event
    global parfunc

    parfunc = {}
    parfunc['arr']          = arr
    parfunc['jagged_dim']   = jagged_dim
    parfunc['scalar_vars']  = scalar_vars
    parfunc['jagged_vars']  = jagged_vars
    parfunc['jvname']       = jvname
    parfunc['shared_array'] = shared_array

    # Multiprocess
    num_cores = multiprocessing.cpu_count()
    print(__name__ + f'.jagged2matrix_multiprocess: {num_cores} asynchronous parallel processes launched')
    t = time.time()
    
    pool = multiprocessing.Pool(processes=num_cores)
    pool.map(my_func, zip(range(N), range(entry_start, entry_stop)))
    #pool.map(partial(my_func, b=parfunc), range(N))

    #from joblib import Parallel, delayed
    #from tqdm_joblib import tqdm_joblib

    #with tqdm_joblib(desc="My calculation", total=10) as progress_bar:
    #Parallel(n_jobs=num_cores)(delayed(my_func)(i) for i in range(N))

    elapsed = time.time() - t
    print(__name__ + f'jagged2matrix_multiprocess: Processing took {elapsed:0.1f} sec')
    print(shared_array)

    return shared_array

# ------------------------------------------------------------------------


#@numba.njit
def jagged2matrix(arr, scalar_vars, jagged_vars, jagged_dim, entry_start=None, entry_stop=None, null_value=float(-999.0)):
    """
    Transform a "jagged" event container to a matrix (rows ~ event, columns ~ variables)
    
    Args:
        arr:           Awkward array type input for N events
        scalar_vars:   Scalar variables to pick (list of strings)
        jagged_vars:   Jagged variables to pick (list of strings)
        jagged_dim:    Maximum dimension per jagged variable (integer array)
        null_value:    Default value for empty ([]) jagged entries
    
    Returns:
        Fixed dimensional 2D-numpy matrix (N x [# scalar var x {#jagged var x maxdim}_i])
    """

    if len(jagged_vars) != len(jagged_dim):
        raise Exception(__name__ + f'.jagged2matrix: len(jagged_vars) != len(jagged_maxdim) {len(jagged_vars)} != {len(jagged_dim)}')

    entry_start, entry_stop, N = slice_range(start=entry_start, stop=entry_stop, N=len(arr))
    D = len(scalar_vars) + int(np.sum(np.array(jagged_dim)))
    
    print(__name__ + f'.jagged2matrix: Creating a matrix with dimensions [{N} x {D}] ({N*D*64/8/1024**3:0.3f} GB)')

    # Pre-processing of jagged vars
    jvname = []
    for j in range(len(jagged_vars)):
        # Awkward groups jagged variables, e.g. 'sv_x' to sv.x
        jvname.append(jagged_vars[j].split('_', 1)) # argument 1 takes the first '_' occurance

    # Return matrix
    shared_array = np.full((N,D), null_value)

    ## Pure scalar vars
    for j in range(len(scalar_vars)):
        shared_array[:,j] = ak.to_numpy(ak.ravel(arr[scalar_vars[j]])[entry_start:entry_stop])
    
    t = time.time()
    
    # Loop over events
    for index in tqdm(zip(range(N), range(entry_start, entry_stop))):
        
        k  = len(scalar_vars)
        i  = index[0]
        ev = index[1]

        ## Jagged vars
        for j in range(len(jagged_vars)):

            x  = ak.to_numpy(arr[ev][jvname[j][0]][jvname[j][1]])
            if len(x) > 0:
                d_this = np.min([len(x), jagged_dim[j]])
                shared_array[i, k:k+d_this] = x[0:d_this]
            
            # Increase block counter
            k += jagged_dim[j]

    elapsed = time.time() - t
    print(__name__ + f'jagged2matrix: Processing took {elapsed:0.1f} sec')
    print(shared_array)

    return shared_array


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
    def __init__(self, y_true, y_pred, weights=None, num_classes=2, hist=True, valrange='prob',
        N_mva_bins=30, verbose=True, num_bootstrap=0):
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
        self.acc = -1
        self.auc = -1
        self.fpr = -1
        self.tpr = -1
        self.thresholds = -1
        
        self.tpr_bootstrap = None
        self.fpr_bootstrap = None
        self.auc_bootstrap = None
        self.acc_bootstrap = None

        self.mva_bins = []
        self.mva_hist = []

        # Make sure they are binary (scikit ROC functions cannot handle continuous values)
        y_true = np.round(y_true)

        self.num_classes = num_classes

        # Transform N x 2 to N x 1 (pick class[1] probabilities as the signal)
        if (num_classes == 2) and (np.squeeze(y_pred).ndim == 2):
            y_pred = y_pred[:,-1]

        # Make sure the weights array is 1-dimensional, not sparse array of (events N) x (num class K)
        if (weights is not None) and (np.squeeze(weights).ndim > 1):
            weights = np.sum(weights, axis=1)
        
        # Check numerical validity
        """
        if (np.squeeze(y_pred).ndim > 1):
            ok = np.isfinite(np.sum(y_pred,axis=1))
        else:
            ok = np.isfinite(y_pred)
        
        y_true = y_true[ok]
        y_pred = y_pred[ok]
        if weights is not None:
            weights = weights[ok]
        """
        
        # Invalid input
        if len(np.unique(y_true)) <= 1:
            if verbose:
                print(__name__ + f'.Metric: only one class present cannot evaluate metrics (return -1)')

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
        ## Compute Metrics
        out = compute_metrics(num_classes=num_classes, y_true=y_true, y_pred=y_pred, weights=weights)

        self.acc        = out['acc']
        self.auc        = out['auc']
        self.fpr        = out['fpr']
        self.tpr        = out['tpr']
        self.thresholds = out['thresholds']

        if num_bootstrap > 0:

            self.tpr_bootstrap = (-1)*np.ones((num_bootstrap, len(self.tpr)))
            self.fpr_bootstrap = (-1)*np.ones((num_bootstrap, len(self.fpr)))
            self.auc_bootstrap = (-1)*np.ones(num_bootstrap)
            self.acc_bootstrap = (-1)*np.ones(num_bootstrap)

            for i in range(num_bootstrap):

                # ------------------
                trials = 0
                while True:
                    ind = np.random.choice(range(len(y_true)), size=len(y_true), replace=True)
                    if len(np.unique(y_true[ind])) > 1 or trials > 100: # Protection with very low per class stats
                        break
                    else:
                        trials += 1
                if trials > 100:
                    print(__name__ + f'.Metric: bootstrap fail (check your input statistics)')
                    continue
                # ------------------

                out = compute_metrics(num_classes=num_classes, y_true=y_true[ind], y_pred=y_pred[ind], weights=weights[ind])

                if out['auc'] > 0:

                    self.auc_bootstrap[i] = out['auc']
                    self.acc_bootstrap[i] = out['acc']

                    # Interpolate ROC-curve (re-sample to match the non-bootstrapped x-axis)
                    from scipy import interpolate

                    func = interpolate.interp1d(out['fpr'], out['tpr'], 'linear')
                    self.tpr_bootstrap[i,:] = func(self.fpr)

                    func = interpolate.interp1d(out['tpr'], out['fpr'], 'linear')
                    self.fpr_bootstrap[i,:] = func(self.tpr)


def compute_metrics(num_classes, y_true, y_pred, weights):

    acc = -1
    auc = -1
    fpr = -1
    tpr = -1
    thresholds = -1

    try:
        if  num_classes == 2:
            fpr, tpr, thresholds = metrics.roc_curve(y_true=y_true, y_score=y_pred, sample_weight=weights)
            auc = metrics.roc_auc_score(y_true=y_true,  y_score=y_pred, sample_weight=weights)
            acc = metrics.accuracy_score(y_true=y_true, y_pred=np.round(y_pred), sample_weight=weights)
        else:
            fpr, tpr, thresholds = None, None, None
            auc = metrics.roc_auc_score(y_true=y_true,  y_score=y_pred, sample_weight=None, \
                        average="weighted", multi_class='ovo', labels=np.arange(num_classes))
            acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred.argmax(axis=1), sample_weight=weights)
    except:
        print(__name__ + f'.compute_metrics: Unable to compute ROC-metrics (check the input statistics)')
    
    return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': auc, 'acc': acc}
