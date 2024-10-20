# Auxialary functions
# 
# m.mieskolainen@imperial.ac.uk, 2024

import math
import numpy as np
import awkward as ak
import re
import time
import pickle
import os
import glob
import torch
from datetime import datetime
import torch
import random
import yaml
import gc

import numba
from tqdm import tqdm

import sklearn
from sklearn import metrics
import scipy
from scipy import interpolate

# ------------------------------------------
from icenet import print
# ------------------------------------------


def yaml_dump(data: dict, filename: str):
    """
    Dump dictionary to YAML with custom style
    
    Args:
        data: dictionary
        filename: full path
    """
    # Custom representer to force flow style for lists
    def flow_style_list_representer(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    # Custom representer to force block style for dictionaries
    def block_style_dict_representer(dumper, data):
        return dumper.represent_mapping('tag:yaml.org,2002:map', data, flow_style=False)

    class NoSortDumper(yaml.Dumper):
        def represent_dict(self, data):
            return self.represent_mapping('tag:yaml.org,2002:map', data.items(), flow_style=False)

    # Register the custom representers with NoSortDumper
    NoSortDumper.add_representer(list, flow_style_list_representer)
    NoSortDumper.add_representer(dict, block_style_dict_representer)
    
    # Save the YAML string with mixed styles to a file
    with open(filename, 'w') as yaml_file:
        yaml.dump(data, yaml_file, Dumper=NoSortDumper)


def recursive_concatenate(array_list, max_batch_size: int=32, axis: int=0):
    """
    Concatenate a list of arrays in a recursive way
    (to avoid possible problems with one big concatenation e.g. with Awkward)
    
    Args:
        array_list:      a list of Awkward or Numpy arrays
        max_batch_size:  maximum number of list elements per concatenation
        axis:            axis to concatenate over
    
    Returns:
        concatenated array
    """
    
    n = len(array_list)
    
    # Base case
    if n == 1:
        return array_list[0]
    
    # Concatenate directly
    elif n <= max_batch_size:
        if isinstance(array_list[0], ak.Array):
            return ak.concatenate(array_list, axis=axis)
        else:
            return np.concatenate(array_list, axis=axis)
    
    # Split the list into two halves and recursively concatenate each half
    else:
        mid   = (n + 1) // 2  # handle odd length
        left  = recursive_concatenate(array_list[:mid], max_batch_size=max_batch_size, axis=axis)
        right = recursive_concatenate(array_list[mid:], max_batch_size=max_batch_size, axis=axis)
        if isinstance(left, ak.Array) or isinstance(right, ak.Array):
            return ak.concatenate([left, right], axis=axis)
        else:
            return np.concatenate([left, right], axis=axis)


def concatenate_and_clean(array_list: list, axis: int=0):
    """
    Concatenate a list of arrays and clean memory
    
    Args:
        array_list: a list of Awkward or numpy arrays
    Returns:
        concatenated array
    """
    if isinstance(array_list[0], ak.Array):
        result = ak.concatenate(array_list, axis=axis)
    else:
        result = np.concatenate(array_list, axis=axis)
    
    del array_list
    gc.collect()
    
    return result

def set_random_seed(seed):
    """
    Set random seeds
    """
    print(f'{seed} (random, numpy, torch)')
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_datetime():
    """
    Return datetime string of style '2024-06-04--14-45-07'
    """
    return str(datetime.now()).replace(' ', '_').replace(':','-').split('.')[0]

def q_exp(x, q: float=1.0):
    """
    q-exponent
    """    
    exp_ = torch.exp if type(x) is torch.Tensor else np.exp
    return exp_(x) if np.abs(q - 1.0) < 1E-4 else (1 + (1-q)*x)**(1.0 / (1-q))

def q_log(x, q: float=1.0):
    """
    q-logarithm
    """    
    log_ = torch.log if type(x) is torch.Tensor else np.log    
    return log_(x) if np.abs(q - 1.0) < 1E-4 else (x**(1-q) - 1) / (1-q)

def inverse_sigmoid(p: np.ndarray, EPS=1E-9):
    """
    Stable inverse sigmoid function
    """
    # log(p) - log(1-p)
    return np.log(np.clip(p, EPS, 1.0-EPS)) - np.log(np.clip(1-p, EPS, 1.0-EPS))

def _positive_sigmoid(x: np.ndarray):
    return 1 / (1 + np.exp(-x))

def _negative_sigmoid(x: np.ndarray):
    exp = np.exp(x) # Cache exp for speed
    return exp / (exp + 1)

def sigmoid(x: np.ndarray):
    """
    Stable sigmoid function
    """
    positive = (x >= 0)
    negative = ~positive
    
    result = np.empty_like(x)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])

    return result

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation
    """
    average  = np.average(values, weights=weights)
    variance = np.average((values - average)**2, weights=weights)
    
    return average, np.sqrt(variance)

def replace_param(default, raytune):
    """
    Parameter replacement
    """
    new_param = {}
    for key in default.keys():
        new_param[key] = raytune[key] if key in raytune.keys() else default[key]
    
    return new_param

def unmask(x, mask, default_value=-1):
    """
    Unmasking function
    """
    out = default_value * np.ones(len(mask))
    out[mask] = x
    return out

def cartesian_product(*arrays):
    """
    N-dimensional generalized cartesian product between arrays
    
    Args:
        *arrays: a list of numpy arrays
    
    Example:
        cartesian_product(*[np.array([1,2,3]), np.array([100,200,500])])
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    
    return arr.reshape(-1, la)


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


def red(X, ids, param, mode=None, exclude_tag='exclude_MVA_vars', include_tag='include_MVA_vars', verbose=True):
    """
    Reduce the input set variables of X (start with all include, then evaluate exclude, then evaluate include)
    
    Remember that using python sets() is not necessarily stable over runs ! (do not rely on the order of sets)
    
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
        
        reduced = list(set(np.array(ids).tolist()) - set(np.array(ids)[mask].tolist()))
        reduced.sort() # Make it deterministic order
        
        if verbose:
            print(f'Included input variables: {np.array(ids)[mask]}', 'yellow')
            print(f'Excluded input variables: {reduced}', 'red')
    else:
        print(f'Using a full set of input variables', 'red')
    
    if   mode == 'X':
        return X[:, mask]
    elif mode == 'ids':
        return np.array(ids)[mask].tolist()
    else:
        return X[:, mask], np.array(ids)[mask].tolist()


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

def pick_index(all_ids: list, vars: list):
    """
    Return indices in all_ids corresponding to vars
    
    (vars can contain regexp)
    
    Args:
        all_ids: list of strings, e.g. ['a','b','c']
        vars:    list of string to pick, e.g. ['a', 'c'] or ['.*']
    
    Returns:
        index array, variable names list
    """
    
    var_names = process_regexp_ids(all_ids=all_ids, ids=vars)
    pick_ind  = np.array(np.where(np.isin(all_ids, var_names))[0], dtype=int)

    return pick_ind, var_names


#def parse_vars(items):
#    """
#    Parse a series of key-value pairs and return a dictionary
#    """
#    d = {}
#
#    if items:
#        for item in items:
#            key, value = parse_var(item)
#            d[key] = value
#    return d


def ak2numpy(x: ak.Array, fields: list, null_value: float=-999.0, dtype='float32'):
    """
    Unzip awkward array to numpy array per column (awkward Record)
    
    Args:
        x:            awkward array
        fields:       record field names to extract
        null_value:   missing element value
        dtype:        final numpy array dtype
    
    Returns:
        numpy array with columns ordered as 'fields' parameter
    """
    out = np.full((len(x), len(fields)), null_value, dtype=dtype)
    
    for i in range(len(fields)):
        
        y = ak.to_numpy(x[fields[i]], allow_missing=True)
        
        if np.ma.isMaskedArray(y): # Process missing elements
            y = np.ma.filled(y, fill_value=null_value)
        
        out[:,i] = y
    
    return out


def jagged_ak_to_numpy(arr, scalar_vars, jagged_vars, jagged_maxdim,
                       entry_start=None, entry_stop=None, null_value: float=-999.0, dtype='float32'):
    """
    Transform jagged awkward array to fixed dimensional numpy data
    
    Args:
        arr:           jagged awkward array
        scalar_vars:   Scalar variable names
        jagged_vars:   Jagged variable names
        jagged_maxdim: Maximum dimension per jagged category
        null_value:    Fill null value
    Returns:
        numpy array, ids
    """
    
    # Create tuplet expanded jagged variable names
    jagged_dim      = []
    all_jagged_vars = []
        
    for i in range(len(jagged_vars)):
        
        # Split by the first "_" occuring
        dim = int(jagged_maxdim[jagged_vars[i].split('_', 1)[0]])
        jagged_dim.append(dim)
        
        # Create names of type 'varname_j'
        # (xgboost does not accept [,], or <,> and (,) can be problematic otherwise)
        for j in range(dim):
            all_jagged_vars.append(f'{jagged_vars[i]}_{j}')
    
    # Parameters
    arg = {
        'entry_start': entry_start,
        'entry_stop':  entry_stop,
        'scalar_vars': scalar_vars,
        'jagged_vars': jagged_vars,
        'jagged_dim':  jagged_dim,
        'null_value':  null_value,
        'dtype':       dtype
    }
    
    ids = scalar_vars + all_jagged_vars # First scalar, then jagged !

    return jagged2matrix(arr=arr, **arg), ids


def jagged2matrix(arr, scalar_vars, jagged_vars, jagged_dim,
    entry_start=None, entry_stop=None, null_value: float=-999.0, mode: str='columnar', dtype='float32'):
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
    
    # Print stats
    mem_size = N*D*32/8/1024**3 # 32 bit
    if dtype == 'float64': mem_size *= 2
    
    print(f'Creating a matrix with dimensions [{N} x {D}] ({mem_size:0.3f} GB)')

    # Pre-processing of jagged variable names
    jvname = []
    for j in range(len(jagged_vars)):
        # Awkward groups jagged variables, e.g. 'sv_x' to sv.x
        jvname.append(jagged_vars[j].split('_', 1)) # argument 1 takes the first '_' occurance

    # Return matrix
    shared_array = np.full((N,D), null_value, dtype=dtype)

    ## Pure scalar vars (very fast)
    for j in range(len(scalar_vars)):
        shared_array[:,j] = ak.to_numpy(ak.ravel(arr[scalar_vars[j]])[entry_start:entry_stop])
    
    t = time.time()

    ## Jagged vars dimension by dimension (very fast), with the maximum
    # number of objects dictated by by jagged_dim[]
    if mode == 'columnar':

        k  = len(scalar_vars)

        for j in tqdm(range(len(jagged_vars))):
            for dim in range(1, jagged_dim[j] + 1):

                # E.g. jet.pt ~ a = jet, b = pt
                # Boolean mask for events, at least dim number of objects (e.g. jets)
                a,b  = jvname[j][0], jvname[j][1]
                mask = (ak.num(arr[entry_start:entry_stop][a][b]) >= dim)

                # Push to numpy array
                shared_array[mask, k+(dim-1)] = \
                    arr[entry_start:entry_stop][mask][a][b][:, dim-1]

            # Increase jagged block counter
            k += jagged_dim[j]

    else:

        # Jagged variables by explicit event loop (slow)
        for index in tqdm(zip(range(N), range(entry_start, entry_stop))):

            k  = len(scalar_vars)
            i  = index[0]
            ev = index[1]
            
            ## Variable by variable
            for j in range(len(jagged_vars)):

                x  = ak.to_numpy(arr[ev][jvname[j][0]][jvname[j][1]])
                
                if len(x) > 0:
                    d_this = np.min([len(x), jagged_dim[j]])
                    shared_array[i, k:k+d_this] = x[0:d_this]
                
                # Increase jagged block counter
                k += jagged_dim[j]

    elapsed = time.time() - t
    print(f'Processing took {elapsed:0.1f} sec')

    return shared_array


def jagged2tensor(X, ids, xyz, x_binedges, y_binedges, dtype='float32'):
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
    T = np.zeros((X.shape[0], len(xyz), len(x_binedges)-1, len(y_binedges)-1), dtype=dtype)

    # Choose targets
    for c in range(len(xyz)):
        ind = [ids.index(x) for x in xyz[c]]

        # Loop over all events
        for i in tqdm(range(X.shape[0])):
            T[i,c,:,:] = arrays2matrix(x_arr=X[i,ind[0]], y_arr=X[i,ind[1]], z_arr=X[i,ind[2]], 
                x_binedges=x_binedges, y_binedges=y_binedges)

    print(f'Returning tensor with shape {T.shape}')

    return T


def arrays2matrix(x_arr, y_arr, z_arr, x_binedges, y_binedges, dtype='float32'):
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
    A  = np.zeros((len(x_binedges)-1, len(y_binedges)-1), dtype=dtype)
    try:
        for i in range(len(x_ind)):
            A[x_ind[i], y_ind[i]] += z_arr[i]
    except:
        print(f'Not valid input (returning 0-matrix)')

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
    if len(a) < n: # Overflow protection
        n = len(a)
    
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def split_size(a, n):
    """
    As split_start_end() but returns only size per chunk
    """
    ll  = list(split(a,n))
    out = [len(ll[i]) for i in range(len(ll))]

    return out

def explicit_range(entry_start, entry_stop, num_entries):
    """
    Clean None from entry_start and entry_stop
    """
    start = 0 if entry_start is None else entry_start
    stop  = num_entries if entry_stop is None else entry_stop

    return start, stop

def split_start_end(a, n, end_plus=1):
    """
    Returns approx equally sized chunks.

    Args:
        a:        Range, define with range()
        n:        Number of chunks
        end_plus: Python/nympy index style (i.e. + 1 for the end)

    Examples:
        split_start_end(range(100), 3)  returns [[0, 34], [34, 67], [67, 100]]
        split_start_end(range(5,25), 3) returns [[5, 12], [12, 19], [19, 25]]
    """
    ll  = list(split(a,n))
    out = []

    for i in range(len(ll)):
        out.append([ll[i][0], ll[i][-1] + end_plus])

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
    
    print(f'vec.shape = {vec.shape}')

    intmat = binaryvec2int(vec)
    BMAT   = generatebinary(K)
    print(f'{ids}')
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
        print(f'Binary matrix dimension {K} x {N}')

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
        Y = np.zeros(X.shape[0], dtype=np.int64)
    
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


def getmtime(filename):
    """ Return the last modification time of a file, reported by os.stat()
    """
    return os.stat(filename).st_mtime


def create_model_filename(path: str, label: str, filetype='.dat', epoch:int=None):
    """
    Create model filename
    
    This function automatically takes the minimum validation loss epoch / iteration
    
    if epoch == - 1, we try to find the best loss model
       epoch == - 2, we take the latest epoch
    
    """
    def createfilename(i):
        return f'{path}/{label}_{i}{filetype}'
    
    if epoch is None or epoch < 0:
        
        print(f'Loading the latest model by timestamp', 'yellow')

        list_of_files = glob.glob(f'{path}/{label}_*{filetype}')
        
        if len(list_of_files) == 0:
            txt  = f'Could not find any files for model "{label}"'
            txt += f" under path {path}"
            raise Exception(txt)
        
        # The latest model
        filename = max(list_of_files, key=os.path.getctime)
        
        # ----------------------------------------------------
        # Try to find the best model
        
        if epoch == -1:
            
            succeeded = False
                
            try:
                # Try with pickle load
                with open(filename, 'rb') as file:
                    data  = pickle.load(file)
                succeeded = True
                
            except:
                # Try with torch load
                try:
                    data = torch.load(filename, map_location = 'cpu')
                    succeeded = True
                
                except Exception as e:
                    print(e)
                    print(f'Problem in finding the model [{label}] with the minimum validation loss', 'red')
            
            if succeeded:
                # Take the minimum validation loss epoch index
                losses = np.array(data['losses']['val_losses'])
                idx    = np.argmin(losses)
                
                str = f'Found the best model at epoch [{idx}] with validation loss = {losses[idx]:0.4f}'
                print(f'{str}', 'magenta')
                
                filename = createfilename(idx)
        # ----------------------------------------------------
        
    else:
        print(f'Loading the model with the provided epoch = {epoch}', 'yellow')
        filename = createfilename(epoch)
    
    dt = datetime.fromtimestamp(getmtime(filename))
    print(f'Found a model file: {filename} (modified {dt})', 'green')
    
    return filename


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
    def __init__(self, y_true, y_pred, weights=None, class_ids=[0,1], hist=True, valrange='prob',
        N_mva_bins=30, verbose=True, num_bootstrap=0, exclude_neg_class=True):
        """
        Args:
            y_true     : true classifications
            y_pred     : predicted probabilities per class (N x 1), (N x 2) or (N x K) dimensional array
            weights    : event weights
            class_ids  : class indices
            
            hist       : histogram soft decision values
            valrange   : histogram range selection type
            N_mva_bins : number of bins
            num_bootstrap     : number of bootstrap trials
            exclude_neg_class : exclude (special) negative class from y_true
        
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
        
        # Make sure they are integer (scikit ROC functions cannot handle continuous values)
        y_true = np.round(y_true)
        
        # ----------------------------------------------
        # Special classes excluded
        if exclude_neg_class:
            include = (y_true >= 0)
            y_true  = y_true[include]
            y_pred  = y_pred[include]
            if weights is not None:
                weights = weights[include]
        # ----------------------------------------------
        
        if class_ids is None:
            self.class_ids = np.unique(y_true.astype(int))
        else:
            self.class_ids = class_ids
        
        self.num_classes = len(self.class_ids)

        # Invalid input
        if self.num_classes <= 1:
            if verbose:
                print(f'only one class present cannot evaluate metrics (return -1)')

            return # Return None
        
        # Transform N x 2 to N x 1 (pick class[1] probabilities as the signal)
        if (self.num_classes == 2) and (np.squeeze(y_pred).ndim == 2):
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
            
            for c in self.class_ids:
                ind    = (y_true == c)
                counts = []
                
                if np.sum(ind) != 0:
                    w = weights[ind] if weights is not None else None
                    x = y_pred[ind] if self.num_classes == 2 else y_pred[ind,c]
                    counts, edges = np.histogram(x, weights=w, bins=self.mva_bins)
                self.mva_hist.append(counts)
        else:
            self.mva_bins = None
            self.mva_hist = None

        # ------------------------------------
        ## Compute Metrics
        out = compute_metrics(class_ids=self.class_ids, y_true=y_true, y_pred=y_pred, weights=weights)

        self.acc        = out['acc']
        self.auc        = out['auc']
        self.fpr        = out['fpr']
        self.tpr        = out['tpr']
        self.thresholds = out['thresholds']
        
        # Compute bootstrap
        if (num_bootstrap is not None and num_bootstrap > 0) and type(self.tpr) is not int:
            
            self.tpr_bootstrap = (-1)*np.ones((num_bootstrap, len(self.tpr)))
            self.fpr_bootstrap = (-1)*np.ones((num_bootstrap, len(self.fpr)))
            self.auc_bootstrap = (-1)*np.ones(num_bootstrap)
            self.acc_bootstrap = (-1)*np.ones(num_bootstrap)

            for i in range(num_bootstrap):
                
                # ------------------
                trials     = 0
                max_trials = 10000
                while True:
                    ind = np.random.choice(range(len(y_true)), size=len(y_true), replace=True)
                    if len(np.unique(y_true[ind])) > 1 or trials > max_trials: # Protection with very low per class stats
                        break
                    else:
                        trials += 1
                if trials > max_trials:
                    print(f'bootstrap failed (check the input per class statistics)')
                    continue
                # ------------------
                
                ww  = weights[ind] if weights is not None else None
                out = compute_metrics(class_ids=self.class_ids, y_true=y_true[ind], y_pred=y_pred[ind], weights=ww)
                
                if out['auc'] > 0:
                    
                    self.auc_bootstrap[i] = out['auc']
                    self.acc_bootstrap[i] = out['acc']

                    # Interpolate ROC-curve (re-sample to match the non-bootstrapped x-axis)
                    func = interpolate.interp1d(out['fpr'], out['tpr'], 'linear')
                    self.tpr_bootstrap[i,:] = func(self.fpr)
                    
                    func = interpolate.interp1d(out['tpr'], out['fpr'], 'linear')
                    self.fpr_bootstrap[i,:] = func(self.tpr)


def sort_fpr_tpr(fpr, tpr):
    """
    For numerical stability with negative weighted events
    """
    fpr = np.clip(fpr, 0.0, 1.0)
    tpr = np.clip(tpr, 0.0, 1.0)
    
    sorted_index = np.argsort(fpr) # x-axis needs to be monotonic
    fpr_sorted   = np.array(fpr)[sorted_index]
    tpr_sorted   = np.array(tpr)[sorted_index]
    
    return fpr_sorted, tpr_sorted


def auc_score(fpr, tpr):
    """
    AUC-ROC via numerical intergration
    
    Args:
        fpr:  false positive rate array
        tpr:  true positive rate array
    
    Call sort_fpr_tpr before this function for numerical stability.
    
    Returns:
        AUC score
    """
    auc = scipy.integrate.trapz(y=tpr, x=fpr)

    return np.clip(auc, 0.0, 1.0)


def compute_metrics(class_ids, y_true, y_pred, weights):

    acc = -1
    auc = -1
    fpr = -1
    tpr = -1
    thresholds = -1

    # Fix NaN
    num_nan = np.sum(~np.isfinite(y_pred))
    if num_nan > 0:
        print(f'Found {num_nan} NaN/Inf (set to zero)')
        y_pred[~np.isfinite(y_pred)] = 0 # Set to zero
    
    try:
        if  len(class_ids) == 2:
            fpr, tpr, thresholds = metrics.roc_curve(y_true=y_true, y_score=y_pred, sample_weight=weights)
            
            if np.min(weights < 0): # Protect
                fpr, tpr = sort_fpr_tpr(fpr=fpr, tpr=tpr)
            
            # By integration
            auc = auc_score(fpr=fpr, tpr=tpr)
            
            #auc = metrics.roc_auc_score(y_true=y_true,  y_score=y_pred, sample_weight=weights)
            acc = metrics.accuracy_score(y_true=y_true, y_pred=np.round(y_pred), sample_weight=weights)
        else:
            fpr, tpr, thresholds = None, None, None
            auc = metrics.roc_auc_score(y_true=y_true,  y_score=y_pred, sample_weight=None, \
                        average="weighted", multi_class='ovo', labels=class_ids)
            acc = metrics.accuracy_score(y_true=y_true, y_pred=y_pred.argmax(axis=1), sample_weight=weights)
    
    except Exception as e:
        print(f'Unable to compute ROC-metrics: {e}')
        for c in class_ids:
            print(f'num of class[{c}] = {np.sum(y_true == c)}')

    return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': auc, 'acc': acc}
