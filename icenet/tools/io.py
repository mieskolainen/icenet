# Input data containers and memory management
#
# m.mieskolainen@imperial.ac.uk, 2025

import numpy as np
import awkward as ak

import numba
import copy
import torch
import os
import psutil
import subprocess
import re
import pathlib
from natsort import natsorted

# MVA imputation
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

from sklearn.experimental import enable_iterative_imputer # Needs this
from sklearn.impute import IterativeImputer

# Command line arguments
from glob import glob
from braceexpand import braceexpand
import copy
import time

import hashlib
import base64

from icenet.tools import aux
from icenet.tools import stx

# ------------------------------------------
from icenet import print
# ------------------------------------------

def get_file_timestamp(file_path: str):
    """
    Return file timestamp as a string
    """
    if os.path.exists(file_path):
        # Get the file last modification time
        timestamp = os.path.getmtime(file_path)
        # Convert it to a readable format
        readable_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        return readable_time
    else:
        return f"File '{file_path}' does not exist."

def rootsafe(txt):
    """
    Change character due to ROOT
    """
    return txt.replace('-', '_').replace('+','_').replace('/','_').replace('*','_')

def safetxt(txt):
    """
    Protection for '/'
    """
    if type(txt) is str:
        return txt.replace('/', '|')
    else:
        return txt

def count_files_in_dir(path):
    """
    Count the number of files in a path
    """
    return len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])

def make_hash_sha256_file(filename):
    """
    Create SHA256 hash from a file
    """
    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()

def make_hash_sha256_object(o):
    """
    Create SHA256 hash from an object

    Args:
        o: python object (e.g. dictionary)
    
    Returns:
        hash
    """
    hasher = hashlib.sha256()
    hasher.update(repr(make_hashable(o)).encode())
    hash_str = base64.b64encode(hasher.digest()).decode()

    # May cause problems with directories
    hash_str = hash_str.replace('/',  '_')
    hash_str = hash_str.replace('\\', '__')
    hash_str = hash_str.replace('.',  '___')
    
    return hash_str


def make_hashable(o):
    """
    Turn a python object into hashable type (recursively)
    """
    if isinstance(o, (tuple, list)):
        return tuple((make_hashable(e) for e in o))
    
    if isinstance(o, dict):
        return tuple(sorted((k,make_hashable(v)) for k,v in o.items()))
    
    if isinstance(o, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in o))
    
    return o


def glob_expand_files(datasets, datapath, recursive_glob=False):
    """
    Do global / brace expansion of files

    Args:
        datasets: dataset filename with glob syntax (can be a list of files)
        datapath: root path to files
    
    Returns:
        files: full filenames including the path
    """
    print("")
    print(f"Supported syntax: <filename_*>, <filename_0>, <filename_[0-99]>, <filename_{{0,3,4}}>")
    print("See https://docs.python.org/3/library/glob.html and brace expansion (be careful, do not use [,] brackets in your filenames)")
    print("")
    
    # Remove unnecessary []
    if type(datasets) is list and len(datasets) == 1:
        datasets = datasets[0]
    
    # Try first to brace expand
    try:
        datasets = list(braceexpand(datasets))
    except:
        True
    
    #print(__name__ + f'.glob_expand_files: After braceexpand: {datasets}')

    if (len(datasets) == 1) and ('[' in datasets[0]) and (']' in datasets[0]):

        print(f'Parsing of range [first-last] ...')

        res   = re.findall(r'\[.*?\]', datasets[0])[0]
        temp  = res[1:-1]

        numbers = temp.split('-')        
        first   = int(numbers[0])
        last    = int(numbers[1])

        print(f'Obtained range of files: [{first}, {last}]')

        # Split and add
        parts = datasets[0].split(res)
        datasets[0] = parts[0] + '{'
        for i in range(first, last+1):
            datasets[0] += f'{i}'
            if i != last:
                datasets[0] += ','
        datasets[0] += '}' + parts[1]

        datasets = list(braceexpand(datasets[0]))
        #print(__name__ + f'.glob_expand_files: After expanding the range: {datasets}')

    # Parse input files into a list
    files = []
    for data in datasets:
        
        # This does e.g. _*.root expansion (finds the files)
        x = str(pathlib.Path(datapath) / data)
        expanded_files = glob(x, recursive=recursive_glob)
        
        files.extend(expanded_files)
    
    if files == []:
       files = [datapath]
    
    # Normalize e.g. for accidental multiple slashes
    files = [os.path.normpath(f) for f in files]
    
    # Make them unique and natural sorted
    files = natsorted(set(files))
    
    #print(__name__ + f'.glob_expand_files: Final files: {files}')
    
    return files


def showmem(color='red'):
    print(f"""Process RAM: {process_memory_use():0.2f} GB [total RAM in use {psutil.virtual_memory()[2]} %]""", color)

def showmem_cuda(device='cuda:0', color='red'):
    print(f"Process RAM: {process_memory_use():0.2f} GB [total RAM in use {psutil.virtual_memory()[2]} %] | VRAM usage: {get_gpu_memory_map()} GB [total VRAM {torch_cuda_total_memory(device):0.2f} GB]", color)


def get_gpu_memory_map():
    """Get the GPU VRAM use in GB.
    
    Returns:
        dictionary with keys as device ids [integers]
        and values the memory used by the GPU.
    """
    try:
        
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ], encoding='utf-8')

        # into dictionary
        gpu_memory = [int(x)/1024.0 for x in result.strip().split('\n')]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        return gpu_memory_map
    
    except Exception as e:
        print(f'Error: Could not run nvidia-smi: {e}')
        return None

def torch_cuda_total_memory(device='cuda:0'):
    """
    Return CUDA device VRAM available in GB.
    """
    return torch.cuda.get_device_properties(device).total_memory / 1024.0**3


def process_memory_use():
    """
    Return system memory (RAM) used by the process in GB.
    """
    pid = os.getpid()
    py  = psutil.Process(pid)
    return py.memory_info()[0]/2.**30


def checkinfnan(x, value = 0):
    """ Check inf and Nan values and replace with a default value.
    """
    inf_ind = np.isinf(x)
    nan_ind = np.isnan(x)
    
    x[inf_ind] = value
    x[nan_ind] = value

    if np.sum(inf_ind) > 0:
        print(f'Inf found, replacing with {value}', 'red')
    if np.sum(nan_ind) > 0:
        print(f'NaN found, replacing with {value}', 'red')    
    return x


class fastarray1:
    """ 1D pre-memory occupied buffer arrays for histogramming etc.
    """
    def __init__(self, capacity = 32):
        self.capacity = capacity
        self.data = np.zeros((self.capacity))
        self.size = 0

    # use with x.update([1,3,5,1])
    #
    def update(self, row):
        for x in row:
            self.add(x)
    
    # use with x.add(32)
    #
    def add(self, x):
        if self.size == self.capacity:
            print(f'increasing current capacity = {self.capacity} to 2x')
            self.capacity *= 2
            newdata = np.zeros((self.capacity))
            newdata[:self.size] = self.data
            self.data = newdata

        self.data[self.size] = x
        self.size += 1

    # Get values
    #
    def values(self):
        return self.data[0:self.size]

    # Reset index, keep the buffersize
    def reset(self):
        self.size = 0

def index_list(target_list, keys):
    """
    Use e.g. x_subset = x[:, io.index_list(ids, variables)]
    
    """
    index = []
    for key in keys:
        index.append(target_list.index(key))
    return index
    

def pick_vars(data, set_of_vars):
    """ Choose the active set of input variables.

    Args:
        data:        IceXYW type object
        set_of_vars: Variables to pick
    Returns:
        newind:      Chosen indices
        newvars:     Chosen variables
    """
    
    newind  = np.where(np.isin(data.ids, set_of_vars))
    newind  = np.array(newind).flatten()
    newvars = []
    for i in newind :
        newvars.append(data.ids[i])

    return newind, newvars

class IceXYW:
    """
    Args:
        x : data object
        y : target output data
        w : weights
    """
    
    # constructor
    def __init__(self, x = np.array([]), y = np.array([]), w = None, ids = None):
        
        self.x = x
        self.y = y
        self.w = w
        self.ids = ids

        if   isinstance(x, np.ndarray):
            self.concat = np.concatenate
        elif isinstance(x, ak.Array):
            self.concat = ak.concatenate
        else:
            raise Exception(__name__ + f'.IceXYW.__init__: Unknown input array type')
    
    def find_ind(self, key):
        """
        Return column index corresponding to key
        """
        return int(np.where(np.array(self.ids, dtype=np.object_) == key)[0])
    
    def __getitem__(self, key):
        """ Advanced indexing with a variable or a list of variables """
        
        if type(key) is not list:       # access with a single variable name
            
            try:
                select = key
                new = IceXYW(x=self.x[select, ...], y=self.y[select], w=self.w[select], ids=self.ids)
                return new
            except:
                True
            
            if key in self.ids:         # direct access
                col = self.ids.index(key)
                ids = [key]
            
            elif isinstance(self.x, np.ndarray): # might be a cut string, try that
                
                select = stx.eval_boolean_syntax(expr=key, X=self.x, ids=self.ids, verbose=True)
                return IceXYW(x=self.x[select, ...], y=self.y[select], w=self.w[select], ids=self.ids)
            
            else:
                raise Exception(__name__ + f'[operator]: Cannot execute')
        
        else:                          # list of variables
            col,ids = pick_vars(data=self, set_of_vars=key)
        
        if isinstance(self.x, np.ndarray):
            return IceXYW(x=self.x[..., col], y=self.y, w=self.w, ids=ids)
        else:
            return IceXYW(x=self.x[col], y=self.y, w=self.w, ids=ids)
    
    # length operator
    def __len__(self):
        return len(self.x)
    
    # + operator
    def __add__(self, other):

        if (len(self.x) == 0): # still empty
            return other

        x = self.concat((self.x, other.x), axis=0)
        y = self.concat((self.y, other.y), axis=0)
        if self.w is not None:
            w = self.concat((self.w, other.w), axis=0)

        return IceXYW(x, y, w)

    # += operator
    def __iadd__(self, other):

        if (len(self.x) == 0): # still empty
            return other

        self.x = self.concat((self.x, other.x), axis=0)
        self.y = self.concat((self.y, other.y), axis=0)
        if self.w is not None:
            self.w = self.concat((self.w, other.w), axis=0)

        return self

    # filter operator
    def classfilter(self, classid):

        ind = (self.y == classid)

        x = self.x[ind]
        y = self.y[ind]

        if self.w is not None:
            w = self.w[ind]
        else:
            w = self.w
        
        return IceXYW(x=x, y=y, w=w, ids=self.ids)
        
    # Permute events
    def permute(self, permutation):
        
        self.x = self.x[permutation]
        self.y = self.y[permutation]

        if self.w is not None:
            self.w = self.w[permutation]

        return self


def split_data_simple(X, frac, permute=True):
    """ Split machine learning data into train, validation, test sets
    
    Args:
        X:         data as a list of event objects (such as torch geometric Data)
        frac:      split fraction
    """

    ### Permute events to have random mixing between events
    if permute:
        randi = np.random.permutation(len(X)).tolist()
        X = [X[i] for i in randi]
    
    N     = len(X)
    N_A   = round(N * frac)
    N_B   = N - N_A
    
    N_trn = N_A
    N_val = round(N_B / 2)
    N_tst = N - N_trn - N_val

    # Split
    X_trn = X[0:N_trn]
    X_val = X[N_trn:N_trn + N_val]
    X_tst = X[N - N_tst:]
    
    print(f"fractions [train: {len(X_trn)/N:0.3f}, validate: {len(X_val)/N:0.3f}, test: {len(X_tst)/N:0.3f}]")
    
    return X_trn, X_val, X_tst


def split_data(X, Y, W, ids, frac=[0.5, 0.1, 0.4], permute=True):
    """ Split machine learning data into train, validation, test sets
    
    Args:
        X:         data matrix
        Y:         target matrix
        W:         weight matrix
        ids:       variable names of columns
        frac:      fraction [train, validate, evaluate] (sum to 1)
        rngseed:   random seed
    """

    ### Permute events to have random mixing between classes
    if permute:
        randi = np.random.permutation(len(X))
        X = X[randi]
        Y = Y[randi]
        if W is not None:
            W = W[randi]
    
    # --------------------------------------------------------------------

    frac  = np.array(frac)
    frac  = frac / np.sum(frac)
    
    # Get event counts
    N     = len(X)
    N_trn = int(round(N * frac[0]))
    N_tst = int(round(N * frac[2]))
    N_val = N - N_trn - N_tst
    
    # 1. Train
    X_trn = X[0:N_trn]
    Y_trn = Y[0:N_trn]
    if W is not None:
        W_trn = W[0:N_trn]
    else:
        W_trn = None

    # 2. Validation
    X_val = X[N_trn:N_trn + N_val]
    Y_val = Y[N_trn:N_trn + N_val]
    if W is not None:
        W_val = W[N_trn:N_trn + N_val]
    else:
        W_val = None

    # 3. Test
    X_tst = X[N - N_tst:]
    Y_tst = Y[N - N_tst:]
    if W is not None:
        W_tst = W[N - N_tst:]
    else:
        W_tst = None

    # --------------------------------------------------------
    # Construct
    trn = IceXYW(x = X_trn, y = Y_trn, w=W_trn, ids=ids)
    val = IceXYW(x = X_val, y = Y_val, w=W_val, ids=ids)
    tst = IceXYW(x = X_tst, y = Y_tst, w=W_tst, ids=ids)
    # --------------------------------------------------------
    
    print(f"fractions [train: {len(X_trn)/N:0.3f}, validate: {len(X_val)/N:0.3f}, test: {len(X_tst)/N:0.3f}]")
    
    return trn, val, tst


def impute_data(X, imputer=None, dim=None, values=[-999], labels=None, algorithm='iterative', fill_value=0, knn_k=6):
    """ Data imputation (treatment of missing values, Nan and Inf).
    
    NOTE: This function can impute only fixed dimensional input currently (not Jagged numpy arrays)
    
    Args:
        X         : Input data matrix [N vectors x D dimensions]
        imputer   : Pre-trained imputator, default None
        dim       : Array of active dimensions to impute
        values    : List of special integer values indicating the need for imputation
        labels    : List containing textual description of input variables
        algorithm : 'constant', mean', 'median', 'iterative', knn_k'
        knn_k     : knn k-nearest neighbour parameter
        
    Returns:
        X         : Imputed output data
    """
    
    if dim is None:
        dim = np.arange(X.shape[1])
    
    if labels is None:
        labels = np.zeros(X.shape[1])

    N = X.shape[0]
    
    # Count NaN
    for j in dim:
        nan_ind = np.isnan(np.array(X[:,j], dtype=np.float32))
        found   = np.sum(nan_ind)
        if found > 0:
            print(f'Column {j} Number of {nan_ind} NaN found ({found/len(X):0.3E}) [{labels[j]}]', 'red')
    
    # Loop over dimensions
    for j in dim:

        # Set NaN for special values
        M_tot = 0
        for z in values:
            
            ind = np.isclose(np.array(X[:,j], dtype=np.float32), z)
            X[ind, j] = np.nan
            
            M = np.sum(ind)
            M_tot += M

            if (M/N > 0):
                print(f'Column {j} fraction [{M/N:0.3E}] with value {z} [{labels[j]}]')

        if (M_tot == N): # Protection, if all are now NaN
            # Set to zero so Imputer Function below does not remove the full column!!
            X[:,j] = 0.0

    # Treat infinities (inf)
    for j in dim:

        inf_ind = np.isinf(np.array(X[:,j], dtype=np.float32))
        X[inf_ind, j] = np.nan
        
        found = np.sum(inf_ind)
        if found > 0:
            print(f'Column {j} Number of {found} Inf found ({found/len(X):0.3E}) [{labels[j]}]', 'red')
    
    if imputer == None:

        # Fill missing values
        if   algorithm == 'constant':
            imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=fill_value)
        elif algorithm == 'mean':
            imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        elif algorithm == 'median':
            imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        elif algorithm == 'iterative':
            imputer = IterativeImputer(missing_values=np.nan)
        elif algorithm == 'knn':
            imputer = KNNImputer(missing_values=np.nan, n_neighbors=knn_k)
        else:
            raise Exception(__name__ + '.impute_data: Unknown algorithm chosen')
    
    # Fit and transform
    imputer.fit(X[:,dim])
    X[:,dim] = imputer.transform(X[:,dim])
    
    print('[done] \n')
    
    return X, imputer


def calc_madscore(X : np.array):
    """ Calculate robust normalization.

    Args:
        X : Input with [# vectors x # dimensions]

    Returns:
        X_m   : Median vector
        X_mad : Median deviation vector 
    """

    X_m   = np.zeros((X.shape[1]))
    X_mad = np.zeros((X.shape[1]))

    # Calculate mean and std based on the training data
    for i in range(X.shape[1]) :
        X_m[i]   = np.median(X[:,i])
        X_mad[i] = np.median(np.abs(X[:,i] - X_m[i]))

        if (np.isnan(X_mad[i])):
            raise Exception(__name__ + f': Fatal error with MAD[index = {i}] is NaN')
        if (np.isinf(X_mad[i])):
            raise Exception(__name__ + f': Fatal error with MAD[index = {i}] is Inf')

    return X_m, X_mad

def calc_zscore_tensor(T):
    """
    Compute z-score normalization for tensors.
    Args:
        T : input tensor data (events x channels x rows x cols, ...)
    Returns:
        mu, std tensors
    """
    Y   = copy.deepcopy(T)
    Y[~np.isfinite(Y)] = 0
    mu  = np.mean(Y, axis=0)
    std = np.std(Y, axis=0)

    return mu, std

def apply_zscore_tensor(T, mu, std, EPS=1E-12):
    """
    Apply z-score normalization for tensors.
    """
    Y = copy.deepcopy(T)
    Y[~np.isfinite(Y)] = 0

    # Over all events
    for i in range(T.shape[0]):
        Y[i,...] = (Y[i,...] - mu) / max(std, EPS)
    return Y

def calc_zscore(X: np.array, weights: np.array = None):
    """ Calculate 0-mean & unit-variance normalization.

    Args:
        X       : Input with [N x dim]
        weights : Event weights
    
    Returns:
        X_mu  : Mean vector
        X_std : Standard deviation vector 
    """

    X_mu  = np.zeros((X.shape[1]))
    X_std = np.zeros((X.shape[1]))

    # Calculate mean and std based on the training data
    for i in range(X.shape[1]):
        
        xval = X[:,i]
        
        if weights is not None:
            X_mu[i], X_std[i] = aux.weighted_avg_and_std(xval, weights)
        else:
            X_mu[i], X_std[i] = np.mean(xval), np.std(xval)
        
        if (np.isnan(X_std[i])):
            raise Exception(__name__ + f': Fatal error with std[index = {i}] is NaN')

        if (np.isinf(X_std[i])):
            raise Exception(__name__ + f': Fatal error with std[index = {i}] is Inf')

    return X_mu, X_std


@numba.njit(parallel=True)
def apply_zscore(X : np.array, X_mu, X_std, EPS=1E-12):
    """ Z-score normalization
    """

    Y = np.zeros(X.shape)
    for i in range(len(X_mu)):
        Y[:,i] = (X[:,i] - X_mu[i]) / max(X_std[i], EPS)
    return Y

@numba.njit(parallel=True)
def reverse_zscore(Y: np.array, X_mu, X_std, EPS=1E-12):
    """ Reverse Z-score normalization
    """
    X = np.zeros(Y.shape)
    for i in range(len(X_mu)):
        X[:, i] = Y[:, i] * max(X_std[i], EPS) + X_mu[i]
    return X

@numba.njit(parallel=True)
def apply_madscore(X : np.array, X_m, X_mad, EPS=1E-12):
    """ MAD-score normalization
    """

    Y = np.zeros(X.shape)
    scale = 0.6745 # 0.75th of z-normal
    for i in range(len(X_m)):
        Y[:,i] = scale * (X[:,i] - X_m[i]) / max(X_mad[i], EPS)
    return Y
