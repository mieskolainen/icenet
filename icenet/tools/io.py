# Input data containers and memory management
# 
# Mikael Mieskolainen, 2021
# m.mieskolainen@imperial.ac.uk


import numpy as np
import numba
import copy
import torch
import os
import psutil
import subprocess

import sys

from termcolor import colored, cprint

from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def showmem(color='red'):
    cprint(__name__ + f""".showmem: Process RAM usage: {process_memory_use():0.2f} GB 
        [total RAM in use {psutil.virtual_memory()[2]} %]""", color)


def get_gpu_memory_map():
    """Get the GPU VRAM use in GB.
    
    Returns:
        dictionary with keys as device ids [integers]
        and values the memory used by the GPU.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')

    # into dictionary
    gpu_memory = [int(x)/1024.0 for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def torch_cuda_total_memory(device):
    """
    Return CUDA device VRAM available in GB.
    """
    return torch.cuda.get_device_properties(device).total_memory / 1024.0**3


def process_memory_use():
    """
    Return system memory (RAM) used by the process in GB.
    """
    pid = os.getpid()
    py = psutil.Process(pid)
    return py.memory_info()[0]/2.**30


def checkinfnan(x, value = 0):
    """ Check inf and Nan values and replace with a default value.
    """
    inf_ind = np.isinf(x)
    nan_ind = np.isnan(x)
    
    x[inf_ind] = value
    x[nan_ind] = value

    if np.sum(inf_ind) > 0:
        print(colored(__name__ + f'.checkinfnan: Inf found, replacing with {value}', 'red'))
    if np.sum(nan_ind) > 0:
        print(colored(__name__ + f'.checkinfnan: NaN found, replacing with {value}', 'red'))    
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
            print(f'fastarray1.add: increasing current capacity = {self.capacity} to 2x')
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


class DATASET:
    """ Main class for datasets
    """
    def __init__(self, func_loader, files, frac, rngseed, class_id = []):

        if (class_id == []):
            class_id = [0,1] # By default two classes [0,1]
        
        if (files == []):
            cprint(__name__ + f'.DATASET.__init__: files = [], relying on func_loader()', 'red')
            files = ['dummy.root'] # Will be defined in func_loader
        else:
            cprint(__name__ + f'.DATASET.__init__: files = {files}', 'green')

        self.trn = Data()
        self.val = Data()
        self.tst = Data()
        
        for f in files :
            X, Y, self.ids = func_loader(root_path=f, class_id=class_id)
            trn, val, tst   = split_data(X=X, Y=Y, frac=frac, rngseed=rngseed, class_id=class_id)

            self.trn += trn
            self.val += val
            self.tst += tst

        self.n_dims  = self.trn.x.shape[1]
        print(__name__ + f'.__init__: n_dims = {self.n_dims}')


class Data:
    """
    Args:
        x : data                [N vectors x D dimensions]
        y : target output data  [N vectors]
    """
    
    # constructor
    def __init__(self, x = np.array([]), y = np.array([])):
        self.N = x.shape[0]
        self.x = x
        self.y = y.flatten()

    # + operator
    def __add__(self, other):

        if (len(self.x) == 0): # still empty
            return other

        x = np.concatenate((self.x, other.x), axis=0)
        y = np.concatenate((self.y, other.y.flatten()), axis=0)

        return Data(x, y)

    # += operator
    def __iadd__(self, other):

        if (len(self.x) == 0): # still empty
            return other

        self.x = np.concatenate((self.x, other.x), axis=0)
        self.y = np.concatenate((self.y, other.y.flatten()), axis=0)
        self.N = len(self.y)
        return self

    # filter operator
    def classfilter(self, classid):

        x = self.x[self.y == classid]
        y = self.y[self.y == classid]
        
        return Data(x, y)


def split_data(X, Y, frac, rngseed, class_id = []):
    """ Split into [A = train & validation] + [B = test] sets
    """

    ### Permute events to have random mixing between classes (a must!)
    np.random.seed(int(rngseed)) # seed it!
    randi = np.random.permutation(X.shape[0])
    X = X[randi]
    Y = Y[randi].squeeze() # Squeeze, so it is for sure a single dimensional
    
    N     = X.shape[0]
    N_A   = round(N * frac)
    N_B   = N - N_A
    
    N_trn = round(N_A * frac)
    N_val = N_A - N_trn
    N_tst = N_B

    # A. Train
    X_trn = X[0:N_trn]
    Y_trn = Y[0:N_trn]

    # A. Validation
    X_val = X[N_trn:N_trn + N_val]
    Y_val = Y[N_trn:N_trn + N_val]

    # B. Test
    X_tst = X[N-N_tst:N]
    Y_tst = Y[N-N_tst:N]

    trn = Data()
    val = Data()
    tst = Data()

    # No spesific class selected
    if (class_id == []) : 

        trn = Data(x = X_trn, y = Y_trn)
        val = Data(x = X_val, y = Y_val)
        tst = Data(x = X_tst, y = Y_tst)
    
    # Loop over all classes selected
    else:

        for c in class_id : 
            ind  = (Y_trn[:,...] == c)
            trn += Data(x = X_trn[ind], y = np.ones(ind.sum())*c)

            ind  = (Y_val[:,...] == c)
            val += Data(x = X_val[ind], y = np.ones(ind.sum())*c)

            ind  = (Y_tst[:,...] == c)
            tst += Data(x = X_tst[ind], y = np.ones(ind.sum())*c)

    ### Permute events once again to have random mixing between classes
    def mix(data):
        randi  = np.random.permutation(data.x.shape[0])
        data.x = data.x[randi]
        data.y = data.y[randi]
        return data
    
    trn = mix(trn)
    val = mix(val)
    tst = mix(tst)
    
    print(__name__ + ".split_data: fractions [train: {:0.3f}, validate: {:0.3f}, test: {:0.3f}]".
        format(X_trn.shape[0]/ N, X_val.shape[0]/ N, X_tst.shape[0] / N))

    return trn, val, tst


def impute_data(X, imputer=None, dim=[], values=[-999], labels=[], algorithm='iterative', fill_value=0, knn_k=6):
    """ Data imputation (treatment of missing values, Nan and Inf).
    
    Args:
        X         : Input data matrix [N vectors x D dimensions]
        imputer   : Pre-trained imputator, default None
        dim       : Array of active dimensions to impute
        values    : List of special integer values indicating the need for imputation
        labels    : List containing textual description of input variables
        algorithm : 'constant', mean', 'median', 'iterative', knn_k'
        knn_k     : knn k-nearest neighbour parameter
        
    Returns:
        X       : Imputed output data
    """
    
    if dim == []:
        dim = np.arange(X.shape[1])

    if labels == []:
        labels = np.zeros(X.shape[1])

    N = X.shape[0]

    # Count NaN
    for j in dim:
        nan_ind = np.isnan(np.array(X[:,j], dtype=np.float))
        if np.sum(nan_ind) > 0:
            cprint(__name__ + f'.impute_data: Column {j} Number of {nan_ind} NaN found [{labels[j]}]', 'red')
    
    # Loop over dimensions
    for j in dim:

        # Set NaN for special values
        M_tot = 0
        for z in values:
            
            ind = np.isclose(np.array(X[:,j], dtype=np.float), z)
            X[ind, j] = np.nan
            
            M = np.sum(ind)
            M_tot += M

            if (M/N > 0):
                print(__name__ + f'.impute_data: Column {j} fraction [{M/N:0.3E}] with value {z} [{labels[j]}]')

        if (M_tot == N): # Protection, if all are now NaN
            # Set to zero so Imputer Function below does not remove the full column!!
            X[:,j] = 0.0

    # Treat infinities (inf)
    for j in dim:

        inf_ind = np.isinf(np.array(X[:,j], dtype=np.float))
        X[inf_ind, j] = np.nan
        if np.sum(inf_ind) > 0:
            cprint(__name__ + f'.impute_data: Column {j} Number of {np.sum(inf_ind)} Inf found [{labels[j]}]', 'red')
    
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

    print(__name__ + '.impute_data: [done] \n')
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
        Y[i,...] = (Y[i,...] - mu) / (std + EPS)
    return Y

def calc_zscore(X : np.array):
    """ Calculate 0-mean & unit-variance normalization.

    Args:
        X : Input with [# vectors x # dimensions]
    
    Returns:
        X_mu  : Mean vector
        X_std : Standard deviation vector 
    """

    X_mu  = np.zeros((X.shape[1]))
    X_std = np.zeros((X.shape[1]))

    # Calculate mean and std based on the training data
    for i in range(X.shape[1]) :
        X_mu[i]  = np.mean(X[:,i])
        X_std[i] = np.std(X[:,i])

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
        Y[:,i] = (X[:,i] - X_mu[i]) / (X_std[i] + EPS)
    return Y


@numba.njit(parallel=True)
def apply_madscore(X : np.array, X_m, X_mad, EPS=1E-12):
    """ MAD-score normalization
    """

    Y = np.zeros(X.shape)
    scale = 0.6745 # 0.75th of z-normal
    for i in range(len(X_m)):
        Y[:,i] = scale * (X[:,i] - X_m[i]) / (X_mad[i] + EPS)
    return Y


def pick_vars(data : DATASET, set_of_variables):
    """ Choose the active set of input variables.
    """

    newind  = np.where(np.isin(data.ids, set_of_variables))
    newind  = np.array(newind).flatten()
    newvars = []
    for i in newind :
        newvars.append(data.ids[i])

    return newind, newvars
