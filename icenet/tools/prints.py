# Printing functions
# 
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import matplotlib.pyplot as plt
import numpy as np
import torch
import xgboost
from termcolor import colored

from . import aux


def printbar(marker='-', marks = 75):
    """ Print bar.
    """
    for i in range(marks):
        print(marker, end='')
    print('')


def colored_row(x, active_color='green', inactive_color='white', threshold=0.5, **kwargs):
    """ Color vector elements.
    """
    row = ''
    for i in range(len(x)):
        if x[i] > threshold:
            row += colored(f'{x[i]} ', active_color)
        else:
            row += colored(f'{x[i]} ', inactive_color)
    return row


def print_colored_matrix(X, **kwargs):
    """ Print matrix with two colors (suitable for binary matrices).
    """
    for i in range(X.shape[0]):
        print(colored_row(X[i,:], **kwargs))


def set_arr_format(precision):
    """ Set numpy array print format.
    """
    a = '{' + ':.{}f'.format(precision) + '}'
    float_formatter = a.format
    np.set_printoptions(formatter = {'float_kind' : float_formatter })


def printbranch(d):
    """ Print a branch.
    """
    for key,values in d.items():
        print('{:50s} [size = {}]'.format(key, values[0].size))
        print(values)
        print('')


def print_flow(flow):
    """ Print a cut or infoflows.
    """
    total = flow['total']
    for index, (key, value) in enumerate(flow.items()):
        frac = value / total
        print(f'{index} | {key:20s} | {value:6.0f} [{frac:6.4f}]')


def print_variables(X : np.array, VARS, active_dim = []):
    """ Print in a format (# samples x # dimensions)
    """
    print('\n')
    print(__name__ + f'.print_variables:')

    if active_dim == []:
        active_dim = np.arange(0,len(VARS))

    print(f'active_dim: {active_dim} \n')

    print('[i] variable_name : [min, med, max]   mean +- std   [[isinf, isnan]]')
    for j in active_dim:
        
        x = np.array(X[:,j], dtype=np.float)

        minval = np.min(x)
        maxval = np.max(x)
        mean   = np.mean(x)
        med    = np.median(x)
        std    = np.std(x)

        isinf  = np.any(np.isinf(x))
        isnan  = np.any(np.isnan(x))

        print('[{: >3}]{: >35} : [{: >10.2E}, {: >10.2E}, {: >10.2E}] \t {: >10.2E} +- {: >10.2E}   [[{}, {}]]'
            .format(j, VARS[j], minval, med, maxval, mean, std, isinf, isnan))
    print('\n')

