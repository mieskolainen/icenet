# Printing functions
# 
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import psutil
from typing import List

from termcolor import colored, cprint
from icenet.tools import aux


def print_RAM_usage():
    """ 
    """
    cprint(__name__ + f""".prints: Process RAM usage: {io.process_memory_use():0.2f} GB [total RAM in use: {psutil.virtual_memory()[2]} %]""", 'red')


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


def print_variables(X : np.array, ids: List[str], W=None, exclude_vals=None):
    """ Print in a format (# samples x # dimensions)
    """

    print('\n')
    print(__name__ + f'.print_variables:')

    print(f'Excluding values: {exclude_vals}')
    print('[i] variable_name : [min, med, max] [#unique]   mean +- std   [[isinf, isnan]]')
    
    for j in range(len(ids)):
        try:
            x   = np.array(X[:,j], dtype=np.float).squeeze()
            ind = np.ones(len(x), dtype=bool)
            
            if exclude_vals is not None:
                # Exclude special values    
                for k in range(len(exclude_vals)):
                    ind = np.logical_and(ind, (x != exclude_vals[k]))

            x = x[ind]

            minval     = np.min(x)
            med        = np.median(x)
            maxval     = np.max(x)
            mean,std   = aux.weighted_avg_and_std(values=x, weights=W[ind])
            num_unique = len(np.unique(x))

            isinf  = np.any(np.isinf(x))
            isnan  = np.any(np.isnan(x))

            print('[{: >3}]{: >35} : [{: >10.2E}, {: >10.2E}, {: >10.2E}] {: >10} \t ({: >10.2E} +- {: >10.2E})   [[{}, {}]]'
                .format(j, ids[j], minval, med, maxval, num_unique, mean, std, isinf, isnan))
        except Exception as e:
            print(e)
            print(f'[{j: >3}] Cannot print variable "{ids[j]}" (probably non-scalar type)')

    print('\n')

