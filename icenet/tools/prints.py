# Printing functions
# 
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import psutil
from typing import List
from prettytable import PrettyTable

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


def print_weights(weights, y, output_file=None):
    """
    Print event weights table
    """
    
    class_ids = np.unique(y.astype(int))
    
    table = PrettyTable(["class", "events", "sum(w)", "mean(w)", "std(w)",
                         "min(w)", "Q1(w)", "Q5(w)", "med(w)", "Q95(w)", "Q99(w)", "max(w)"]) 
    
    for c in class_ids:
        ind = (y == c)
        
        prc = np.percentile(weights[ind], [0, 1, 5, 50, 95, 99, 100])
        
        table.add_row([f'{c}',
                       f'{np.sum(ind)}',
                       f'{np.round(np.sum(weights[ind]), 2)}',
                       f'{np.mean(weights[ind]):0.3E}',
                       f'{np.std(weights[ind]):0.3E}',
                       f'{prc[0]:0.3E}',
                       f'{prc[1]:0.3E}',
                       f'{prc[2]:0.3E}',
                       f'{prc[3]:0.3E}',
                       f'{prc[4]:0.3E}',
                       f'{prc[5]:0.3E}',
                       f'{prc[6]:0.3E}'
                       ])
    
    print(table)
    print('')
    
    # Print to file
    if output_file is not None:
        with open(output_file, 'w') as f:
            print(table, file=f)
    
    return table


def print_variables(X : np.array, ids: List[str], W=None, exclude_vals=None, output_file=None):
    """ Print statistics of X
    
    Args:
        X            : array (n x dim)
        ids          : variable names (dim)
        W            : event weights
        exclude_vals : exclude special values from the stats
    
    Returns:
        prettyprint table of stats
    """
    
    print('\n')
    print(__name__ + f'.print_variables:')

    print(f'Excluding values: {exclude_vals}')
    
    table = PrettyTable(["i", "variable", "min", "Q1", "Q5", "med", "Q95", "Q99", "max", "# unique", "mean", "std", "#Inf", "#NaN"]) 
    
    for j in range(len(ids)):
        try:
            x   = np.array(X[:,j], dtype=np.float32).squeeze()
            ind = np.ones(len(x), dtype=bool)
            
            if exclude_vals is not None:
                # Exclude special values    
                for k in range(len(exclude_vals)):
                    ind = np.logical_and(ind, (x != exclude_vals[k]))

            x = x[ind]

            prc = np.percentile(x, [0, 1, 5, 50, 95, 99, 100])
            
            if W is not None:
                mean,std = aux.weighted_avg_and_std(values=x, weights=W[ind])
            else:
                mean,std = np.mean(x), np.std(x) 
            num_unique = len(np.unique(x))
            
            isinf  = np.sum(np.isinf(x))
            isnan  = np.sum(np.isnan(x))

            table.add_row([f'{j}',
                           f'{ids[j]}',
                           f'{prc[0]:10.2E}',
                           f'{prc[1]:10.2E}',
                           f'{prc[2]:10.2E}',
                           f'{prc[3]:10.2E}',
                           f'{prc[4]:10.2E}',
                           f'{prc[5]:10.2E}',
                           f'{prc[6]:10.2E}',
                           f'{num_unique}',
                           f'{mean:10.2E}', f'{std:10.2E}',
                           f'{isinf}', f'{isnan}'])
        
        except Exception as e:
            print(e)
            print(f'[{j: >3}] Cannot print variable "{ids[j]}" (probably non-scalar type)')
    
    print(table)
    print('')

    # Print to file
    if output_file is not None:
        with open(output_file, 'w') as f:
            print(table, file=f)
    
    return table
