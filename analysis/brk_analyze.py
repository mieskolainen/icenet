# B/RK analysis [EVALUATION] code
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

# icenet system paths
import sys
sys.path.append(".")

# Configure plotting backend
import matplotlib
matplotlib.use('Agg')

import h5py
import uproot
import pickle
import xgboost
import matplotlib as plt
import os
import torch
import numpy as np

import numba
from numba import jit

# icenet
from iceplot import iceplot


from icenet.tools import process
from icenet.tools import aux
from icenet.tools import io
from icenet.tools import prints

# icebrk
from icebrk import common
from icebrk import loop
from icebrk import histos
from icebrk import features


def print_pred(output, BMAT):
    """ Predictions by different algorithms """

    for key in output['P'].keys():

        P       = output['P'][key]
        
        P_sum   = np.sum(P,    axis=0)
        P_sum2  = np.sum(P**2, axis=0)
        Y_i_hat = np.argmax(P, axis=1)

        print(f'\n<< INTEGRAL ESTIMATE | ALGORITHM {key} >>')

        for i in range(len(P_sum)):
            value = P_sum[i]
            error = np.sqrt(P_sum2[i])
            print(prints.colored_row(BMAT[i,:]) + f' {i:4d} : {value:6.1f} +- {error:6.1f} counts')
        prints.set_arr_format(0)
        print(f'\nargmax:     {Y_i_hat}')

        S   = np.sum(P_sum[1:])
        S_E = np.sqrt(np.sum(P_sum2[1:]))
        print(f'SIGNAL: {S:.2f} +- {S_E:.2f}')


def print_true(output, BMAT):
    """ MC ground truth """

    Y = output['Y']

    Y_sum = np.zeros(BMAT.shape[0])
    Y_i   = aux.binvec2powersetindex(Y, BMAT)

    for i in range(len(Y_sum)):
        Y_sum[i] = np.sum(Y_i == i)

    print('\n<< MC GROUND TRUTH >>')
    for i in range(len(Y_sum)):
        value = Y_sum[i]
        error = np.sqrt(value) # Simple poisson
        print(prints.colored_row(BMAT[i,:]) + f'{i:4d} : {value:6.1f} +- {error:6.1f} counts')
    prints.set_arr_format(0)
    print(f'\nargmax:     {Y_i}')

    S   = np.sum(Y_sum[1:])
    S_E = np.sqrt(np.sum(Y_sum[1:]))
    print(f'SIGNAL: {S:.2f} +- {S_E:.2f}')


# Main function
#
def main() :
    
    args, cli = process.read_config(config_path='./configs/brk')
    iodir = aux.makedir(f'./output/{args["rootname"]}/{cli.tag}/')
    paths = io.glob_expand_files(datasets=cli.datasets, datapath=cli.datapath)

    targetdir = aux.makedir(f'./figs/{args["rootname"]}/{cli.tag}/eval/')
    
    # ====================================================================

    ## Binary matrix
    BMAT = aux.generatebinary(args['MAXT3'], args['MAXN'])
    print(f'POWERSET [NCLASS = {BMAT.shape[0]}]:')
    prints.print_colored_matrix(BMAT)
    
    ### Load pickle data
    MC_output = pickle.load(open(iodir + 'MC_output.pkl', 'rb')) # (r)ead (b)inary
    DA_output = pickle.load(open(iodir + 'DA_output.pkl', 'rb')) # (r)ead (b)inary
    
    print(MC_output)
    print(DA_output)


    ### Histograms of observables
    print('\n** PLOTTING HISTOGRAMS **')

    MC_template = {
        'data'   : None,
        'weights': None,
        'label'  : 'MC (reco) truth',
        'hfunc'  : 'hist',
        'style'  : iceplot.hist_style_step,
        'obs'    : None,
        'hdata'  : None,
        'color'  : (0,0,0)
    }
    dataA_template = {
        'data'   : None,
        'weights': None,
        'label'  : '$icenet$ (DEPS)',
        'hfunc'  : 'errorbar',
        'style'  : iceplot.errorbar_style,
        'obs'    : None,
        'hdata'  : None,
        'color'  : iceplot.imperial_green
    }
    dataB_template = {
        'data'   : None,
        'weights': None,
        'label'  : '$icenet$ (MAXO)',
        'hfunc'  : 'errorbar',
        'style'  : iceplot.errorbar_style,
        'obs'    : None,
        'hdata'  : None,
        'color'  : iceplot.imperial_dark_blue
    }
    dataC_template = {
        'data'   : None,
        'weights': None,
        'label'  : '$icenet$ (XGB)',
        'hfunc'  : 'errorbar',
        'style'  : iceplot.errorbar_style,
        'obs'    : None,
        'hdata'  : None,
        'color'  : iceplot.imperial_dark_red
    }

    ## Over all observables
    for obs in DA_output['hobj']['S']['0'].keys():

        print(__name__ + f': Histogram [{obs}]')
        sources = []

        # MC
        sources.append(MC_template.copy())
        
        # DATA
        sources.append(dataA_template.copy())
        sources.append(dataB_template.copy())
        sources.append(dataC_template.copy())

        # Update the observables
        for i in range(len(sources)): sources[i].update({'obs' : histos.obs_all[obs]})

        # MC
        k = 0
        for i in range(len(MC_output['hobj']['S'])):
            if MC_output['hobj']['S'][str(i)][obs].bins is not None:
                sources[k]['hdata'] = MC_output['hobj']['S'][str(i)][obs]
                k += 1
        # DATA
        for i in range(len(DA_output['hobj']['S'])):
            if DA_output['hobj']['S'][str(i)][obs].bins is not None:
                sources[k]['hdata'] = DA_output['hobj']['S'][str(i)][obs]
                k += 1
        # Plot
        for yscale in ['linear']:
            iceplot.set_global_style()
            fig1, ax1 = iceplot.superplot(sources, ratio_plot=True, yscale=yscale, legend_counts=True)
            fig1.savefig(f'{targetdir}/h_{obs}_{yscale}.pdf', bbox_inches='tight')

    ### PERFORMANCE
    if len(MC_output['Y']) > 0:
        print_true(MC_output, BMAT)

    if len(DA_output['P']) > 0:
        print_pred(DA_output, BMAT)

    print('\n' + __name__+ ' DONE')

if __name__ == '__main__' :

    main()
