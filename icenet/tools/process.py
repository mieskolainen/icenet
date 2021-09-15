# Common input & data reading routines
#
# Mikael Mieskolainen, 2021
# m.mieskolainen@imperial.ac.uk

import argparse
import yaml
import numpy as np
from termcolor import colored, cprint

import torch

from icenet.tools import aux


# Command line arguments
from glob import glob
from braceexpand import braceexpand


def read_config(config_path='./configs/eid'):
    """
    Commandline and YAML configuration reader
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type = str, default='tune0')
    parser.add_argument("--datapath", type = str, default=".")
    parser.add_argument("--datasets", type = str, default="*")

    cli = parser.parse_args()
    
    # -------------------------------------------------------------------
    ## Read configuration
    args = {}
    config_yaml_file = cli.config + '.yml'
    with open(config_path + '/' + config_yaml_file, 'r') as stream:
        try:
            args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            
    args['config'] = cli.config

    # -------------------------------------------------------------------
    ### Set image and graph constructions on/off
    args['graph_on'] = False
    args['image_on'] = False

    for i in range(len(args['active_models'])):
        ID    = args['active_models'][i]
        param = args[f'{ID}_param']

        if ('graph' in param['train']) or ('graph' in param['predict']):
            args['graph_on'] = True
        if ('image' in param['train']) or ('image' in param['predict']):
            args['image_on'] = True

    print('\n')
    cprint(__name__ + f'.read_config: graph_on = {args["graph_on"]}', 'yellow')
    cprint(__name__ + f'.read_config: image_on = {args["image_on"]}', 'yellow')    

    # -------------------------------------------------------------------
    # Do brace expansion
    datasets = list(braceexpand(cli.datasets))

    # Parse input files into a list
    args['root_files'] = list()
    for data in datasets:
        filepath = glob(cli.datapath + '/' + data + '.root')
        if filepath != []:
            for i in range(len(filepath)):
                args['root_files'].append(filepath[i])
    # -------------------------------------------------------------------
    
    print(args)
    print('')
    print(" torch.__version__: " + torch.__version__)
    print("")
    print(" Try 'filename_*' ")
    print(" Try 'filename_[0-99]' ")
    print(" Try 'filename_0' ")
    print(" Try 'filename_{0,3,4}' ")
    print(" Google <glob wildcards> and brace expansion.")
    print("")

    return args, cli


def compute_eta_pt_reweights(data, args, N_class=2, EPS=1e-12):
    """ Compute (eta,pt) reweighting coefficients.
    Args:
        data    : training data object
        args    : arguments object
    Returns:
        weights : array of re-weights
    """

    ### Re-weighting variables
    RV = {}
    RV['pt']  = data.trn.x[:,data.VARS.index('trk_pt')].astype(np.float)
    RV['eta'] = data.trn.x[:,data.VARS.index('trk_eta')].astype(np.float)
    
    
    ### Pre-transform
    for var in ['pt', 'eta']:
        mode = args['reweight_param'][f'transform_{var}']

        if   mode == 'log10':
            RV[var] = np.log10(RV[var] + EPS)

            # Bins
            args['reweight_param'][f'bins_{var}'][0] = np.log10(args['reweight_param'][f'bins_{var}'][0] + EPS)
            args['reweight_param'][f'bins_{var}'][1] = np.log10(args['reweight_param'][f'bins_{var}'][1])

        elif mode == 'sqrt':
            RV[var] = np.sqrt(RV[var])

            # Bins
            args['reweight_param'][f'bins_{var}'][0] = np.sqrt(args['reweight_param'][f'bins_{var}'][0])
            args['reweight_param'][f'bins_{var}'][1] = np.sqrt(args['reweight_param'][f'bins_{var}'][1])

        elif mode == 'square':
            RV[var] = RV[var]**2

            # Bins
            args['reweight_param'][f'bins_{var}'][0] = (args['reweight_param'][f'bins_{var}'][0])**2
            args['reweight_param'][f'bins_{var}'][1] = (args['reweight_param'][f'bins_{var}'][1])**2

        elif mode == None:
            True
        else:
            raise Except(__name__ + '.compute_reweights: Unknown pre-transform')

    # Binning setup
    binedges = {}
    for var in ['pt', 'eta']:
        if   args['reweight_param'][f'binmode_{var}'] == 'linear':
            binedges[var] = np.linspace(
                                 args['reweight_param'][f'bins_{var}'][0],
                                 args['reweight_param'][f'bins_{var}'][1],
                                 args['reweight_param'][f'bins_{var}'][2])

        elif args['reweight_param'][f'binmode_{var}'] == 'log':
            binedges[var] = np.logspace(
                                 np.log10(np.max([args['reweight_param'][f'bins_{var}'][0], EPS])),
                                 np.log10(args['reweight_param'][f'bins_{var}'][1]),
                                 args['reweight_param'][f'bins_{var}'][2], base=10)
        else:
            raise Except(__name__ + ': Unknown re-weight binning mode')
    
    print(__name__ + f".compute_reweights: reference_class: <{args['reweight_param']['reference_class']}>")


    ### Compute 2D-PDFs for each class
    pdf     = {}
    for c in range(N_class):
        pdf[c] = aux.pdf_2D_hist(X_A=RV['pt'][data.trn.y==c], X_B=RV['eta'][data.trn.y==c],
                                    binedges_A=binedges['pt'], binedges_B=binedges['eta'])

    pdf['binedges_A'] = binedges['pt']
    pdf['binedges_B'] = binedges['eta']


    # Compute event-by-event weights
    if args['reweight_param']['reference_class'] != -1:
        
        trn_weights = aux.reweightcoeff2D(
            X_A = RV['pt'], X_B = RV['eta'], pdf = pdf, y = data.trn.y, N_class=N_class,
            equal_frac       = args['reweight_param']['equal_frac'],
            reference_class  = args['reweight_param']['reference_class'],
            max_reg          = args['reweight_param']['max_reg'])
    else:
        # No re-weighting
        weights_doublet = np.zeros((data.trn.x.shape[0], N_class))
        for c in range(N_class):    
            weights_doublet[data.trn.y == c, c] = 1
        trn_weights = np.sum(weights_doublet, axis=1)

    # Compute the sum of weights per class for the output print
    frac = np.zeros(N_class)
    sums = np.zeros(N_class)
    for c in range(N_class):
        frac[c] = np.sum(data.trn.y == c)
        sums[c] = np.sum(trn_weights[data.trn.y == c])
    
    print(__name__ + f'.compute_reweights: sum[trn.y==c]: {frac}')
    print(__name__ + f'.compute_reweights: sum[trn_weights[trn.y==c]]: {sums}')
    print(__name__ + f'.compute_reweights: [done] \n')
    
    return trn_weights


def compute_reweights_XY(X, Y, VARS, args):
    """ Compute (eta,pt) reweighting coefficients.

    Args:
        data    : training data object
        args    : arguments object
    Returns:
        weights : array of re-weights
    """

    # ...
    
    return True
