# Electron ID [TRAINING] steering code
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

# icenet system paths
import _icepaths_

import math
import numpy as np
import torch
import argparse
import pprint
import os
import datetime
import json
import pickle
import sys
import yaml
import copy
#import graphviz
import torch_geometric
from termcolor import cprint

# matplotlib
from matplotlib import pyplot as plt

# scikit
from sklearn         import metrics
from sklearn.metrics import accuracy_score

# icenet
from icenet.tools import io
from icenet.tools import aux
from icenet.tools import plots
from icenet.tools import prints

# deep learning
from icenet.deep  import train

# iceid
from iceid import common
from iceid import graphio


# Main function
#
def main() :

    ### Get input
    data, args, features = common.init()


    ### Print ranges
    #prints.print_variables(X=data.trn.x, VARS=data.VARS)
    
    ### Compute reweighting weights
    trn_weights = common.compute_reweights(data=data, args=args)
        

    ### Plot some kinematic variables
    targetdir = f'./figs/eid/{args["config"]}/reweight/1D_kinematic/'
    os.makedirs(targetdir, exist_ok = True)
    for k in ['trk_pt', 'trk_eta', 'trk_phi', 'trk_p']:
        plots.plotvar(x = data.trn.x[:, data.VARS.index(k)], y = data.trn.y, weights = trn_weights, var = k, NBINS = 70,
            targetdir = targetdir, title = f"training re-weight reference_class: {args['reweight_param']['reference_class']}")
    
    # --------------------------------------------------------------------

    ### Parse data into graphs
    graph = {}
    if args['graph_on']:
        graph['trn'] = graphio.parse_graph_data(X=data.trn.x, Y=data.trn.y, VARS=data.VARS, 
            features=features, global_on=args['graph_param']['global_on'], coord=args['graph_param']['coord'])
        graph['val'] = graphio.parse_graph_data(X=data.val.x, Y=data.val.y, VARS=data.VARS,
            features=features, global_on=args['graph_param']['global_on'], coord=args['graph_param']['coord'])
    

    ### Plot variables
    if args['plot_param']['basic_on'] == True:
        print(__name__ + f': plotting basic histograms ...')
        targetdir = f'./figs/eid/{args["config"]}/train/1D_all/'; os.makedirs(targetdir, exist_ok = True)
        plots.plotvars(X = data.trn.x, y = data.trn.y, NBINS = 70, VARS = data.VARS,
            weights = trn_weights, targetdir = targetdir, title = f'training reweight reference: {args["reweight_param"]["mode"]}')

    ### Split and factor data
    data, data_tensor, data_kin = common.splitfactor(data=data, args=args)
    
    ### Print scalar variables
    fig,ax = plots.plot_correlations(data.trn.x, data.VARS)
    targetdir = f'./figs/eid/{args["config"]}/train/'; os.makedirs(targetdir, exist_ok = True)
    plt.savefig(fname = targetdir + 'correlations.pdf', pad_inches = 0.2, bbox_inches='tight')
    
    print(__name__ + ': Active variables:')
    prints.print_variables(X=data.trn.x, VARS=data.VARS)
    

    # Add args['modeldir']
    args["modeldir"] = f'./checkpoint/eid/{args["config"]}/'; os.makedirs(args["modeldir"], exist_ok = True)
    

    ### Execute training
    trainloop(data = data, data_tensor = data_tensor, data_kin = data_kin, data_graph = graph, trn_weights = trn_weights, args = args)
    
    print(__name__ + ' [done]')


# Main training function
#
def trainloop(data, data_tensor, data_kin, data_graph, trn_weights, args) :
    print(__name__ + f": Input with {data.trn.x.shape[0]} events and {data.trn.x.shape[1]} dimensions ")

    # @@ Tensor normalization @@
    if args['image_on'] and (args['varnorm_tensor'] == 'zscore'):
            
        print('\nZ-score normalizing tensor variables ...')
        X_mu_tensor, X_std_tensor = io.calc_zscore_tensor(data_tensor['trn'])
        for key in ['trn', 'val']:
            data_tensor[key] = io.apply_zscore_tensor(data_tensor[key], X_mu_tensor, X_std_tensor)
        
        # Save it for the evaluation
        pickle.dump([X_mu_tensor, X_std_tensor], open(args["modeldir"] + '/zscore_tensor.dat', 'wb'))    
    
    # --------------------------------------------------------------------

    # @@ Truncate outliers (component by component) from the training set @@
    if args['outlier_param']['algo'] == 'truncate' :
        for j in range(data.trn.x.shape[1]):

            minval = np.percentile(data.trn.x[:,j], args['outlier_param']['qmin'])
            maxval = np.percentile(data.trn.x[:,j], args['outlier_param']['qmax'])

            data.trn.x[data.trn.x[:,j] < minval, j] = minval
            data.trn.x[data.trn.x[:,j] > maxval, j] = maxval

    # @@ Variable normalization @@
    if args['varnorm'] == 'zscore' :

        print('\nZ-score normalizing variables ...')
        X_mu, X_std = io.calc_zscore(data.trn.x)
        data.trn.x  = io.apply_zscore(data.trn.x, X_mu, X_std)
        data.val.x  = io.apply_zscore(data.val.x, X_mu, X_std)

        # Save it for the evaluation
        pickle.dump([X_mu, X_std], open(args['modeldir'] + '/zscore.dat', 'wb'))

    elif args['varnorm'] == 'madscore' :

        print('\nMAD-score normalizing variables ...')
        X_m, X_mad  = io.calc_madscore(data.trn.x)
        data.trn.x  = io.apply_madscore(data.trn.x, X_m, X_mad)
        data.val.x  = io.apply_madscore(data.val.x, X_m, X_mad)

        # Save it for the evaluation
        pickle.dump([X_m, X_mad], open(args['modeldir'] + '/madscore.dat', 'wb'))
    
    prints.print_variables(data.trn.x, data.VARS)

    ### Pick training data into PyTorch format
    X_trn = torch.from_numpy(data.trn.x).type(torch.FloatTensor)
    Y_trn = torch.from_numpy(data.trn.y).type(torch.LongTensor)

    X_val = torch.from_numpy(data.val.x).type(torch.FloatTensor)
    Y_val = torch.from_numpy(data.val.y).type(torch.LongTensor)


    # Loop over active models
    for i in range(len(args['active_models'])):

        ID    = args['active_models'][i]
        param = args[f'{ID}_param']
        print(f'Training <{ID}> | {param} \n')

        if   param['train'] == 'graph':
            train.train_graph(data_trn=data_graph['trn'], data_val=data_graph['val'], args=args, param=param)

        elif param['train'] == 'graph_xgb':
            train.train_graph_xgb(data_trn=data_graph['trn'], data_val=data_graph['val'], trn_weights=trn_weights, args=args, param=param)  
        
        elif param['train'] == 'flr':
            train.train_flr(data=data, trn_weights=trn_weights, args=args,param=param)
        
        elif param['train'] == 'xgb':
            train.train_xgb(data=data, trn_weights=trn_weights, args=args, param=param)

        elif param['train'] == 'cnn':
            train.train_cnn(data=data, data_tensor=data_tensor, Y_trn=Y_trn, Y_val=Y_val, trn_weights=trn_weights, args=args, param=param)
            
        #elif param['train'] == 'xtx':
        #    train.train_xtx(X_trn=X_trn, Y_trn=Y_trn, X_val=X_val, Y_val=Y_val, data_kin=data_kin, args=args, param=param)

        elif param['train'] == 'dmlp':
            train.train_dmlp(X_trn=X_trn, Y_trn=Y_trn, X_val=X_val, Y_val=Y_val, trn_weights=trn_weights, args=args, param=param)
        
        elif param['train'] == 'lgr':
            train.train_lgr(X_trn=X_trn,  Y_trn=Y_trn, X_val=X_val, Y_val=Y_val, trn_weights=trn_weights, args=args, param=param)
                
        elif param['train'] == 'dmax':
            train.train_dmax(X_trn=X_trn, Y_trn=Y_trn, X_val=X_val, Y_val=Y_val, trn_weights=trn_weights, args=args, param=param)

        elif param['train'] == 'flow':
            train.train_flow(data=data, trn_weights=trn_weights, args=args, param=param)

        else:
            raise Exception(__name__ + f'.Unknown param["train"] = {param["train"]} for ID = {ID}')


if __name__ == '__main__' :

   main()

