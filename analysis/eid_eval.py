# Electron ID [EVALUATION] code
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

from termcolor import cprint

# xgboost
import xgboost

# matplotlib
from matplotlib import pyplot as plt

# scikit
from sklearn         import metrics
from sklearn.metrics import accuracy_score

# icenet
from icenet.tools import io
from icenet.tools import aux
from icenet.tools import plots

from icenet.deep import predict

# iceid
from configs.eid.mvavars import *
from iceid import common
from iceid import graphio


targetdir = ''


# Main function
#
def main() :

    ### Get input
    data, args, features = common.init()

    data_graph = {}
    data_graph['tst'] = graphio.parse_graph_data(X=data.tst.x, Y=data.tst.y, VARS=data.VARS,
        features=features, global_on=args['graph_param']['global_on'], coord=args['graph_param']['coord'])
    
    
    #########################################################
    varname = 'ele_mva_value_depth15'

    print(f'\nEvaluate <{varname}> classifier ...')
    try:
        ind  = data.VARS.index(varname)
        y    = np.array(data.tst.y, dtype=np.float)
        yhat = np.array(data.tst.x[:,ind], dtype=np.float)

        met_elemva = aux.Metric(y_true = y, y_soft = yhat)
        roc_mstats.append(met_elemva)
        roc_labels.append('elemva15')
    except:
        print(__name__ + 'Variable not found')
    #########################################################

    ### Split and factor data
    data, data_tensor, data_kin = common.splitfactor(data=data, args=args)

    ### Execute
    global targetdir
    targetdir = f'./figs/eid/{args["config"]}/eval/'; os.makedirs(targetdir, exist_ok = True)
    args["modeldir"] = f'./checkpoint/eid/{args["config"]}/'; os.makedirs(args["modeldir"], exist_ok = True)

    evaluate(data=data, data_tensor=data_tensor, data_kin=data_kin, data_graph=data_graph, args=args)

    print(__name__ + ' [Done]')


# Aux function to save results
roc_mstats = []
roc_labels = []


def saveit(func_predict, X, y, X_kin, VARS_kin, pt_edges, eta_edges, label):
    """
    ROC curve plotter wrapper function.
    """
    fig, ax, met = plots.binned_AUC(func_predict = func_predict, X = X, y = y, X_kin = X_kin, \
        VARS_kin = VARS_kin, pt_edges = pt_edges, eta_edges = eta_edges, label = label)
    
    global roc_mstats
    global roc_labels
    roc_mstats.append(met)
    roc_labels.append(label)

    global targetdir
    filename = targetdir + '/' + label + '_AUC.pdf'
    plt.savefig(filename, bbox_inches='tight')


def evaluate(data, data_tensor, data_kin, data_graph, args):
    """
    Evaluate classifiers.
    """
    
    # --------------------------------------------------------------------
    ### Collect data
    X        = data.tst.x
    y        = data.tst.y
    X_kin    = data_kin.tst.x
    X_2D     = data_tensor['tst']
    X_graph  = data_graph['tst']

    VARS_kin = data_kin.VARS
    # --------------------------------------------------------------------

    print(__name__ + ": Input with {} events and {} dimensions ".format(X.shape[0], X.shape[1]))
    
    modeldir  = f'./checkpoint/eid/{args["config"]}/'; os.makedirs(modeldir, exist_ok = True)

    pt_edges  = args['plot_param']['pt_edges']
    eta_edges = args['plot_param']['eta_edges']     

    try:

        ### Tensor variable normalization
        if args['varnorm_tensor'] == 'zscore':

            print('\nZ-score normalizing tensor variables ...')
            X_mu_tensor, X_std_tensor = pickle.load(open(modeldir + '/zscore_tensor.dat', 'rb'))
            X_2D = io.apply_zscore_tensor(X_2D, X_mu_tensor, X_std_tensor)
        
        ### Variable normalization
        if args['varnorm'] == 'zscore':

            print('\nZ-score normalizing variables ...')
            X_mu, X_std = pickle.load(open(modeldir + '/zscore.dat', 'rb'))
            X = io.apply_zscore(X, X_mu, X_std)

        elif args['varnorm'] == 'madscore':

            print('\nMAD-score normalizing variables ...')
            X_m, X_mad = pickle.load(open(modeldir + '/madscore.dat', 'rb'))
            X = io.apply_madscore(X, X_m, X_mad)

    except:
        cprint('\n' + __name__ + f' WARNING: Problem in normalization. Continuing without! \n', 'red')
    
    # --------------------------------------------------------------------
    # For pytorch based
    X_ptr    = torch.from_numpy(X).type(torch.FloatTensor)
    X_2D_ptr = torch.from_numpy(X_2D).type(torch.FloatTensor)
    # --------------------------------------------------------------------

    # Loop over active models
    for i in range(len(args['active_models'])):

        ID = args['active_models'][i]
        param = args[f'{ID}_param']
        print(f'Training <{ID}> | {param} \n')
        
        if   param['predict'] == 'torch_graph':
            func_predict = predict.pred_torch(args=args, param=param)
            saveit(func_predict = func_predict, X = X_graph, y = y, X_kin = X_kin, VARS_kin = VARS_kin, pt_edges = pt_edges, eta_edges = eta_edges, label = param['label'])
        
        elif param['predict'] == 'graph_xgb':
            func_predict = predict.pred_graph_xgb(args=args, param=param)
            saveit(func_predict = func_predict, X = X_graph, y = y, X_kin = X_kin, VARS_kin = VARS_kin, pt_edges = pt_edges, eta_edges = eta_edges, label = param['label'])

        elif param['predict'] == 'flr':
            func_predict = predict.pred_flr(args=args, param=param)
            saveit(func_predict = func_predict, X = X, y = y, X_kin = X_kin, VARS_kin = VARS_kin, pt_edges = pt_edges, eta_edges = eta_edges, label = param['label'])

        elif param['predict'] == 'xgb':
            func_predict = predict.pred_xgb(args=args, param=param)
            saveit(func_predict = func_predict, X = X, y = y, X_kin = X_kin, VARS_kin = VARS_kin, pt_edges = pt_edges, eta_edges = eta_edges, label = param['label'])

        elif param['predict'] == 'torch_image':
            func_predict = predict.pred_torch(args=args, param=param)

            X_ = {}
            X_['x'] = X_2D_ptr # image tensors
            X_['u'] = X_ptr    # global features
            
            saveit(func_predict = func_predict, X = X_, y = y, X_kin = X_kin, VARS_kin = VARS_kin, pt_edges = pt_edges, eta_edges = eta_edges, label = param['label'])

        #elif param['predict'] == 'xtx':
        #    train.train_xtx(X_trn=X_trn, Y_trn=Y_trn, X_val=X_val, Y_val=Y_val, data_kin=data_kin, args=args, param=param)
        
        elif param['predict'] == 'torch_generic':
            func_predict = predict.pred_torch(args=args, param=param)
            saveit(func_predict = func_predict, X = X_ptr, y = y, X_kin = X_kin, VARS_kin = VARS_kin, pt_edges = pt_edges, eta_edges = eta_edges, label = param['label'])

        elif param['predict'] == 'torch_flow':
            func_predict = predict.pred_flow(args=args, param=param, n_dims=X_ptr.shape[1])
            saveit(func_predict = func_predict, X = X_ptr, y = y, X_kin = X_kin, VARS_kin = VARS_kin, pt_edges = pt_edges, eta_edges = eta_edges, label = param['label'])
        else:
            raise Exception(__name__ + f'.Unknown param["predict"] = {param["predict"]} for ID = {ID}')

    
    ### Plot all ROC curves
    targetdir = f'./figs/eid/{args["config"]}/eval/'; os.makedirs(targetdir, exist_ok = True)
    plots.ROC_plot(roc_mstats, roc_labels, title = 'training re-weight reference_class: ' + str(args['reweight_param']['reference_class']),
        filename = targetdir + 'ROC')


if __name__ == '__main__' :

   main()

