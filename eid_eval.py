# Electron ID [EVALUATION] code
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

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

# xgboost
import xgboost

# matplotlib
from matplotlib import pyplot as plt

# scikit
from sklearn         import metrics
from sklearn.metrics import accuracy_score

# icenet
import sys
sys.path.append(".")
import _icepaths_

from icenet.tools import io
from icenet.tools import aux
from icenet.tools import plots

from icenet.algo  import flr
from icenet.deep  import bnaf
from icenet.deep  import dopt
from icenet.deep  import dbnf
from icenet.deep  import mlgr
from icenet.deep  import maxo

# iceid
from configs.eid.mvavars import *
from iceid import common

targetdir = ''


# Main function
#
def main() :

    ### Get input
    data, args = common.init()

    #########################################################
    print('\nEvaluate ele_mva_value classifier ...')
    met_elemva = aux.Metric(y_true = data.tst.y, y_soft = data.tst.x[:, data.VARS.index('ele_mva_value')])
    
    roc_mstats.append(met_elemva)
    roc_labels.append('elemva')
    #########################################################
    
    ### Pick kinematic variables out
    newind, newvars = io.pick_vars(data, KINEMATIC_ID)
    
    data_kin       = copy.deepcopy(data)
    data_kin.trn.x = data.trn.x[:, newind]
    data_kin.val.x = data.val.x[:, newind]
    data_kin.tst.x = data.tst.x[:, newind]
    data_kin.VARS  = newvars

    ### Choose active variables
    newind, newvars = io.pick_vars(data, globals()[args['inputvar']])

    data.trn.x = data.trn.x[:, newind]
    data.val.x = data.val.x[:, newind]
    data.tst.x = data.tst.x[:, newind]    
    data.VARS  = newvars

    ### Execute
    global targetdir
    targetdir = f'./figs/eid/{args["config"]}/eval/'; os.makedirs(targetdir, exist_ok = True)

    evaluate(X = data.tst.x, y = data.tst.y, X_kin = data_kin.tst.x, VARS_kin = data_kin.VARS, args = args)

    print(__name__ + ' [Done]')


# Aux function to save results
roc_mstats = []
roc_labels = []


def saveit(func_predict, X, y, X_kin, VARS_kin, pt_edges, eta_edges, label):
    fig, ax, met = plots.binned_AUC(func_predict = func_predict, X = X, y = y, X_kin = X_kin, VARS_kin = VARS_kin, pt_edges = pt_edges, eta_edges = eta_edges, label = label)
    
    global roc_mstats
    global roc_labels
    roc_mstats.append(met)
    roc_labels.append(label)

    global targetdir
    filename = targetdir + '/' + label + '_AUC.pdf'
    plt.savefig(filename, bbox_inches='tight')


# Test the classifiers
#
def evaluate(X, y, X_kin, VARS_kin, args) :

    print(__name__ + ": Input with {} events and {} dimensions ".format(X.shape[0], X.shape[1]))

    modeldir = f'./checkpoint/eid/{args["config"]}/'; os.makedirs(modeldir, exist_ok = True)

    pt_edges  = args['plot_param']['pt_edges']
    eta_edges = args['plot_param']['eta_edges'] 

    ### Variable normalization
    if args['varnorm'] == 'zscore':

        print('\nZ-score normalizing variables ...')
        X_mu, X_std = pickle.load(open(modeldir + '/zscore.dat', 'rb'))
        X = io.apply_zscore(X, X_mu, X_std)

    elif args['varnorm'] == 'madscore':

        print('\nMAD-score normalizing variables ...')
        X_m, X_mad = pickle.load(open(modeldir + '/madscore.dat', 'rb'))
        X = io.apply_madscore(X, X_m, X_mad)

    ###
    if args['flr_param']['active']:

        label = args['flr_param']['label']
        print(f'\nEvaluate {label} classifier ...')

        b_pdfs, s_pdfs, bin_edges = pickle.load(open(modeldir + '/FLR_model_rw_' + args['reweight_param']['mode'] + '.dat', 'rb'))
        def func_predict(X):
            return flr.predict(X, b_pdfs, s_pdfs, bin_edges)

        # Evaluate (pt,eta) binned AUC
        saveit(func_predict = func_predict, X = X, y = y, X_kin = X_kin, VARS_kin = VARS_kin, pt_edges = pt_edges, eta_edges = eta_edges, label = label)
        
    ###
    if args['xgb_param']['active']:

        label = args['xgb_param']['label']
        print(f'\nEvaluate {label} classifier ...')

        xgb_model = pickle.load(open(modeldir + '/XGB_model_rw_' + args['reweight_param']['mode'] + '.dat', 'rb'))
        
        def func_predict(X):
            return xgb_model.predict(xgboost.DMatrix(data = X))

        # Evaluate (pt,eta) binned AUC
        saveit(func_predict = func_predict, X = X, y = y, X_kin = X_kin, VARS_kin = VARS_kin, pt_edges = pt_edges, eta_edges = eta_edges, label = label)


    # --------------------------------------------------------------------
    # For pytorch based
    X_ptr = torch.from_numpy(X).type(torch.FloatTensor)
    # --------------------------------------------------------------------

    ###
    if args['xtx_param']['active']:

        label = args['xtx_param']['label']
        y_tot      = np.array([])
        y_pred_tot = np.array([])

        AUC = np.zeros((len(pt_edges)-1, len(eta_edges)-1))

        for i in range(len(pt_edges) - 1):
            for j in range(len(eta_edges) - 1):

                pt_range  = [ pt_edges[i],  pt_edges[i+1]]
                eta_range = [eta_edges[j], eta_edges[j+1]]
                
                # Indices
                tst_ind = np.logical_and(aux.pick_ind(X_kin[:, VARS_kin.index('trk_pt')],   pt_range),
                                         aux.pick_ind(X_kin[:, VARS_kin.index('trk_eta')], eta_range))

                print('\nEvaluate {} classifier ...'.format(label))
                print('*** PT = [{:.3f},{:.3f}], ETA = [{:.3f},{:.3f}] ***'.format(
                    pt_range[0], pt_range[1], eta_range[0], eta_range[1]))
                xtx_model = aux.load_checkpoint('{}/{}_checkpoint_bin_{}_{}.pth'.format(modeldir, label, i, j))
                xtx_model.eval() # Turn on eval mode!
                
                signalclass = 1
                y_pred = xtx_model.softpredict(X_ptr)[tst_ind, signalclass].detach().numpy()
                
                met_xtx = aux.Metric(y_true = y[tst_ind], y_pred = y_pred)
                print('AUC = {:.5f}'.format(met_xtx.auc))

                # Accumulate
                y_tot      = np.concatenate((y_tot, y[tst_ind]))
                y_pred_tot = np.concatenate((y_pred_tot, y_pred))

                AUC[i,j]   = met_xtx.auc

        # Evaluate total performance
        met = aux.Metric(y_true = y_tot, y_pred = y_pred_tot)
        roc_mstats.append(met)
        roc_labels.append(label)

        fig,ax = plots.plot_auc_matrix(AUC, pt_edges, eta_edges)
        ax.set_title('{}: Integrated AUC = {:.3f}'.format(label, met.auc))
        
        targetdir = f'./figs/eid/{args["config"]}/eval/'; os.makedirs(targetdir, exist_ok = True)
        plt.savefig('{}/{}_AUC.pdf'.format(targetdir, label), bbox_inches='tight')
        
    ###
    if args['mlgr_param']['active']:

        label = args['mlgr_param']['label']
        print(f'\nEvaluate {label} classifier ...')

        mlgr_model = aux.load_checkpoint(modeldir + '/MLGR_checkpoint_rw_' + args['reweight_param']['mode'] + '.pth')
        mlgr_model.eval() # Turn on eval mode!
        
        def func_predict(X):
            signalclass = 1
            return mlgr_model.softpredict(X)[:, signalclass].detach().numpy()

        # Evaluate (pt,eta) binned AUC
        saveit(func_predict = func_predict, X = X_ptr, y = y, X_kin = X_kin, VARS_kin = VARS_kin, pt_edges = pt_edges, eta_edges = eta_edges, label = label)

    ###
    if args['dmax_param']['active']:

        label = args['dmax_param']['label']
        print(f'\nEvaluate {label} classifier ...')

        dmax_model = aux.load_checkpoint(modeldir + '/DMAX_checkpoint_rw_' + args['reweight_param']['mode'] + '.pth')
        dmax_model.eval() # Turn on eval mode!

        def func_predict(X):
            signalclass = 1
            return dmax_model.softpredict(X)[:, signalclass].detach().numpy()

        # Evaluate (pt,eta) binned AUC
        saveit(func_predict = func_predict, X = X_ptr, y = y, X_kin = X_kin, VARS_kin = VARS_kin, pt_edges = pt_edges, eta_edges = eta_edges, label = label)
        
    ###
    if args['dbnf_param']['active']:

        label = args['dbnf_param']['label']
        print(f'\nEvaluate {label} classifier ...')

        dbnf_param = args['dbnf_param']
        dbnf_param['n_dims'] = X.shape[1]
        def func_predict(X):
            y_pred_dbnf = dbnf.predict(dbnf_param, X_ptr.numpy(),
                          ['class_0_rw_' + args['reweight_param']['mode'],
                           'class_1_rw_' + args['reweight_param']['mode']], modeldir)

        # Evaluate (pt,eta) binned AUC
        saveit(func_predict = func_predict, X = X_ptr, y = y, X_kin = X_kin, VARS_kin = VARS_kin, pt_edges = pt_edges, eta_edges = eta_edges, label = label)


    ### Plot ROC curves
    targetdir = f'./figs/eid/{args["config"]}/eval/'; os.makedirs(targetdir, exist_ok = True)
    plots.ROC_plot(roc_mstats, roc_labels, title = 'reweight mode:' + args['reweight_param']['mode'],
        filename = targetdir + 'ROC')


if __name__ == '__main__' :

   main()

