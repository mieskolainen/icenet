# Electron ID [TRAINING] code
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
import graphviz

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
from icenet.tools import prints

from icenet.algo  import flr
from icenet.deep  import bnaf
from icenet.deep  import dopt
from icenet.deep  import dbnf
from icenet.deep  import mlgr
from icenet.deep  import maxo


from icenet.optim import adam
from icenet.optim import adamax
from icenet.optim import scheduler

# iceid
from configs.eid.mvavars import *
from iceid import common


# Main function
#
def main() :

    ### Get input
    data, args = common.init()
    
    ### Print ranges
    prints.print_variables(data.trn.x, data.VARS)

    ### Compute reweighting weights
    trn_weights = common.compute_reweights(data, args)
    
    
    ### Plot variables
    if args['plot_param']['basic_on']:
        targetdir = f'./figs/eid/{args["config"]}/train/1D_all/'
        os.makedirs(targetdir, exist_ok = True)
        plots.plotvars(X = data.trn.x, y = data.trn.y, NBINS = 70, VARS = data.VARS, weights = trn_weights, 
            targetdir = targetdir, title = f'training reweight reference: {args["reweight_param"]["mode"]}')


    ### Pick kinematic variables out
    newind, newvars = io.pick_vars(data, KINEMATIC_ID)

    data_kin        = copy.deepcopy(data)
    data_kin.trn.x  = data.trn.x[:, newind]
    data_kin.val.x  = data.val.x[:, newind]
    data_kin.tst.x  = data.tst.x[:, newind]
    data_kin.VARS   = newvars

    ### Choose active variables
    newind, newvars = io.pick_vars(data, globals()[args['inputvar']])

    data.trn.x = data.trn.x[:, newind]
    data.val.x = data.val.x[:, newind]
    data.tst.x = data.tst.x[:, newind]    
    data.VARS  = newvars

    ### Print variables
    fig,ax = plots.plot_correlations(data.trn.x, data.VARS)
    targetdir = f'./figs/eid/{args["config"]}/train/'; os.makedirs(targetdir, exist_ok = True)
    plt.savefig(fname = targetdir + 'correlations.pdf', pad_inches = 0.2, bbox_inches='tight')

    print(__name__ + ': Active variables:')
    prints.print_variables(data.trn.x, data.VARS)

    ### Execute
    train(data = data, data_kin = data_kin, trn_weights = trn_weights, args = args)
    print(__name__ + ' [done]')


# Main training function
#
def train(data, data_kin, trn_weights, args) :

    print(__name__ + f": Input with {data.trn.x.shape[0]} events and {data.trn.x.shape[1]} dimensions ")
    
    modeldir = f'./checkpoint/eid/{args["config"]}/'; os.makedirs(modeldir, exist_ok = True)

    # Truncate outliers (component by component) from the training set
    if args['outlier_param']['algo'] == 'truncate' :
        for j in range(data.trn.x.shape[1]):

            minval = np.percentile(data.trn.x[:,j], args['outlier_param']['qmin'])
            maxval = np.percentile(data.trn.x[:,j], args['outlier_param']['qmax'])

            data.trn.x[data.trn.x[:,j] < minval, j] = minval
            data.trn.x[data.trn.x[:,j] > maxval, j] = maxval

    # Variable normalization
    if args['varnorm'] == 'zscore' :

        print('\nZ-score normalizing variables ...')
        X_mu, X_std = io.calc_zscore(data.trn.x)
        data.trn.x  = io.apply_zscore(data.trn.x, X_mu, X_std)
        data.val.x  = io.apply_zscore(data.val.x, X_mu, X_std)

        # Save it for the evaluation
        pickle.dump([X_mu, X_std], open(modeldir + '/zscore.dat', 'wb'))

    elif args['varnorm'] == 'madscore' :

        print('\nMAD-score normalizing variables ...')
        X_m, X_mad  = io.calc_madscore(data.trn.x)
        data.trn.x  = io.apply_madscore(data.trn.x, X_m, X_mad)
        data.val.x  = io.apply_madscore(data.val.x, X_m, X_mad)

        # Save it for the evaluation
        pickle.dump([X_m, X_mad], open(modeldir + '/madscore.dat', 'wb'))
    
    prints.print_variables(data.trn.x, data.VARS)

    ### Pick training data into PyTorch format
    X_trn = torch.from_numpy(data.trn.x).type(torch.FloatTensor)
    Y_trn = torch.from_numpy(data.trn.y).type(torch.LongTensor)

    X_val = torch.from_numpy(data.val.x).type(torch.FloatTensor)
    Y_val = torch.from_numpy(data.val.y).type(torch.LongTensor)


    ### CLASSIFIER
    if args['flr_param']['active']:

        label = args['flr_param']['label']

        print(f'\nTraining {label} classifier ...')
        b_pdfs, s_pdfs, bin_edges = flr.train(X = data.trn.x, y = data.trn.y, weights = trn_weights, param = args['flr_param'])
        pickle.dump([b_pdfs, s_pdfs, bin_edges],
            open(modeldir + '/FLR_model_rw_' + args['reweight_param']['mode'] + '.dat', 'wb'))

        def func_predict(X):
            return flr.predict(X, b_pdfs, s_pdfs, bin_edges)

        ### Plot contours
        if args['plot_param']['contours_on']:
            targetdir = f'./figs/eid/{args["config"]}/train/2D_contours/flr/'; os.makedirs(targetdir, exist_ok = True)
            plots.plot_decision_contour(lambda x : func_predict(x),
                X = data.trn.x, y = data.trn.y, labels = data.VARS, targetdir = targetdir, matrix = 'numpy')

    ### CLASSIFIER
    if args['xgb_param']['active']:

        label = args['xgb_param']['label']

        print(f'\nTraining {label} classifier ...')
        dtrain = xgboost.DMatrix(data = data.trn.x, label = data.trn.y, weight = trn_weights)
        dtest  = xgboost.DMatrix(data = data.val.x, label = data.val.y)

        evallist = [(dtrain, 'train'), (dtest, 'eval')]
        results = dict()
        xgb_model = xgboost.train(params = args['xgb_param'], dtrain = dtrain,
            num_boost_round = args['xgb_param']['num_boost_round'], evals = evallist, evals_result = results, verbose_eval = True)

        ## Save
        pickle.dump(xgb_model, open(modeldir + '/XGB_model_rw_' + args['reweight_param']['mode'] + '.dat', 'wb'))

        losses   = results['train']['logloss']
        trn_aucs = results['train']['auc']
        val_aucs = results['eval']['auc']

        # Plot evolution
        plotdir  = f'./figs/eid/{args["config"]}/train/'; os.makedirs(plotdir, exist_ok = True)
        fig,ax = plots.plot_train_evolution(losses, trn_aucs, val_aucs, label)
        plt.savefig(f'{plotdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()

        ## Plot feature importance (xgb does Not return it for all of them)
        fscores = xgb_model.get_score(importance_type='gain')
        print(fscores)

        D = data.trn.x.shape[1]
        xx = np.arange(D)
        yy = np.zeros(D)

        for i in range(D):
            try:
                yy[i] = fscores['f' + str(i)]
            except:
                yy[i] = 0.0

        fig = plt.figure(figsize=(12,8))
        bars = plt.barh(xx, yy, align='center', height=0.5, tick_label=data.VARS)
        plt.xlabel('f-score')

        targetdir = f'./figs/eid/{args["config"]}/train'; os.makedirs(targetdir, exist_ok = True)
        plt.savefig(f'{targetdir}/xgb_importance.pdf', bbox_inches='tight'); plt.close()

        ## Plot decision tree
        #xgboost.plot_tree(xgb_model, num_trees=2)
        #plt.savefig('{}/xgb_tree.pdf'.format(targetdir), bbox_inches='tight'); plt.close()        
        
        ### Plot contours
        if args['plot_param']['contours_on']:
            targetdir = f'./figs/eid/{args["config"]}/train/2D_contours/xgb/'; os.makedirs(targetdir, exist_ok = True)
            plots.plot_decision_contour(lambda x : xgb_model.predict(x),
                X = X_trn, y = Y_trn, labels = data.VARS, targetdir = targetdir, matrix = 'xgboost')

    ### CLASSIFIER
    if args['mlgr_param']['active']:

        label = args['mlgr_param']['label']
        
        print(f'\nTraining {label} classifier ...')
        mlgr_model = mlgr.MLGR(D = X_trn.shape[1], C = 2)
        mlgr_model, losses, trn_aucs, val_aucs = dopt.train(model = mlgr_model, X_trn = X_trn, Y_trn = Y_trn, X_val = X_val, Y_val = Y_val,
            trn_weights = trn_weights, param = args['mlgr_param'])
        
        # Plot evolution
        plotdir  = f'./figs/eid/{args["config"]}/train/'; os.makedirs(plotdir, exist_ok = True)
        fig,ax = plots.plot_train_evolution(losses, trn_aucs, val_aucs, label)
        plt.savefig(f'{plotdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()

        ## Save
        checkpoint = {'model': mlgr.MLGR(D = X_trn.shape[1], C = 2), 'state_dict': mlgr_model.state_dict()}
        torch.save(checkpoint, modeldir + '/MLGR_checkpoint_rw_' + args['reweight_param']['mode'] + '.pth')

        ### Plot contours
        if args['plot_param']['contours_on']:
            targetdir = f'./figs/eid/{args["config"]}/train/2D_contours/mlgr/'; os.makedirs(targetdir, exist_ok = True)
            plots.plot_decision_contour(lambda x : mlgr_model.softpredict(x),
                X = X_trn, y = Y_trn, labels = data.VARS, targetdir = targetdir, matrix = 'torch')

    ### CLASSIFIER
    if args['xtx_param']['active']:

        label = args['xtx_param']['label']

        pt_edges  = args['plot_param']['pt_edges']
        eta_edges = args['plot_param']['eta_edges'] 
        
        for i in range(len(pt_edges) - 1):
            for j in range(len(eta_edges) - 1):

                pt_range  = [ pt_edges[i],  pt_edges[i+1]]
                eta_range = [eta_edges[j], eta_edges[j+1]]

                # Indices
                trn_ind = np.logical_and(aux.pick_ind(data_kin.trn.x[:, data_kin.VARS.index('trk_pt')],   pt_range),
                                         aux.pick_ind(data_kin.trn.x[:, data_kin.VARS.index('trk_eta')], eta_range))

                val_ind = np.logical_and(aux.pick_ind(data_kin.val.x[:, data_kin.VARS.index('trk_pt')],   pt_range),
                                         aux.pick_ind(data_kin.val.x[:, data_kin.VARS.index('trk_eta')], eta_range))

                print('*** PT = [{:.3f},{:.3f}], ETA = [{:.3f},{:.3f}] ***'.format(
                    pt_range[0], pt_range[1], eta_range[0], eta_range[1]))

                # Compute weights for this hyperbin (balance class ratios)
                yy = data.trn.y[trn_ind]
                frac = [0,0]
                for k in range(2):
                    frac[k] = np.sum(yy == k) / yy.shape[0]

                print('*** frac = [{:.3f},{:.3f}]'.format(frac[0], frac[1]))

                # Inverse weights
                weights = np.zeros(yy.shape[0])
                for k in range(weights.shape[0]):
                    weights[k] = 1.0 / frac[int(yy[k])] / yy.shape[0] / 2
                
                print('weightsum = {}'.format(np.sum(weights[yy == 0])))

                # Train
                xtx_model = mlgr.MLGR(D = X_trn.shape[1], C = 2)
                xtx_model, losses, trn_aucs, val_aucs = dopt.train(model = xtx_model,
                    X_trn = X_trn[trn_ind,:], Y_trn = Y_trn[trn_ind],
                    X_val = X_val[val_ind,:], Y_val = Y_val[val_ind], trn_weights = weights, param = args['xtx_param'])

                # Save
                checkpoint = {'model': mlgr.MLGR(D = X_trn.shape[1], C = 2), 'state_dict': xtx_model.state_dict()}
                torch.save(checkpoint, f'{modeldir}/XTX_checkpoint_bin_{i}_{j}.pth')

    ### CLASSIFIER
    if args['dmax_param']['active']:

        label = args['dmax_param']['label']
        
        print(f'\nTraining {label} classifier ...')
        dmax_model = maxo.MAXOUT(D = X_trn.shape[1], C = 2, num_units=args['dmax_param']['num_units'], neurons=args['dmax_param']['neurons'], dropout=args['dmax_param']['dropout'])
        dmax_model, losses, trn_aucs, val_aucs = dopt.train(model = dmax_model, X_trn = X_trn, Y_trn = Y_trn, X_val = X_val, Y_val = Y_val,
            trn_weights = trn_weights, param = args['dmax_param'])

        # Plot evolution
        plotdir  = f'./figs/eid/{args["config"]}/train/'; os.makedirs(plotdir, exist_ok = True)
        fig,ax = plots.plot_train_evolution(losses, trn_aucs, val_aucs, label)
        plt.savefig(f'{targetdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()

        ## Save
        checkpoint = {'model': maxo.MAXOUT(D = X_trn.shape[1], C = 2, num_units=args['dmax_param']['num_units'], neurons=args['dmax_param']['neurons'], dropout=args['dmax_param']['dropout']), 'state_dict': dmax_model.state_dict()}
        targetdir = f'./checkpoint/{args["config"]}/'; os.makedirs(targetdir, exist_ok = True)
        torch.save(checkpoint, modeldir + '/DMAX_checkpoint_rw_' + args['reweight_param']['mode'] + '.pth')
            
        ### Plot contours
        if args['plot_param']['contours_on']:
            targetdir = f'./figs/eid/{args["config"]}/train/2D_contours/dmax/'; os.makedirs(targetdir, exist_ok = True)
            plots.plot_decision_contour(lambda x : dmax_model.softpredict(x),
                X = X_trn, y = Y_trn, labels = data.VARS, targetdir = targetdir, matrix = 'torch')

    ### CLASSIFIER
    if args['dbnf_param']['active']:

        label = args['dbnf_param']['label']

        dbnf_param = args['dbnf_param']
        dbnf_param['n_dims'] = data.trn.x.shape[1]

        print(f'\nTraining {label} classifier ...')

        for classid in [0,1] :
            dbnf_param['model'] = 'class_' + str(classid) + '_rw_' + args['reweight_param']['mode']

            # Load datasets
            trn = data.trn.classfilter(classid)
            val = data.val.classfilter(classid)

            # Load re-weighting weights
            weights = trn_weights[data.trn.y == classid]

            # Create model
            dbnf_model = dbnf.create_model(dbnf_param, verbose = True)

            # Create optimizer & scheduler
            optimizer = adam.Adam(dbnf_model.parameters(), lr = dbnf_param['learning_rate'],
                amsgrad = True, polyak = dbnf_param['polyak'])

            sched = scheduler.ReduceLROnPlateau(optimizer, factor = dbnf_param['decay'],
                                          patience = dbnf_param['patience'], cooldown = dbnf_param['cooldown'],
                                          min_lr = dbnf_param['min_lr'], verbose = True,
                                          early_stopping = dbnf_param['early_stopping'],
                                          threshold_mode = 'abs')

            print(f'Training density for class = {classid} ...')
            dbnf.train(dbnf_model, optimizer, sched, trn.x, val.x, weights, dbnf_param, modeldir)


if __name__ == '__main__' :

   main()

