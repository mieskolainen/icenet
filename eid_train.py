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
import torch_geometric

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
from icenet.tools import prints

from icenet.algo  import flr
from icenet.deep  import bnaf
from icenet.deep  import dopt
from icenet.deep  import dbnf
from icenet.deep  import mlgr
from icenet.deep  import maxo
from icenet.deep  import cnn
from icenet.deep  import graph


from icenet.optim import adam
from icenet.optim import adamax
from icenet.optim import scheduler

# iceid
from iceid import common
from iceid import graphio


modeldir = ''


# Main function
#
def main() :

    ### Get input
    data, args, features = common.init()

    ### Print ranges
    #prints.print_variables(X=data.trn.x, VARS=data.VARS)
    
    ### Compute reweighting weights
    trn_weights = common.compute_reweights(data=data, args=args)
    
    graph = {}
    graph['trn'] = graphio.parse_graph_data(X=data.trn.x, Y=data.trn.y, VARS=data.VARS, features=features)
    graph['val'] = graphio.parse_graph_data(X=data.val.x, Y=data.val.y, VARS=data.VARS, features=features)
    graph['tst'] = graphio.parse_graph_data(X=data.tst.x, Y=data.tst.y, VARS=data.VARS, features=features)

    
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


    global modeldir
    modeldir = f'./checkpoint/eid/{args["config"]}/'; os.makedirs(modeldir, exist_ok = True)
    
    
    ### Execute training
    train(data = data, data_tensor = data_tensor, data_kin = data_kin, trn_weights = trn_weights, args = args)
    
    ### Graph net based training
    graph_train(data_trn=graph['trn'], data_val=graph['val'], args=args)
    
    print(__name__ + ' [done]')


# Graphnet training function
#
def graph_train(data_trn, data_val, args, num_classes=2):
    
    num_node_features   = data_trn[0].x.shape[1]
    num_global_features = len(data_trn[0].u)
    
    conv_type = args['gnet_param']['conv_type']
    if   conv_type == 'GAT':
        model = graph.GATNet(D = num_node_features, C = num_classes, G = num_global_features, task='graph')
    elif conv_type == 'DEC':
        model = graph.DECNet(D = num_node_features, C = num_classes, G = num_global_features, task='graph')
    elif conv_type == 'SG':
        model = graph.SGNet(D = num_node_features, C = num_classes, G = num_global_features, task='graph')
    elif conv_type == 'spline':
        model = graph.SplineNet(D = num_node_features, C = num_classes, G = num_global_features, task='graph')
    else:
        raise Except(name__ + f'.graph_train: Unknown network convolution model "conv_type" = {conv_type}')
    
    # CPU or GPU
    model, device = dopt.model_to_cuda(model=model, device_type=args['gnet_param']['device'])

    # Count the number of parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(__name__ + f'.graph_train: Number of free parameters = {params}')

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args['gnet_param']['learning_rate'], weight_decay=args['gnet_param']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Data loaders
    train_loader = torch_geometric.data.DataLoader(data_trn, batch_size=args['gnet_param']['batch_size'], shuffle=True)
    test_loader  = torch_geometric.data.DataLoader(data_val, batch_size=512, shuffle=False)
    
    for epoch in range(args['gnet_param']['epochs']):
        loss               = graph.train(model=model, loader=train_loader, optimizer=optimizer, device=device)
        test_acc, test_auc = graph.test( model=model, loader=test_loader,  optimizer=optimizer, device=device)

        print(f'Epoch {epoch+1:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f} (ACC), {test_auc:.4f} (AUC)')
        scheduler.step()

    ## Save
    label = args['gnet_param']['label']
    checkpoint = {'model': model, 'state_dict': model.state_dict()}
    torch.save(checkpoint, modeldir + f'/{label}_checkpoint_rw_' + args['reweight_param']['mode'] + '.pth')


# Main training function
#
def train(data, data_tensor, data_kin, trn_weights, args) :
    print(__name__ + f": Input with {data.trn.x.shape[0]} events and {data.trn.x.shape[1]} dimensions ")

    # @@ Tensor normalization @@
    if args['varnorm_tensor'] == 'zscore':
        
        print('\nZ-score normalizing tensor variables ...')
        X_mu_tensor, X_std_tensor = io.calc_zscore_tensor(data_tensor['trn'])
        for key in ['trn', 'val']:
            data_tensor[key] = io.apply_zscore_tensor(data_tensor[key], X_mu_tensor, X_std_tensor)
        
        # Save it for the evaluation
        pickle.dump([X_mu_tensor, X_std_tensor], open(modeldir + '/zscore_tensor.dat', 'wb'))    
    
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

    # -------------------------------------------------------------------------------
    # Into torch format

    X_trn_2D = torch.tensor(data_tensor['trn'], dtype=torch.float)
    X_val_2D = torch.tensor(data_tensor['val'], dtype=torch.float)
    DIM = X_trn_2D.shape
    # -------------------------------------------------------------------------------

    ### CLASSIFIER
    if args['flr_param']['active']:

        label = args['flr_param']['label']

        print(f'\nTraining {label} classifier ...')
        b_pdfs, s_pdfs, bin_edges = flr.train(X = data.trn.x, y = data.trn.y, weights = trn_weights, param = args['flr_param'])
        pickle.dump([b_pdfs, s_pdfs, bin_edges],
            open(modeldir + f'/{label}_model_rw_' + args['reweight_param']['mode'] + '.dat', 'wb'))

        def func_predict(X):
            return flr.predict(X, b_pdfs, s_pdfs, bin_edges)

        ### Plot contours (TOO SLOW!)
        """
        if args['plot_param']['contours_on']:
            targetdir = f'./figs/eid/{args["config"]}/train/2D_contours/{label}/'; os.makedirs(targetdir, exist_ok = True)
            plots.plot_decision_contour(lambda x : func_predict(x),
                X = data.trn.x, y = data.trn.y, labels = data.VARS, targetdir = targetdir, matrix = 'numpy')
        """
    
    ### CLASSIFIER
    if args['cnn_param']['active']:

        label = args['cnn_param']['label']

        print(f'\nTraining {label} classifier ...')
        cnn_model = cnn.CNN(C=2, nchannels=DIM[1], nrows=DIM[2], ncols=DIM[3], \
            dropout_cnn=args['cnn_param']['dropout_cnn'], dropout_mlp=args['cnn_param']['dropout_mlp'], mlp_dim=args['cnn_param']['mlp_dim'])
        cnn_model, losses, trn_aucs, val_aucs = \
            dopt.train(model = cnn_model, X_trn = X_trn_2D, Y_trn = Y_trn, X_val = X_val_2D, Y_val = Y_val,
                        trn_weights = trn_weights, param = args['cnn_param'])
        
        # Plot evolution
        plotdir = f'./figs/eid/{args["config"]}/train/'; os.makedirs(plotdir, exist_ok=True)
        fig,ax  = plots.plot_train_evolution(losses, trn_aucs, val_aucs, label)
        plt.savefig(f'{plotdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()
        
        ## Save
        checkpoint = {'model': cnn_model, 'state_dict': cnn_model.state_dict()}
        torch.save(checkpoint, modeldir + f'/{label}_checkpoint_rw_' + args['reweight_param']['mode'] + '.pth')

        ### Plot contours
        if args['plot_param']['contours_on']:
            targetdir = f'./figs/eid/{args["config"]}/train/2D_contours/cnn/'; os.makedirs(targetdir, exist_ok=True)
            plots.plot_decision_contour(lambda x : cnn_model.softpredict(x),
                X = X_trn, y = Y_trn, labels = data.VARS, targetdir = targetdir, matrix = 'torch')

    ### CLASSIFIER
    if args['cdmx_param']['active']:

        label = args['cdmx_param']['label']

        print(f'\nTraining {label} classifier ...')
        cmdx_model = cnn.CNN_DMAX(D = X_trn.shape[1], C=2, nchannels=DIM[1], nrows=DIM[2], ncols=DIM[3], \
            dropout_cnn = args['cdmx_param']['dropout_cnn'], neurons = args['cdmx_param']['neurons'], \
            num_units = args['cdmx_param']['num_units'], dropout = args['cdmx_param']['dropout'])

        cmdx_model, losses, trn_aucs, val_aucs = dopt.dualtrain(model = cmdx_model, X1_trn = X_trn_2D, X2_trn = X_trn, \
            Y_trn = Y_trn, X1_val = X_val_2D, X2_val = X_val, Y_val = Y_val, trn_weights = trn_weights, param = args['cdmx_param'])
        
        # Plot evolution
        plotdir = f'./figs/eid/{args["config"]}/train/'; os.makedirs(plotdir, exist_ok=True)
        fig,ax  = plots.plot_train_evolution(losses, trn_aucs, val_aucs, label)
        plt.savefig(f'{plotdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()
        
        ## Save
        checkpoint = {'model': cmdx_model, 'state_dict': cmdx_model.state_dict()}
        torch.save(checkpoint, modeldir + f'/{label}_checkpoint_rw_' + args['reweight_param']['mode'] + '.pth')

        ### Plot contours
        #if args['plot_param']['contours_on']:
        #    targetdir = f'./figs/eid/{args["config"]}/train/2D_contours/{label}/'; os.makedirs(targetdir, exist_ok=True)
        #    plots.plot_decision_contour(lambda x : cdmx_model.softpredict(x1,x2),
        #        X = X_trn, y = Y_trn, labels = data.VARS, targetdir = targetdir, matrix = 'torch')


    ### CLASSIFIER
    if args['xgb_param']['active']:

        label = args['xgb_param']['label']

        if args['xgb_param']['tree_method'] == 'auto':
            args['xgb_param'].update({'tree_method' : 'gpu_hist' if torch.cuda.is_available() else 'hist'})

        print(f'\nTraining {label} classifier ...')

        print(f'before extension: {data.trn.x.shape}')

        # Extended data
        x_trn_    = data.trn.x #np.c_[data.trn.x, data_tensor['trn'][:,0,:,:].reshape(len(data.trn.x), -1)]
        x_val_    = data.val.x #np.c_[data.val.x, data_tensor['val'][:,0,:,:].reshape(len(data.val.x), -1)]
        
        print(f'after extension: {x_trn_.shape}')

        dtrain    = xgboost.DMatrix(data = x_trn_, label = data.trn.y, weight = trn_weights)
        dtest     = xgboost.DMatrix(data = x_val_, label = data.val.y)

        evallist  = [(dtrain, 'train'), (dtest, 'eval')]
        results   = dict()
        xgb_model = xgboost.train(params = args['xgb_param'], dtrain = dtrain,
            num_boost_round = args['xgb_param']['num_boost_round'], evals = evallist, evals_result = results, verbose_eval = True)
        
        ## Save
        pickle.dump(xgb_model, open(modeldir + f'/{label}_model_rw_' + args['reweight_param']['mode'] + '.dat', 'wb'))

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
        plt.xlabel('f-score (gain)')

        targetdir = f'./figs/eid/{args["config"]}/train'; os.makedirs(targetdir, exist_ok = True)
        plt.savefig(f'{targetdir}/{label}_importance.pdf', bbox_inches='tight'); plt.close()

        ## Plot decision tree
        #xgboost.plot_tree(xgb_model, num_trees=2)
        #plt.savefig('{}/xgb_tree.pdf'.format(targetdir), bbox_inches='tight'); plt.close()        
        
        ### Plot contours
        if args['plot_param']['contours_on']:
            targetdir = f'./figs/eid/{args["config"]}/train/2D_contours/{label}/'; os.makedirs(targetdir, exist_ok = True)
            plots.plot_decision_contour(lambda x : xgb_model.predict(x),
                X = X_trn, y = Y_trn, labels = data.VARS, targetdir = targetdir, matrix = 'xgboost')

    ### CLASSIFIER
    if args['mlgr_param']['active']:

        label = args['mlgr_param']['label']
        
        print(f'\nTraining {label} classifier ...')
        mlgr_model = mlgr.MLGR(D = X_trn.shape[1], C=2)
        mlgr_model, losses, trn_aucs, val_aucs = dopt.train(model = mlgr_model, X_trn = X_trn, Y_trn = Y_trn, X_val = X_val, Y_val = Y_val,
            trn_weights = trn_weights, param = args['mlgr_param'])
        
        # Plot evolution
        plotdir  = f'./figs/eid/{args["config"]}/train/'; os.makedirs(plotdir, exist_ok=True)
        fig,ax = plots.plot_train_evolution(losses, trn_aucs, val_aucs, label)
        plt.savefig(f'{plotdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()
        
        ## Save
        checkpoint = {'model': mlgr_model, 'state_dict': mlgr_model.state_dict()}
        torch.save(checkpoint, modeldir + f'/{label}_checkpoint_rw_' + args['reweight_param']['mode'] + '.pth')

        ### Plot contours
        if args['plot_param']['contours_on']:
            targetdir = f'./figs/eid/{args["config"]}/train/2D_contours/{label}/'; os.makedirs(targetdir, exist_ok=True)
            plots.plot_decision_contour(lambda x : mlgr_model.softpredict(x),
                X = X_trn, y = Y_trn, labels = data.VARS, targetdir = targetdir, matrix = 'torch')

    ### CLASSIFIER
    '''
    if args['xtx_param']['active']:

        label = args['xtx_param']['label']

        pt_edges  = args['plot_param']['pt_edges']
        eta_edges = args['plot_param']['eta_edges'] 
        
        for i in range(len(pt_edges) - 1):
            for j in range(len(eta_edges) - 1):

                try:

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
                    #xtx_model = mlgr.MLGR(D = X_trn.shape[1], C = 2)
                    xtx_model = maxo.MAXOUT(D = X_trn.shape[1], C = 2, num_units=args['xtx_param']['num_units'], neurons=args['xtx_param']['neurons'], dropout=args['xtx_param']['dropout'])
                    xtx_model, losses, trn_aucs, val_aucs = dopt.train(model = xtx_model,
                        X_trn = X_trn[trn_ind,:], Y_trn = Y_trn[trn_ind],
                        X_val = X_val[val_ind,:], Y_val = Y_val[val_ind], trn_weights = weights, param = args['xtx_param'])

                    # Save
                    checkpoint = {'model': xtx_model, 'state_dict': xtx_model.state_dict()}
                    torch.save(checkpoint, f'{modeldir}/{label}_checkpoint_bin_{i}_{j}.pth')

                except:
                    print('Problem with training *** PT = [{:.3f},{:.3f}], ETA = [{:.3f},{:.3f}] ***'.format(pt_range[0], pt_range[1], eta_range[0], eta_range[1])
    '''
    
    ### CLASSIFIER
    if args['dmax_param']['active']:

        label = args['dmax_param']['label']
        
        print(f'\nTraining {label} classifier ...')
        dmax_model = maxo.MAXOUT(D = X_trn.shape[1], C=2, num_units=args['dmax_param']['num_units'], neurons=args['dmax_param']['neurons'], dropout=args['dmax_param']['dropout'])
        dmax_model, losses, trn_aucs, val_aucs = dopt.train(model = dmax_model, X_trn = X_trn, Y_trn = Y_trn, X_val = X_val, Y_val = Y_val,
            trn_weights = trn_weights, param = args['dmax_param'])

        # Plot evolution
        plotdir  = f'./figs/eid/{args["config"]}/train/'; os.makedirs(plotdir, exist_ok = True)
        fig,ax = plots.plot_train_evolution(losses, trn_aucs, val_aucs, label)
        plt.savefig(f'{plotdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()

        ## Save
        checkpoint = {'model': dmax_model, 'state_dict': dmax_model.state_dict()}
        torch.save(checkpoint, modeldir + f'/{label}_checkpoint_rw_' + args['reweight_param']['mode'] + '.pth')
        
        ### Plot contours
        if args['plot_param']['contours_on']:
            targetdir = f'./figs/eid/{args["config"]}/train/2D_contours/{label}/'; os.makedirs(targetdir, exist_ok = True)
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
            if   dbnf_param['optimizer'] == 'Adam':
                optimizer = adam.Adam(dbnf_model.parameters(), lr = dbnf_param['learning_rate'], weight_decay = dbnf_param['weight_decay'], polyak = dbnf_param['polyak'])
            elif dbnf_param['optimizer'] == 'AdamW':
                optimizer = torch.optim.AdamW(dbnf_model.parameters(), lr = dbnf_param['learning_rate'], weight_decay = dbnf_param['weight_decay'])
            
            sched = scheduler.ReduceLROnPlateau(optimizer, factor = dbnf_param['factor'],
                                          patience = dbnf_param['patience'], cooldown = dbnf_param['cooldown'],
                                          min_lr = dbnf_param['min_lr'], verbose = True,
                                          early_stopping = dbnf_param['early_stopping'],
                                          threshold_mode = 'abs')
            
            print(f'Training density for class = {classid} ...')
            dbnf.train(dbnf_model, optimizer, sched, trn.x, val.x, weights, dbnf_param, modeldir)


if __name__ == '__main__' :

   main()

