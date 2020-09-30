# Generic model training wrapper functions
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
#import graphviz
import torch_geometric
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


# Graphnet training function
#
def train_graph(data_trn, data_val, args, param):
    
    num_classes         = 2
    num_node_features   = data_trn[0].x.size(-1)
    num_edge_features   = data_trn[0].edge_attr.size(-1)
    num_global_features = len(data_trn[0].u)
    
    conv_type = param['conv_type']


    if   conv_type == 'GAT':
        model = graph.GATNet(D = num_node_features, C = num_classes, G = num_global_features, task='graph')
    elif conv_type == 'DEC':
        model = graph.DECNet(D = num_node_features, C = num_classes, G = num_global_features, task='graph')
    elif conv_type == 'EC':
        model = graph.ECNet(D = num_node_features, C = num_classes, G = num_global_features, task='graph')    
    elif conv_type == 'SG':
        model = graph.SGNet(D = num_node_features, C = num_classes, G = num_global_features, task='graph')
    elif conv_type == 'SAGE':
        model = graph.SAGENet(D = num_node_features, C = num_classes, G = num_global_features, task='graph')
    elif conv_type == 'NN':
        model = graph.NNNet(D = num_node_features, C = num_classes, E = num_edge_features, G = num_global_features, task='graph')
    elif conv_type == 'GINE':
        model = graph.GINENet(D = num_node_features, C = num_classes, G = num_global_features, task='graph')
    elif conv_type == 'spline':
        model = graph.SplineNet(D = num_node_features, C = num_classes, E = num_edge_features, G = num_global_features, task='graph')
    else:
        raise Except(name__ + f'.graph_train: Unknown network convolution model "conv_type" = {conv_type}')

    # CPU or GPU
    model, device = dopt.model_to_cuda(model=model, device_type=param['device'])

    # Count the number of parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    cprint(__name__ + f'.graph_train: Number of free parameters = {params}', 'yellow')
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=param['learning_rate'], weight_decay=param['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    
    # Data loaders
    train_loader = torch_geometric.data.DataLoader(data_trn, batch_size=param['batch_size'], shuffle=True)
    test_loader  = torch_geometric.data.DataLoader(data_val, batch_size=512, shuffle=False)
    
    for epoch in range(param['epochs']):
        loss               = graph.train(model=model, loader=train_loader, optimizer=optimizer, device=device)
        test_acc, test_auc = graph.test( model=model, loader=test_loader,  optimizer=optimizer, device=device)

        print(f'Epoch {epoch+1:03d}, Loss: {loss:.4f}, Test: {test_acc:.4f} (ACC), {test_auc:.4f} (AUC)')
        scheduler.step()
    
    ## Save
    label = param['label']
    checkpoint = {'model': model, 'state_dict': model.state_dict()}
    torch.save(checkpoint, args['modeldir'] + f'/{label}_checkpoint_rw_' + args['reweight_param']['mode'] + '.pth')


def train_dmax(X_trn, Y_trn, X_val, Y_val, trn_weights, args, param):

    label = param['label']
    
    print(f'\nTraining {label} classifier ...')
    model = maxo.MAXOUT(D = X_trn.shape[1], C=2, num_units=param['num_units'], neurons=param['neurons'], dropout=param['dropout'])
    model, losses, trn_aucs, val_aucs = dopt.train(model = model, X_trn = X_trn, Y_trn = Y_trn, X_val = X_val, Y_val = Y_val,
        trn_weights = trn_weights, param = param)

    # Plot evolution
    plotdir  = f'./figs/eid/{args["config"]}/train/'; os.makedirs(plotdir, exist_ok = True)
    fig,ax = plots.plot_train_evolution(losses, trn_aucs, val_aucs, label)
    plt.savefig(f'{plotdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()

    ## Save
    checkpoint = {'model': model, 'state_dict': model.state_dict()}
    torch.save(checkpoint, args['modeldir'] + f'/{label}_checkpoint_rw_' + args['reweight_param']['mode'] + '.pth')
    
    ### Plot contours
    if args['plot_param']['contours_on']:
        targetdir = f'./figs/eid/{args["config"]}/train/2D_contours/{label}/'; os.makedirs(targetdir, exist_ok = True)
        plots.plot_decision_contour(lambda x : model.softpredict(x),
            X = X_trn, y = Y_trn, labels = data.VARS, targetdir = targetdir, matrix = 'torch')


def train_flr(data, trn_weights, args, param):

    label = param['label']

    print(f'\nTraining {label} classifier ...')
    b_pdfs, s_pdfs, bin_edges = flr.train(X = data.trn.x, y = data.trn.y, weights = trn_weights, param = param)
    pickle.dump([b_pdfs, s_pdfs, bin_edges],
        open(args['modeldir'] + f'/{label}_model_rw_' + args['reweight_param']['mode'] + '.dat', 'wb'))

    def func_predict(X):
        return flr.predict(X, b_pdfs, s_pdfs, bin_edges)

    ### Plot contours (TOO SLOW!)
    """
    if args['plot_param']['contours_on']:
        targetdir = f'./figs/eid/{args["config"]}/train/2D_contours/{label}/'; os.makedirs(targetdir, exist_ok = True)
        plots.plot_decision_contour(lambda x : func_predict(x),
            X = data.trn.x, y = data.trn.y, labels = data.VARS, targetdir = targetdir, matrix = 'numpy')
    """


def train_cdmx(data_tensor, Y_trn, Y_val, trn_weights, args, param):

    '''
    label = args['cdmx_param']['label']

    print(f'\nTraining {label} classifier ...')
    cmdx_model = cnn.CNN_DMAX(D = X_trn.shape[1], C=2, nchannels=DIM[1], nrows=DIM[2], ncols=DIM[3], \
        dropout_cnn = param['dropout_cnn'], neurons = param['neurons'], \
        num_units = param['num_units'], dropout = param['dropout'])

    # -------------------------------------------------------------------------------
    # Into torch format

    X_trn_2D = torch.tensor(data_tensor['trn'], dtype=torch.float)
    X_val_2D = torch.tensor(data_tensor['val'], dtype=torch.float)
    DIM      = X_trn_2D.shape
    # -------------------------------------------------------------------------------

    cmdx_model, losses, trn_aucs, val_aucs = dopt.dualtrain(model = cmdx_model, X1_trn = X_trn_2D, X2_trn = X_trn, \
        Y_trn = Y_trn, X1_val = X_val_2D, X2_val = X_val, Y_val = Y_val, trn_weights = trn_weights, param = param)
    
    # Plot evolution
    plotdir = f'./figs/eid/{args["config"]}/train/'; os.makedirs(plotdir, exist_ok=True)
    fig,ax  = plots.plot_train_evolution(losses, trn_aucs, val_aucs, label)
    plt.savefig(f'{plotdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()
    
    ## Save
    checkpoint = {'model': model, 'state_dict': model.state_dict()}
    torch.save(checkpoint, args['modeldir'] + f'/{label}_checkpoint_rw_' + args['reweight_param']['mode'] + '.pth')
    '''

    ### Plot contours
    #if args['plot_param']['contours_on']:
    #    targetdir = f'./figs/eid/{args["config"]}/train/2D_contours/{label}/'; os.makedirs(targetdir, exist_ok=True)
    #    plots.plot_decision_contour(lambda x : cdmx_model.softpredict(x1,x2),
    #        X = X_trn, y = Y_trn, labels = data.VARS, targetdir = targetdir, matrix = 'torch')


def train_cnn(data_tensor, Y_trn, Y_val, trn_weights, args, param):

    label = param['label']

    # -------------------------------------------------------------------------------
    # Into torch format

    X_trn_2D = torch.tensor(data_tensor['trn'], dtype=torch.float)
    X_val_2D = torch.tensor(data_tensor['val'], dtype=torch.float)
    DIM      = X_trn_2D.shape
    # -------------------------------------------------------------------------------

    print(f'\nTraining {label} classifier ...')
    model = cnn.CNN(C=2, nchannels=DIM[1], nrows=DIM[2], ncols=DIM[3], \
        dropout_cnn = param['dropout_cnn'], dropout_mlp = param['dropout_mlp'], mlp_dim = param['mlp_dim'])

    model, losses, trn_aucs, val_aucs = \
        dopt.train(model = model, X_trn = X_trn_2D, Y_trn = Y_trn, X_val = X_val_2D, Y_val = Y_val,
                    trn_weights = trn_weights, param = param)
    
    # Plot evolution
    plotdir = f'./figs/eid/{args["config"]}/train/'; os.makedirs(plotdir, exist_ok=True)
    fig,ax  = plots.plot_train_evolution(losses, trn_aucs, val_aucs, label)
    plt.savefig(f'{plotdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()
    
    ## Save
    checkpoint = {'model': model, 'state_dict': model.state_dict()}
    torch.save(checkpoint, args['modeldir'] + f'/{label}_checkpoint_rw_' + args['reweight_param']['mode'] + '.pth')

    ### Plot contours
    if args['plot_param']['contours_on']:
        targetdir = f'./figs/eid/{args["config"]}/train/2D_contours/{label}/'; os.makedirs(targetdir, exist_ok=True)
        plots.plot_decision_contour(lambda x : model.softpredict(x),
            X = X_trn, y = Y_trn, labels = data.VARS, targetdir = targetdir, matrix = 'torch')


def train_lgr(X_trn, Y_trn, X_val, Y_val, trn_weights, args, param):

    label = param['label']
    
    print(f'\nTraining {label} classifier ...')
    model = mlgr.MLGR(D = X_trn.shape[1], C=2)
    model, losses, trn_aucs, val_aucs = dopt.train(model = model, X_trn = X_trn, Y_trn = Y_trn, X_val = X_val, Y_val = Y_val,
        trn_weights = trn_weights, param = param)
    
    # Plot evolution
    plotdir = f'./figs/eid/{args["config"]}/train/'; os.makedirs(plotdir, exist_ok=True)
    fig,ax  = plots.plot_train_evolution(losses, trn_aucs, val_aucs, label)
    plt.savefig(f'{plotdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()
    
    ## Save
    checkpoint = {'model': model, 'state_dict': model.state_dict()}
    torch.save(checkpoint, args['modeldir'] + f'/{label}_checkpoint_rw_' + args['reweight_param']['mode'] + '.pth')

    ### Plot contours
    if args['plot_param']['contours_on']:
        targetdir = f'./figs/eid/{args["config"]}/train/2D_contours/{label}/'; os.makedirs(targetdir, exist_ok=True)
        plots.plot_decision_contour(lambda x : model.softpredict(x),
            X = X_trn, y = Y_trn, labels = data.VARS, targetdir = targetdir, matrix = 'torch')


def train_xgb(data, trn_weights, args, param):

    label = param['label']

    if param['tree_method'] == 'auto':
        param.update({'tree_method' : 'gpu_hist' if torch.cuda.is_available() else 'hist'})

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
    model     = xgboost.train(params = param, dtrain = dtrain,
        num_boost_round = param['num_boost_round'], evals = evallist, evals_result = results, verbose_eval = True)
    
    ## Save
    pickle.dump(model, open(args['modeldir'] + f'/{label}_model_rw_' + args['reweight_param']['mode'] + '.dat', 'wb'))

    losses   = results['train']['logloss']
    trn_aucs = results['train']['auc']
    val_aucs = results['eval']['auc']

    # Plot evolution
    plotdir  = f'./figs/eid/{args["config"]}/train/'; os.makedirs(plotdir, exist_ok = True)
    fig,ax   = plots.plot_train_evolution(losses, trn_aucs, val_aucs, label)
    plt.savefig(f'{plotdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()


    ## Plot feature importance (xgb does Not return it for all of them)
    fscores  = model.get_score(importance_type='gain')
    print(fscores)

    D  = data.trn.x.shape[1]
    xx = np.arange(D)
    yy = np.zeros(D)

    for i in range(D):
        try:
            yy[i] = fscores['f' + str(i)]
        except:
            yy[i] = 0.0

    fig  = plt.figure(figsize=(12,8))
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


def train_xtx(X_trn, Y_trn, X_val, Y_val, data_kin, args, param):
    
    label     = param['label']
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

                print('*** PT = [{:.3f},{:.3f}], ETA = [{:.3f},{:.3f}] ***'.
                    format(pt_range[0], pt_range[1], eta_range[0], eta_range[1]))

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
                model = maxo.MAXOUT(D = X_trn.shape[1], C = 2, num_units=param['num_units'], \
                    neurons=param['neurons'], dropout=args['xtx_param']['dropout'])

                model, losses, trn_aucs, val_aucs = dopt.train(model = model,
                    X_trn = X_trn[trn_ind,:], Y_trn = Y_trn[trn_ind],
                    X_val = X_val[val_ind,:], Y_val = Y_val[val_ind], trn_weights = weights, param = param)

                # Save
                checkpoint = {'model': model, 'state_dict': model.state_dict()}
                torch.save(checkpoint, f'{args["modeldir"]}/{label}_checkpoint_bin_{i}_{j}.pth')

            except:
                print('Problem with training *** PT = [{:.3f},{:.3f}], ETA = [{:.3f},{:.3f}] ***'.
                    format(pt_range[0], pt_range[1], eta_range[0], eta_range[1]))


def train_flow(data, trn_weights, args, param):

    label = param['label']
    param['n_dims'] = data.trn.x.shape[1]

    print(f'\nTraining {label} classifier ...')

    for classid in [0,1] :
        param['model'] = 'class_' + str(classid) + '_rw_' + args['reweight_param']['mode']

        # Load datasets
        trn = data.trn.classfilter(classid)
        val = data.val.classfilter(classid)

        # Load re-weighting weights
        weights = trn_weights[data.trn.y == classid]

        # Create model
        model = dbnf.create_model(param, verbose = True)

        # Create optimizer & scheduler
        if   param['optimizer'] == 'Adam':
            optimizer = adam.Adam(model.parameters(), lr = param['learning_rate'], \
                weight_decay = param['weight_decay'], polyak = param['polyak'])

        elif param['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr = param['learning_rate'], \
                weight_decay = param['weight_decay'])
        
        sched = scheduler.ReduceLROnPlateau(optimizer,
                                      factor   = param['factor'],
                                      patience = param['patience'],
                                      cooldown = param['cooldown'],
                                      min_lr   = param['min_lr'],
                                      verbose  = True,
                                      early_stopping = param['early_stopping'],
                                      threshold_mode = 'abs')
        
        print(f'Training density for class = {classid} ...')
        dbnf.train(model, optimizer, sched, trn.x, val.x, weights, param, args['modeldir'])

