# Generic model training wrapper functions [TBD; unify and simplify data structures]
#
# Mikael Mieskolainen, 2021
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
from icenet.deep  import dmlp

from icenet.deep  import maxo
from icenet.deep  import cnn
from icenet.deep  import graph


from icenet.optim import adam
from icenet.optim import adamax
from icenet.optim import scheduler

# iceid
from iceid import common
from iceid import graphio


from termcolor import colored, cprint


# Raytuning
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial


def raytune_main(inputs, gpus_per_trial=1, train_func=None):
    """
    Raytune mainloop
    """
    args  = inputs['args']
    param = inputs['param']

    ### General raytune parameters
    num_samples    = args['raytune_param']['num_samples']
    max_num_epochs = args['raytune_param']['max_num_epochs']


    ### Construct hyperparameter config (setup) from yaml
    steer  = param['raytune']
    config = {}

    for key in args['raytune_setup'][steer]['param']:

        rtp   = args['raytune_setup'][steer]['param'][key]['type']
        value = args['raytune_setup'][steer]['param'][key]['value']

        # Random integer
        if   rtp == 'tune.randint':
            config[key] = tune.randint(value[0], value[1])

        # Fixed array
        elif rtp == 'tune.choice':
            config[key] = tune.choice(value)

        # Log-uniform sampling
        elif rtp == 'tune.loguniform':
            config[key] = tune.loguniform(value[0], value[1])

        # Uniform sampling
        elif rtp == 'tune.uniform':
            config[key] = tune.uniform(value[0], value[1])

        else:
            raise Exception(__name__ + f'.raytune_main: Unknown raytune parameter type = {rtp}')

    # --------------------------------------------------------------------

    # Raytune basic metrics
    reporter   = CLIReporter(metric_columns = ["loss", "AUC", "training_iteration"])

    # Raytune search algorithm
    metric     = args['raytune_setup'][steer]['search_metric']['metric']
    mode       = args['raytune_setup'][steer]['search_metric']['mode']


    # Hyperopt Bayesian / 
    search_alg = HyperOptSearch(metric=metric, mode=mode)

    # Raytune scheduler
    scheduler = ASHAScheduler(
        metric = metric,
        mode   = mode,
        max_t  = max_num_epochs,
        grace_period     = 1,
        reduction_factor = 2)

    # Raytune main setup
    analysis = tune.run(
        partial(train_func, **inputs),
        search_alg          = search_alg,
        resources_per_trial = {"cpu": 8, "gpu": gpus_per_trial},
        config              = config,
        num_samples         = num_samples,
        scheduler           = scheduler,
        progress_reporter   = reporter)

    # Get the best config
    best_trial = analysis.get_best_trial(metric=metric, mode=mode, scope="last")

    cprint(f'raytune: Best trial config:                {best_trial.config}', 'green')
    cprint(f'raytune: Best trial final validation loss: {best_trial.last_result["loss"]}', 'green')
    cprint(f'raytune: Best trial final validation AUC:  {best_trial.last_result["AUC"]}', 'green')


    # GRAPH NETWORKS
    if train_func == train_graph:

        ### Load the best model from raytune folder
        bestparam, conv_type = getgraphparam(config=best_trial.config, **inputs)
        best_trained_model   = getgraphmodel(conv_type=conv_type, netparam=bestparam)

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
            if gpus_per_trial > 1:
                best_trained_model = nn.DataParallel(best_trained_model)
        best_trained_model.to(device)

        best_checkpoint_dir          = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
        best_trained_model.load_state_dict(model_state)

        ### Finally save it under model folder
        checkpoint = {'model': best_trained_model, 'state_dict': best_trained_model.state_dict()}
        torch.save(checkpoint, args['modeldir'] + f'/{param["label"]}' + '_raytune.pth')

    elif train_func == train_xgb:

        # Get the best config
        config = best_trial.config

        # Load the optimal values for the given hyperparameters
        optimal_param = {}
        for key in param.keys():
            optimal_param[key] = config[key] if key in config.keys() else param[key]
        
        print('Best parameters:')
        print(optimal_param)
        
        # ** Final train with the optimal parameters **
        inputs['param'] = optimal_param
        best_trained_model = train_xgb(**inputs)
        
    else:
        raise Exception(__name__ + f'raytune_main: Unknown train_func = {train_func}')

    return best_trained_model


def getgraphmodel(conv_type, netparam):
    """
    Wrapper to return different graph networks
    """

    if   conv_type == 'GAT':
        model = graph.GATNet(**netparam)
    elif conv_type == 'DEC':
        model = graph.DECNet(**netparam)
    elif conv_type == 'PAN':
        model = graph.PANNet(**netparam)
    elif conv_type == 'EC':
        model = graph.ECNet(**netparam)
    elif conv_type == 'SUP':
        model = graph.SUPNet(**netparam)
    elif conv_type == 'SG':
        model = graph.SGNet(**netparam)
    elif conv_type == 'SAGE':
        model = graph.SAGENet(**netparam)
    elif conv_type == 'NN':
        model = graph.NNNet(**netparam)
    elif conv_type == 'GINE':
        model = graph.GINENet(**netparam)
    elif conv_type == 'spline':
        model = graph.SplineNet(**netparam)
    else:
        raise Except(name__ + f'.getgraphmodel: Unknown network convolution model "conv_type" = {conv_type}')
    
    return model


def getgraphparam(config, data_trn, data_val, args, param, num_classes=2):
    """
    Construct graph network parameters
    """
    num_node_features   = data_trn[0].x.size(-1)
    num_edge_features   = data_trn[0].edge_attr.size(-1)
    num_global_features = len(data_trn[0].u)

    netparam = {
        'C'    : int(num_classes),
        'D'    : int(num_node_features),
        'E'    : int(num_edge_features),
        'G'    : int(num_global_features),
        'task' : 'graph'
    }

    # Add model hyperparameter keys
    for key in param['model_param'].keys():
        netparam[key] = config[key] if key in config.keys() else param['model_param'][key]

    return netparam, param['conv_type']


def train_graph(config={}, data_trn=None, data_val=None, args=None, param=None, num_classes=2):
    """
    Train graph neural networks
    
    Args:
        config:   raytune parameter dict
        data_trn: training data
        data_val: validation data
        args:     arg parameters dict
        param:    model parameters dict

    Returns:
        trained model
    """

    ### ** Optimization hyperparameters **
    opt_param = {}
    for key in param['opt_param'].keys():
        opt_param[key]       = config[key] if key in config.keys() else param['opt_param'][key]

    scheduler_param = {}
    for key in param['scheduler_param'].keys():
        scheduler_param[key] = config[key] if key in config.keys() else param['scheduler_param'][key]


    ## ------------------------
    ### Construct model
    netparam, conv_type = getgraphparam(config, data_trn, data_val, args, param)
    model               = getgraphmodel(conv_type, netparam)    

    # CPU or GPU
    model, device = dopt.model_to_cuda(model=model, device_type=param['device'])

    # Count the number of parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    cprint(__name__ + f'.graph_train: Number of free parameters = {params}', 'yellow')
    

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_param['learning_rate'], weight_decay=opt_param['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_param['step_size'], gamma=scheduler_param['gamma'])
    
    # Data loaders
    train_loader = torch_geometric.loader.DataLoader(data_trn, batch_size=opt_param['batch_size'], shuffle=True)
    test_loader  = torch_geometric.loader.DataLoader(data_val, batch_size=512, shuffle=False)
    

    for epoch in range(opt_param['epochs']):

        loss                       = graph.train(model=model, loader=train_loader, optimizer=optimizer, device=device)
        validate_acc, validate_AUC = graph.test( model=model, loader=test_loader,  optimizer=optimizer, device=device)
        
        print(f'Epoch {epoch+1:03d}, train loss: {loss:.4f} | validate: {validate_acc:.4f} (acc), {validate_AUC:.4f} (AUC)')
        scheduler.step()

        # Raytune on
        if len(config) != 0:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

            tune.report(loss = loss, AUC = validate_AUC)
        else:
            ## Save
            checkpoint = {'model': model, 'state_dict': model.state_dict()}
            torch.save(checkpoint, args['modeldir'] + f'/{param["label"]}_' + str(epoch) + '.pth')

    return model


def train_graph_xgb(config={}, data_trn=None, data_val=None, trn_weights=None, args=None, param=None, num_classes=2):
    """
    Train graph model + xgb hybrid model

    Args:
        See other train_*

    Returns:
        trained model
    """

    label = param['label']

    if param['xgb']['tree_method'] == 'auto':
        param['xgb'].update({'tree_method' : 'gpu_hist' if torch.cuda.is_available() else 'hist'})

    print(f'\nTraining {label} classifier ...')

    ### Compute graphnet convolution output
    graph_model = train_graph(data_trn=data_trn, data_val=data_val, args=args, param=param['graph'])
    graph_model = graph_model.to('cpu')    

    ### Find out the latent space dimension -------
    Z = 0
    for i in range(len(data_trn)):

        # Use try-except while we find an event with proper graph information
        try:
            xtest = graph_model.forward(data=data_trn[i], conv_only=True).detach().numpy()
            Z = xtest.shape[-1]  # Find out dimension of the convolution output
            break
        except:
            continue
    
    if Z == 0:
        raise Exception(__name__ + '.train_graph_xgb: Could not auto-detect latent space dimension')
    # -------------------------
    

    ## Evaluate Graph model output
    
    x_trn = np.zeros((len(data_trn), Z + len(data_trn[0].u)))
    x_val = np.zeros((len(data_val), Z + len(data_val[0].u)))

    y_trn = np.zeros(len(data_trn))
    y_val = np.zeros(len(data_val))

    for i in range(x_trn.shape[0]):

        xconv = graph_model.forward(data=data_trn[i], conv_only=True).detach().numpy()
        x_trn[i,:] = np.c_[xconv, [data_trn[i].u.numpy()]]
        y_trn[i]   = data_trn[i].y.numpy()

    for i in range(x_val.shape[0]):
        
        xconv = graph_model.forward(data=data_val[i], conv_only=True).detach().numpy()
        x_val[i,:] = np.c_[xconv, [data_val[i].u.numpy()]]
        y_val[i]   = data_val[i].y.numpy()


    print(f'after extension: {x_trn.shape}')

    dtrain    = xgboost.DMatrix(data = x_trn, label = y_trn, weight = trn_weights)
    dtest     = xgboost.DMatrix(data = x_val, label = y_val)

    evallist  = [(dtrain, 'train'), (dtest, 'eval')]
    results   = dict()
    model     = xgboost.train(params = param['xgb'], dtrain = dtrain,
        num_boost_round = param['xgb']['num_boost_round'], evals = evallist, evals_result = results, verbose_eval = True)

    ## Save
    pickle.dump(model, open(args['modeldir'] + f"/{param['xgb']['label']}_" + str(0) + '.dat', 'wb'))

    losses   = results['train']['logloss']
    trn_aucs = results['train']['auc']
    val_aucs = results['eval']['auc']


    # Plot evolution
    plotdir  = f'./figs/{args["rootname"]}/{args["config"]}/train/'; os.makedirs(plotdir, exist_ok = True)
    fig,ax   = plots.plot_train_evolution(losses, trn_aucs, val_aucs, param['xgb']['label'])
    plt.savefig(f"{plotdir}/{param['xgb']['label']}_evolution.pdf", bbox_inches='tight'); plt.close()


    # ------------------------------------------------------------------------------------
    ## Plot feature importance (xgb does Not return it for all of them)
    fscores  = model.get_score(importance_type='gain')
    print(fscores)

    D  = x_trn.shape[1]
    xx = np.arange(D)
    yy = np.zeros(D)

    for i in range(D):
        try:
            yy[i] = fscores['f' + str(i)]
        except:
            yy[i] = 0.0


    fig  = plt.figure(figsize=(12,8))
    bars = plt.barh(xx, yy, align='center', height=0.5)
    plt.xlabel('f-score (gain)')

    targetdir = f'./figs/{args["rootname"]}/{args["config"]}/train'; os.makedirs(targetdir, exist_ok = True)
    plt.savefig(f'{targetdir}/{label}_importance.pdf', bbox_inches='tight'); plt.close()
    
    ## Plot decision tree
    #xgboost.plot_tree(xgb_model, num_trees=2)
    #plt.savefig('{}/xgb_tree.pdf'.format(targetdir), bbox_inches='tight'); plt.close()
    
    ### Plot contours
    if args['plot_param']['contours_on']:
        targetdir = f'./figs/{args["rootname"]}/{args["config"]}/train/2D_contours/{label}/'; os.makedirs(targetdir, exist_ok = True)
        plots.plot_decision_contour(lambda x : xgb_model.predict(x),
            X = X_trn, y = Y_trn, labels = data.ids, targetdir = targetdir, matrix = 'xgboost')

    return model


def train_dmax(config={}, X_trn=None, Y_trn=None, X_val=None, Y_val=None, trn_weights=None, args=None, param=None, num_classes=2):
    """
    Train dmax neural model

    Args:
        See other train_*

    Returns:
        trained model
    """

    label = param['label']
    
    print(f'\nTraining {label} classifier ...')
    model = maxo.MAXOUT(D = X_trn.shape[1], C=num_classes, **param['model_param'])
    model, losses, trn_aucs, val_aucs = dopt.train(model = model, X_trn = X_trn, Y_trn = Y_trn, X_val = X_val, Y_val = Y_val,
        trn_weights = trn_weights, param = param, modeldir=args['modeldir'])


    # Plot evolution
    plotdir  = f'./figs/{args["rootname"]}/{args["config"]}/train/'; os.makedirs(plotdir, exist_ok = True)
    fig,ax = plots.plot_train_evolution(losses, trn_aucs, val_aucs, label)
    plt.savefig(f'{plotdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()

    ### Plot contours
    if args['plot_param']['contours_on']:
        targetdir = f'./figs/{args["rootname"]}/{args["config"]}/train/2D_contours/{label}/'; os.makedirs(targetdir, exist_ok = True)
        plots.plot_decision_contour(lambda x : model.softpredict(x),
            X = X_trn, y = Y_trn, labels = data.ids, targetdir = targetdir, matrix = 'torch')

    return model


def train_flr(config={}, data=None, trn_weights=None, args=None, param=None):
    """
    Train factorized likelihood model

    Args:
        See other train_*

    Returns:
        trained model
    """

    label = param['label']

    print(f'\nTraining {label} classifier ...')
    b_pdfs, s_pdfs, bin_edges = flr.train(X = data.trn.x, y = data.trn.y, weights = trn_weights, param = param)
    pickle.dump([b_pdfs, s_pdfs, bin_edges],
        open(args['modeldir'] + f'/{label}_' + str(0) + '_.dat', 'wb'))

    def func_predict(X):
        return flr.predict(X, b_pdfs, s_pdfs, bin_edges)

    ### Plot contours (TOO SLOW!)
    """
    if args['plot_param']['contours_on']:
        targetdir = f'./figs/{args["rootname"]}/{args["config"]}/train/2D_contours/{label}/'; os.makedirs(targetdir, exist_ok = True)
        plots.plot_decision_contour(lambda x : func_predict(x),
            X = data.trn.x, y = data.trn.y, labels = data.ids, targetdir = targetdir, matrix = 'numpy')
    """
    return (b_pdfs, s_pdfs)


def train_cdmx(config={}, data_tensor=None, Y_trn=None, Y_val=None, trn_weights=None, args=None, param=None, num_classes=2):
    """
    Train cdmx neural model
    
    Args:
        See other train_*

    Returns:
        trained model
    """

    """
    NOT WORKING CURRENTLY [update code]
    
    label = args['cdmx_param']['label']

    print(f'\nTraining {label} classifier ...')
    cmdx_model = cnn.CNN_DMAX(D = X_trn.shape[1], C=num_classes, nchannels=DIM[1], nrows=DIM[2], ncols=DIM[3], \
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
    plotdir = f'./figs/{args["rootname"]}/{args["config"]}/train/'; os.makedirs(plotdir, exist_ok=True)
    fig,ax  = plots.plot_train_evolution(losses, trn_aucs, val_aucs, label)
    plt.savefig(f'{plotdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()
    
    ## Save
    checkpoint = {'model': model, 'state_dict': model.state_dict()}
    torch.save(checkpoint, args['modeldir'] + f'/{label}_checkpoint' + '.pth')
    """

    ### Plot contours
    #if args['plot_param']['contours_on']:
    #    targetdir = f'./figs/{args["rootname"]}/{args["config"]}/train/2D_contours/{label}/'; os.makedirs(targetdir, exist_ok=True)
    #    plots.plot_decision_contour(lambda x : cdmx_model.softpredict(x1,x2),
    #        X = X_trn, y = Y_trn, labels = data.ids, targetdir = targetdir, matrix = 'torch')

    # return model


def train_cnn(config={}, data=None, data_tensor=None, Y_trn=None, Y_val=None, trn_weights=None, args=None, param=None, num_classes=2):
    """
    Train CNN neural model

    Args:
        See other train_*

    Returns:
        trained model
    """

    label = param['label']

    # -------------------------------------------------------------------------------
    # Into torch format

    X_trn_2D = torch.tensor(data_tensor['trn'], dtype=torch.float)
    X_val_2D = torch.tensor(data_tensor['val'], dtype=torch.float)
    DIM      = X_trn_2D.shape
    
    # Train
    X_trn = {}
    X_trn['x'] = X_trn_2D
    X_trn['u'] = data.trn.x

    # Validation
    X_val = {}
    X_val['x'] = X_val_2D
    X_val['u'] = data.val.x

    # -------------------------------------------------------------------------------

    print(f'\nTraining {label} classifier ...')
    model = cnn.CNN_DMAX(D=data.trn.x.shape[1], C=num_classes, nchannels=DIM[1], nrows=DIM[2], ncols=DIM[3], **param['model_param'])

    model, losses, trn_aucs, val_aucs = \
        dopt.train(model = model, X_trn = X_trn, Y_trn = Y_trn, X_val = X_val, Y_val = Y_val,
                    trn_weights = trn_weights, param = param, modeldir=args['modeldir'])

    # Plot evolution
    plotdir = f'./figs/{args["rootname"]}/{args["config"]}/train/'; os.makedirs(plotdir, exist_ok=True)
    fig,ax  = plots.plot_train_evolution(losses, trn_aucs, val_aucs, label)
    plt.savefig(f'{plotdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()


    ### Plot contours
    if args['plot_param']['contours_on']:
        targetdir = f'./figs/{args["rootname"]}/{args["config"]}/train/2D_contours/{label}/'; os.makedirs(targetdir, exist_ok=True)
        plots.plot_decision_contour(lambda x : model.softpredict(x),
            X = X_trn, y = Y_trn, labels = data.ids, targetdir = targetdir, matrix = 'torch')

    return model


def train_dmlp(config={}, X_trn=None, Y_trn=None, X_val=None, Y_val=None, trn_weights=None, args=None, param=None, num_classes=2):
    """
    Train dmlp neural model

    Args:
        See other train_*

    Returns:
        trained model
    """

    label = param['label']
    
    print(f'\nTraining {label} classifier ...')
    model = dmlp.DMLP(D = X_trn.shape[1], C = num_classes, **param['model_param'])
    model, losses, trn_aucs, val_aucs = dopt.train(model = model, X_trn = X_trn, Y_trn = Y_trn, X_val = X_val, Y_val = Y_val,
        trn_weights = trn_weights, param = param, modeldir=args['modeldir'])

    # Plot evolution
    plotdir = f'./figs/{args["rootname"]}/{args["config"]}/train/'; os.makedirs(plotdir, exist_ok=True)
    fig,ax  = plots.plot_train_evolution(losses, trn_aucs, val_aucs, label)
    plt.savefig(f'{plotdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()
    

    ### Plot contours
    if args['plot_param']['contours_on']:
        targetdir = f'./figs/{args["rootname"]}/{args["config"]}/train/2D_contours/{label}/'; os.makedirs(targetdir, exist_ok=True)
        plots.plot_decision_contour(lambda x : model.softpredict(x),
            X = X_trn, y = Y_trn, labels = data.ids, targetdir = targetdir, matrix = 'torch')

    return model


def train_lgr(config={}, X_trn=None, Y_trn=None, X_val=None, Y_val=None, trn_weights=None, args=None, param=None, num_classes=2):
    """
    Train lgr neural model

    Args:
        See other train_*

    Returns:
        trained model
    """

    label = param['label']
    
    print(f'\nTraining {label} classifier ...')
    model = mlgr.MLGR(D = X_trn.shape[1], C = num_classes)
    model, losses, trn_aucs, val_aucs = dopt.train(model = model, X_trn = X_trn, Y_trn = Y_trn, X_val = X_val, Y_val = Y_val,
        trn_weights = trn_weights, param = param, modeldir=args['modeldir'])

    # Plot evolution
    plotdir = f'./figs/{args["rootname"]}/{args["config"]}/train/'; os.makedirs(plotdir, exist_ok=True)
    fig,ax  = plots.plot_train_evolution(losses, trn_aucs, val_aucs, label)
    plt.savefig(f'{plotdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()

    ### Plot contours
    if args['plot_param']['contours_on']:
        targetdir = f'./figs/{args["rootname"]}/{args["config"]}/train/2D_contours/{label}/'; os.makedirs(targetdir, exist_ok=True)
        plots.plot_decision_contour(lambda x : model.softpredict(x),
            X = X_trn, y = Y_trn, labels = data.ids, targetdir = targetdir, matrix = 'torch')

    return model


def train_xgb(config={}, data=None, y_soft=None, trn_weights=None, args=None, param=None, plot_importance=True):
    """
    Train XGBoost model
    
    Args:
        See other train_*

    Returns:
        trained model
    """

    label = param['label']

    if param['tree_method'] == 'auto':
        param.update({'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist'})

    print(f'\nTraining {label} classifier ...')

    ### ** Hyperparameter optimization **
    if config is not {}:
        for key in param.keys():
            param[key] = config[key] if key in config.keys() else param[key]

    ### *********************************

    # Extended data
    x_trn_    = data.trn.x
    x_val_    = data.val.x


    dtrain    = xgboost.DMatrix(data = x_trn_, label = data.trn.y if y_soft is None else y_soft, weight = trn_weights)
    dtest     = xgboost.DMatrix(data = x_val_, label = data.val.y)


    evallist  = [(dtrain, 'train'), (dtest, 'eval')]
    results   = dict()

    print(param)

    model     = xgboost.train(params = param, dtrain = dtrain,
        num_boost_round = param['num_boost_round'], evals = evallist, evals_result = results, verbose_eval = True)

    # Raytune on
    if len(config) != 0:

        epoch = 0 # FIXED

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            pickle.dump(model, open(path, 'wb'))

        tune.report(loss = results['train']['logloss'][-1], AUC = results['eval']['auc'][-1])

    else:
    
        ## Save
        filename = args['modeldir'] + f'/{label}_' + str(0)
        pickle.dump(model, open(filename + '.dat', 'wb'))
        
        # Save in JSON format
        model.save_model(filename + '.json')
        model.dump_model(filename + '.text', dump_format='text')
    
    losses   = results['train']['logloss']
    trn_aucs = results['train']['auc']
    val_aucs = results['eval']['auc']

    # Plot evolution
    plotdir  = f'./figs/{args["rootname"]}/{args["config"]}/train/'; os.makedirs(plotdir, exist_ok = True)
    fig,ax   = plots.plot_train_evolution(losses, trn_aucs, val_aucs, label)
    plt.savefig(f'{plotdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()


    ## Plot feature importance (xgb does Not return it for all of them)
    if plot_importance:

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
        bars = plt.barh(xx, yy, align='center', height=0.5, tick_label=data.ids)
        plt.xlabel('f-score (gain)')

        targetdir = f'./figs/{args["rootname"]}/{args["config"]}/train'; os.makedirs(targetdir, exist_ok = True)
        plt.savefig(f'{targetdir}/{label}_importance.pdf', bbox_inches='tight'); plt.close()


    ## Plot decision tree
    #xgboost.plot_tree(xgb_model, num_trees=2)
    #plt.savefig('{}/xgb_tree.pdf'.format(targetdir), bbox_inches='tight'); plt.close()        
    
    ### Plot contours
    if args['plot_param']['contours_on']:
        targetdir = f'./figs/{args["rootname"]}/{args["config"]}/train/2D_contours/{label}/'; os.makedirs(targetdir, exist_ok = True)
        plots.plot_decision_contour(lambda x : xgb_model.predict(x),
            X = X_trn, y = Y_trn, labels = data.ids, targetdir = targetdir, matrix = 'xgboost')

    return model


def train_xtx(config={}, X_trn=None, Y_trn=None, X_val=None, Y_val=None, data_kin=None, args=None, param=None, num_classes=2):
    """
    Train xtx neural model
    
    Args:
        See other train_*

    Returns:
        trained model
    """

    label     = param['label']
    pt_edges  = args['plot_param']['pt_edges']
    eta_edges = args['plot_param']['eta_edges'] 

    for i in range(len(pt_edges) - 1):
        for j in range(len(eta_edges) - 1):

            try:
                pt_range  = [ pt_edges[i],  pt_edges[i+1]]
                eta_range = [eta_edges[j], eta_edges[j+1]]

                # Indices
                trn_ind = np.logical_and(aux.pick_ind(data_kin.trn.x[:, data_kin.ids.index('trk_pt')],   pt_range),
                                         aux.pick_ind(data_kin.trn.x[:, data_kin.ids.index('trk_eta')], eta_range))

                val_ind = np.logical_and(aux.pick_ind(data_kin.val.x[:, data_kin.ids.index('trk_pt')],   pt_range),
                                         aux.pick_ind(data_kin.val.x[:, data_kin.ids.index('trk_eta')], eta_range))

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
                
                print(f'weightsum = {np.sum(weights[yy == 0])}')


                # Train
                #xtx_model = mlgr.MLGR(D = X_trn.shape[1], C = 2)
                model = maxo.MAXOUT(D = X_trn.shape[1], C = num_classes, **param['model_param'])

                # Set hyperbin label
                param['label'] = f'{label}_bin_{i}_{j}'

                model, losses, trn_aucs, val_aucs = dopt.train(model = model,
                    X_trn = X_trn[trn_ind,:], Y_trn = Y_trn[trn_ind],
                    X_val = X_val[val_ind,:], Y_val = Y_val[val_ind], trn_weights = weights, param = param, modeldir=args['modeldir'])


            except:
                print('Problem with training *** PT = [{:.3f},{:.3f}], ETA = [{:.3f},{:.3f}] ***'.
                    format(pt_range[0], pt_range[1], eta_range[0], eta_range[1]))

    return True


def train_flow(config={}, data=None, trn_weights=None, args=None, param=None, num_classes=2):
    """
    Train normalizing flow (BNAF) neural model

    Args:
        See other train_*

    Returns:
        trained model
    """

    label = param['label']

    # Set input dimensions
    param['model_param']['n_dims'] = data.trn.x.shape[1]

    print(f'\nTraining {label} classifier ...')
    
    for classid in range(num_classes):
        param['model'] = 'class_' + str(classid)

        # Load datasets
        trn = data.trn.classfilter(classid)
        val = data.val.classfilter(classid)

        # Load re-weighting weights
        weights = trn_weights[data.trn.y == classid]

        # Create model
        model = dbnf.create_model(param=param['model_param'], verbose = True)

        # Create optimizer & scheduler
        if   param['opt_param']['optimizer'] == 'Adam':
            optimizer = adam.Adam(model.parameters(), lr = param['opt_param']['learning_rate'], \
                weight_decay = param['opt_param']['weight_decay'], polyak = param['opt_param']['polyak'])
        
        elif param['opt_param']['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr = param['opt_param']['learning_rate'], \
                weight_decay = param['opt_param']['weight_decay'])
        
        sched = scheduler.ReduceLROnPlateau(optimizer,
                                      factor   = param['scheduler_param']['factor'],
                                      patience = param['scheduler_param']['patience'],
                                      cooldown = param['scheduler_param']['cooldown'],
                                      min_lr   = param['scheduler_param']['min_lr'],
                                      verbose  = True,
                                      early_stopping = param['scheduler_param']['early_stopping'],
                                      threshold_mode = 'abs')
        
        print(f'Training density for class = {classid} ...')
        dbnf.train(model, optimizer, sched, trn.x, val.x, weights, param, args['modeldir'])

    return True
