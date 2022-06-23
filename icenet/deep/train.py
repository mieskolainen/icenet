# Generic model training wrapper functions [TBD; unify and simplify data structures further]
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

import math
import numpy as np
import torch
import torch_geometric

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
import multiprocessing

# xgboost
import xgboost

# matplotlib
from matplotlib import pyplot as plt

# icenet
from icenet.tools import io
from icenet.tools import aux
from icenet.tools import aux_torch

from icenet.tools import plots
from icenet.tools import prints

from icenet.deep  import dopt
from icenet.deep  import deps
from icenet.algo  import flr
from icenet.deep  import bnaf
from icenet.deep  import mlgr
from icenet.deep  import maxo
from icenet.deep  import dmlp
from icenet.deep  import dbnf
from icenet.deep  import vae

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


def getgenericmodel(conv_type, netparam):
    """
    Wrapper to return different torch models
    """

    if   conv_type == 'lgr':
        model = mlgr.MLGR(**netparam)
    elif conv_type == 'dmlp':
        model = dmlp.DMLP(**netparam)
    elif conv_type == 'deps':
        model = deps.DEPS(**netparam)
    elif conv_type == 'maxo':
        model = maxo.MAXOUT(**netparam)
    elif conv_type == 'cnn':
        model = cnn.CNN(**netparam)
    elif conv_type == 'cnn+maxo':
        model = cnn.CNN_MAXO(**netparam)       
    elif conv_type == 'vae':
        model = vae.VAE(**netparam) 
    else:
        raise Exception(__name__ + f'.getgenericmodel: Unknown network <conv_type> = {conv_type}')

    return model


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
        raise Except(name__ + f'.getgraphmodel: Unknown network <conv_type> = {conv_type}')
    
    return model

def getgenericparam(param, D, num_classes, config={}):
    """
    Construct generic torch network parameters
    """
    netparam = {
        'C'    : int(num_classes),
        'D'    : int(D)
    }

    # Add model hyperparameter keys
    if param['model_param'] is not None:
        for key in param['model_param'].keys():
            netparam[key] = config[key] if key in config.keys() else param['model_param'][key]

    return netparam, param['conv_type']

def getgraphparam(data_trn, num_classes, param, config={}):
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
    if param['model_param'] is not None:
        for key in param['model_param'].keys():
            netparam[key] = config[key] if key in config.keys() else param['model_param'][key]

    return netparam, param['conv_type']

def raytune_main(inputs, train_func=None):
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

    ## Flag raytune on for training functions
    inputs['args']['__raytune_running__'] = True

    # Raytune main setup
    analysis = tune.run(
        partial(train_func, **inputs),
        search_alg          = search_alg,
        local_dir           = f'./tmp/ray/local_dir',
        resources_per_trial = {"cpu": multiprocessing.cpu_count(), "gpu": 1 if torch.cuda.is_available() else 0},
        config              = config,
        num_samples         = num_samples,
        scheduler           = scheduler,
        progress_reporter   = reporter)
    
    # Get the best config
    best_trial = analysis.get_best_trial(metric=metric, mode=mode, scope="last")

    cprint(f'raytune: Best trial config:                {best_trial.config}',              'green')
    cprint(f'raytune: Best trial final validation loss: {best_trial.last_result["loss"]}', 'green')
    cprint(f'raytune: Best trial final validation AUC:  {best_trial.last_result["AUC"]}',  'green')


    # Set the best config, training functions will update the parameters
    inputs['config'] = best_trial.config
    inputs['args']['__raytune_running__'] = False
    
    # Torch graph networks
    if (train_func == train_torch_graph) or (train_func == train_torch_generic):

        # Train
        if   train_func == train_torch_graph:
            best_trained_model = train_torch_graph(**inputs)
        elif train_func == train_torch_generic:
            best_trained_model = train_torch_generic(**inputs)
        else:
            raise Exception(__name__ + f'.raytune_main: Unknown error with input')

    ## XGboost
    elif train_func == train_xgb:
        best_trained_model = train_xgb(**inputs)

    else:
        raise Exception(__name__ + f'raytune_main: Unsupported train_func = {train_func}')

    return best_trained_model


def torch_train_loop(model, train_loader, test_loader, args, param, config={}, save_period=5):
    """
    Main training loop for all torch based models
    """

    losses   = []
    trn_aucs = []
    val_aucs = []
    
    model, device = dopt.model_to_cuda(model=model, device_type=param['device'])

    ### ** Optimization hyperparameters [possibly from Raytune] **
    opt_param = {}
    for key in param['opt_param'].keys():
        opt_param[key]       = config[key] if key in config.keys() else param['opt_param'][key]

    scheduler_param = {}
    for key in param['scheduler_param'].keys():
        scheduler_param[key] = config[key] if key in config.keys() else param['scheduler_param'][key]
    
    # Create optimizer
    if   opt_param['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),  lr=opt_param['learning_rate'], weight_decay=opt_param['weight_decay'])
    elif opt_param['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt_param['learning_rate'], weight_decay=opt_param['weight_decay'])
    else:
        raise Exception(__name__ + f".torch_train_loop: Unknown optimizer <{opt_param['optimizer']}> chosen")
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_param['step_size'], gamma=scheduler_param['gamma'])
    
    cprint(__name__ + f'.torch_train_loop: Number of free model parameters = {aux_torch.count_parameters_torch(model)}', 'yellow')
    
    for epoch in range(opt_param['epochs']):

        loss = dopt.train(model=model, loader=train_loader, optimizer=optimizer, device=device, opt_param=opt_param)

        if (epoch % save_period) == 0:
            train_acc, train_auc       = dopt.test(model=model, loader=train_loader, optimizer=optimizer, device=device)
            validate_acc, validate_auc = dopt.test(model=model, loader=test_loader,  optimizer=optimizer, device=device)
        
        # Push
        losses.append(loss)
        trn_aucs.append(train_auc)
        val_aucs.append(validate_auc)

        print(__name__ + f'.torch_train_loop: Epoch {epoch+1:03d} / {opt_param["epochs"]:03d}, loss: {loss:.4f} | Train: {train_acc:.4f} (acc), {train_auc:.4f} (AUC) | Validate: {validate_acc:.4f} (acc), {validate_auc:.4f} (AUC)')
        scheduler.step()
        
        if args['__raytune_running__']:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

            tune.report(loss = loss, AUC = validate_auc)
        else:
            ## Save
            checkpoint = {'model': model, 'state_dict': model.state_dict()}
            torch.save(checkpoint, args['modeldir'] + f'/{param["label"]}_' + str(epoch) + '.pth')
        
    if not args['__raytune_running__']:

        # Plot evolution
        plotdir  = aux.makedir(f'./figs/{args["rootname"]}/{args["config"]}/train/')
        fig,ax   = plots.plot_train_evolution(losses, trn_aucs, val_aucs, param['label'])
        plt.savefig(f"{plotdir}/{param['label']}_evolution.pdf", bbox_inches='tight'); plt.close()

        return model

    return # No return value for raytune


def train_torch_graph(config={}, data_trn=None, data_val=None, args=None, param=None, save_period=5):
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
    print(__name__ + f'.train_torch_graph: Training {param["label"]} classifier ...')

    # Construct model
    netparam, conv_type = getgraphparam(data_trn=data_trn, num_classes=args['num_classes'], param=param, config=config)
    model               = getgraphmodel(conv_type=conv_type, netparam=netparam)    

    ### ** Optimization hyperparameters [possibly from Raytune] **
    opt_param = {}
    for key in param['opt_param'].keys():
        opt_param[key] = config[key] if key in config.keys() else param['opt_param'][key]

    # Data loaders
    train_loader = torch_geometric.loader.DataLoader(data_trn, batch_size=opt_param['batch_size'], shuffle=True)
    test_loader  = torch_geometric.loader.DataLoader(data_val, batch_size=512, shuffle=False)

    return torch_train_loop(model=model, train_loader=train_loader, test_loader=test_loader, \
                args=args, param=param, config=config)


def train_torch_generic(X_trn=None, Y_trn=None, X_val=None, Y_val=None,
    trn_weights=None, val_weights=None, X_trn_2D=None, X_val_2D=None, args=None, param=None, config={}):
    """
    Train generic neural model [R^d x (2D) -> softmax]
    
    Args:
        See other train_*
    
    Returns:
        trained model
    """
    print(__name__ + f'.train_torch_generic: Training {param["label"]} classifier ...')

    model, train_loader, test_loader = \
        torch_construct(X_trn=X_trn, Y_trn=Y_trn, X_val=X_val, Y_val=Y_val, X_trn_2D=X_trn_2D, X_val_2D=X_val_2D, \
                trn_weights=trn_weights, val_weights=val_weights, param=param, args=args, config=config)
    
    return torch_train_loop(model=model, train_loader=train_loader, test_loader=test_loader, \
                args=args, param=param, config=config)


def torch_construct(X_trn, Y_trn, X_val, Y_val, X_trn_2D, X_val_2D, trn_weights, val_weights, param, args, config={}):
    """
    Torch model and data loader constructor

    Args:
        See other train_*
    
    Returns:
        model, train_loader, test_loader
    """

    ## ------------------------
    ### Construct model
    netparam, conv_type = getgenericparam(config=config, param=param, D=X_trn.shape[-1], num_classes=args['num_classes'])
    model               = getgenericmodel(conv_type=conv_type, netparam=netparam)


    if trn_weights is None: trn_weights = torch.tensor(np.ones(Y_trn.shape[0]), dtype=torch.float)
    if val_weights is None: val_weights = torch.tensor(np.ones(Y_val.shape[0]), dtype=torch.float)

    ### Generators
    if (X_trn_2D is not None) and ('cnn' in conv_type):
        training_set   = dopt.DualDataset(X=X_trn_2D, U=X_trn, Y=Y_trn, W=trn_weights)
        validation_set = dopt.DualDataset(X=X_val_2D, U=X_val, Y=Y_val, W=val_weights)
    else:
        training_set   = dopt.Dataset(X=X_trn, Y=Y_trn, W=trn_weights)
        validation_set = dopt.Dataset(X=X_val, Y=Y_val, W=val_weights)

    ### ** Optimization hyperparameters [possibly from Raytune] **
    opt_param = {}
    for key in param['opt_param'].keys():
        opt_param[key]       = config[key] if key in config.keys() else param['opt_param'][key]

    params = {'batch_size'  : opt_param['batch_size'],
              'shuffle'     : True,
              'num_workers' : param['num_workers'],
              'pin_memory'  : True}

    train_loader = torch.utils.data.DataLoader(training_set,   **params)
    test_loader  = torch.utils.data.DataLoader(validation_set, **params)

    return model, train_loader, test_loader


def train_xgb(config={}, data_trn=None, data_val=None, y_soft=None, args=None, param=None, plot_importance=True):
    """
    Train XGBoost model
    
    Args:
        See other train_*
    
    Returns:
        trained model
    """

    if param['tree_method'] == 'auto':
        param.update({'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist'})

    print(__name__ + f'.train_xgb: Training {param["label"]} classifier ...')

    ### ** Optimization hyperparameters [possibly from Raytune] **
    if config is not {}:
        for key in param.keys():
            param[key] = config[key] if key in config.keys() else param[key]

    ### *********************************
    
    dtrain    = xgboost.DMatrix(data = data_trn.x, label = data_trn.y if y_soft is None else y_soft, weight = data_trn.w)
    deval     = xgboost.DMatrix(data = data_val.x, label = data_val.y, weight = data_val.w)
    
    evallist  = [(dtrain, 'train'), (deval, 'eval')]
    results   = dict()
    print(param)

    model     = xgboost.train(params = param, dtrain = dtrain,
        num_boost_round = param['num_boost_round'], evals = evallist, evals_result = results, verbose_eval = True)

    if args['__raytune_running__']:

        epoch = 0 # Fixed to 0 (no epochs in use)
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            pickle.dump(model, open(path, 'wb'))

        tune.report(loss = results['train']['logloss'][-1], AUC = results['eval']['auc'][-1])

    else:
        ## Save
        filename = args['modeldir'] + f'/{param["label"]}_' + str(0)
        pickle.dump(model, open(filename + '.dat', 'wb'))
        model.save_model(filename + '.json')
        model.dump_model(filename + '.text', dump_format='text')
    
        losses   = results['train']['logloss']
        trn_aucs = results['train']['auc']
        val_aucs = results['eval']['auc']

        # Plot evolution
        plotdir  = aux.makedir(f'./figs/{args["rootname"]}/{args["config"]}/train/')
        fig,ax   = plots.plot_train_evolution(losses, trn_aucs, val_aucs, param["label"])
        plt.savefig(f'{plotdir}/{param["label"]}_evolution.pdf', bbox_inches='tight'); plt.close()

        ## Plot feature importance
        if plot_importance:

            fig,ax = plots.plot_xgb_importance(model=model, dim=data_trn.x.shape[1], tick_label=data_trn.ids)
            targetdir = aux.makedir(f'./figs/{args["rootname"]}/{args["config"]}/train')
            plt.savefig(f'{targetdir}/{param["label"]}_importance.pdf', bbox_inches='tight'); plt.close()

        ## Plot decision tree
        #xgboost.plot_tree(xgb_model, num_trees=2)
        #plt.savefig('{}/xgb_tree.pdf'.format(targetdir), bbox_inches='tight'); plt.close()        
        
        ## Plot contours
        if args['plot_param']['contours']['active']:
            targetdir = aux.makedir(f'./figs/{args["rootname"]}/{args["config"]}/train/2D_contours/{param["label"]}/')
            plots.plot_decision_contour(lambda x : xgb_model.predict(x),
                X = X_trn, y = Y_trn, labels = data.ids, targetdir = targetdir, matrix = 'xgboost')

        return model

    return # No return value for raytune


def train_graph_xgb(config={}, data_trn=None, data_val=None, trn_weights=None, val_weights=None, args=None, param=None):
    """
    Train graph model + xgb hybrid model

    Args:
        See other train_*

    Returns:
        trained model
    """
    if param['xgb']['tree_method'] == 'auto':
        param['xgb'].update({'tree_method' : 'gpu_hist' if torch.cuda.is_available() else 'hist'})

    print(__name__ + f'.train_graph_xgb: Training {param["label"]} classifier ...')

    ### Compute graphnet convolution output
    graph_model = train_torch_graph(data_trn=data_trn, data_val=data_val, args=args, param=param['graph'])
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
    else:
        print(__name__  + f'.train_graph_xgb: Latent z-space dimension = {Z} auto-detected')
    # -------------------------

    ## Evaluate Graph model output
    graph_model.eval() # ! important

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

    print(__name__ + f'.train_graph_xgb: After extension: {x_trn.shape}')

    ## Train xgboost
    dtrain    = xgboost.DMatrix(data = x_trn, label = y_trn, weight = trn_weights)
    dtest     = xgboost.DMatrix(data = x_val, label = y_val, weight = val_weights)

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
    plotdir  = aux.makedir(f'./figs/{args["rootname"]}/{args["config"]}/train/')
    fig,ax   = plots.plot_train_evolution(losses, trn_aucs, val_aucs, param['xgb']['label'])
    plt.savefig(f"{plotdir}/{param['xgb']['label']}_evolution.pdf", bbox_inches='tight'); plt.close()
    
    # -------------------------------------------
    ## Plot feature importance

    # Create feature names
    ids = []
    for i in range(Z):                  # Graph-net latent features
        ids.append(f'gnn[{i}]')
    for i in range(len(data_trn[0].u)): # Xgboost features
        ids.append(f'xgb[{i}]')
    
    fig,ax = plots.plot_xgb_importance(model=model, dim=x_trn.shape[1], tick_label=ids)
    targetdir = aux.makedir(f'./figs/{args["rootname"]}/{args["config"]}/train')
    plt.savefig(f'{targetdir}/{param["label"]}_importance.pdf', bbox_inches='tight'); plt.close()
    
    ## Plot decision tree
    #xgboost.plot_tree(xgb_model, num_trees=2)
    #plt.savefig('{}/xgb_tree.pdf'.format(targetdir), bbox_inches='tight'); plt.close()
    
    ### Plot contours
    if args['plot_param']['contours']['active']:
        targetdir = aux.makedir(f'./figs/{args["rootname"]}/{args["config"]}/train/2D_contours/{label}/')
        plots.plot_decision_contour(lambda x : xgb_model.predict(x),
            X = X_trn, y = Y_trn, labels = data.ids, targetdir = targetdir, matrix = 'xgboost')

    return model


def train_flr(config={}, data_trn=None, args=None, param=None):
    """
    Train factorized likelihood model

    Args:
        See other train_*

    Returns:
        trained model
    """
    print(__name__ + f'.train_flr: Training {param["label"]} classifier ...')

    b_pdfs, s_pdfs, bin_edges = flr.train(X = data_trn.x, y = data_trn.y, weights = data_trn.w, param = param)
    pickle.dump([b_pdfs, s_pdfs, bin_edges],
        open(args['modeldir'] + f'/{param["label"]}_' + str(0) + '_.dat', 'wb'))

    def func_predict(X):
        return flr.predict(X, b_pdfs, s_pdfs, bin_edges)

    ### Plot contours (TOO SLOW!)
    """
    if args['plot_param']['contours']['active']:
        targetdir = aux.makedir(f'./figs/{args["rootname"]}/{args["config"]}/train/2D_contours/{param["label"]}/')
        plots.plot_decision_contour(lambda x : func_predict(x),
            X = data_trn.x, y = data_trn.y, labels = data.ids, targetdir = targetdir, matrix = 'numpy')
    """
    return (b_pdfs, s_pdfs)


def train_flow(config={}, data_trn=None, data_val=None, args=None, param=None):
    """
    Train normalizing flow (BNAF) neural model

    Args:
        See other train_*

    Returns:
        trained model
    """
    
    # Set input dimensions
    param['model_param']['n_dims'] = data_trn.x.shape[1]

    print(__name__ + f'.train_flow: Training {param["label"]} classifier ...')
    
    for classid in range(args['num_classes']):
        param['model'] = 'class_' + str(classid)

        # Load datasets
        trn   = data_trn.classfilter(classid)
        val   = data_val.classfilter(classid)

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
        
        print(__name__ + f'.train_flow: Training density for class = {classid} ...')
        dbnf.train(model=model, optimizer=optimizer, scheduler=sched,
            trn_x=trn.x, val_x=val.x, trn_weights=trn.w, param=param, modeldir=args['modeldir'])
        
    return True


def train_xtx(config={}, X_trn=None, Y_trn=None, X_val=None, Y_val=None, data_kin=None, args=None, param=None):
    """
    Train binned neural model (untested function; TODO add weights)
    
    Args:
        See other train_*
    
    Returns:
        trained model
    """

    label  = param['label']
    var0   = param['binning']['var'][0]
    var1   = param['binning']['var'][1]

    edges0 = param['binning']['edges'][0]
    edges1 = param['binning']['edges'][1]


    for i in range(len(edges0) - 1):
        for j in range(len(edges1) - 1):

            try:
                range0 = [edges0[i], edges0[i+1]]
                range1 = [edges1[j], edges1[j+1]]

                # Indices
                trn_ind = np.logical_and(aux.pick_ind(data_kin.trn.x[:, data_kin.ids.index(var0)], range0),
                                         aux.pick_ind(data_kin.trn.x[:, data_kin.ids.index(var1)], range1))

                val_ind = np.logical_and(aux.pick_ind(data_kin.val.x[:, data_kin.ids.index(var0)], range0),
                                         aux.pick_ind(data_kin.val.x[:, data_kin.ids.index(var1)], range1))

                print(__name__ + f'.train_xtx: --- {var0} = {range0}], {var1} = {range1} ---')


                # Compute weights for this hyperbin (balance class ratios)
                y = Y_trn[trn_ind]
                frac = [np.sum(y == k) / y.shape[0] for k in range(args['num_classes'])]

                print(__name__ + f'.train_xtx: --- frac = [{frac[0]:.3f}, {frac[1]:.3f}] ---')

                # Inverse weights
                weights = np.zeros(y.shape[0])
                for k in range(weights.shape[0]):
                    weights[k] = 1.0 / frac[int(y[k])] / y.shape[0] / args['num_classes']
                
                for c in range(args[num_classes]):
                    print(__name__ + f'.train_xtx: class = {c} | sum(weights) = {np.sum(weights[y == c])}')


                # Set hyperbin label and train
                param['label'] = f'{label}_bin_{i}_{j}'

                model, train_loader, test_loader = \
                    train.torch_construct(X_trn = X_trn[trn_ind,:], Y_trn = Y_trn[trn_ind],
                        X_val = X_val[val_ind,:], Y_val = Y_val[val_ind], X_trn_2D=None, X_val_2D=None, \
                     trn_weights=weights, val_weights=None, param=param['model_param'], args=args)

                model = train.torch_train_loop(model=model, train_loader=train_loader, test_loader=test_loader, \
                            args=args, param=param['model_param'])

            except:
                print(__name__ + f'.train_xtx: Problem in training with bin: {var0} = {range0}], {var1} = {range1}')

    return True
