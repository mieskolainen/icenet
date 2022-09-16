# Generic model training wrapper functions [TBD; unify and simplify data structures further]
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

import math
import numpy as np
import torch
import torch_geometric
from tqdm import tqdm

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

# matplotlib
from matplotlib import pyplot as plt

# icenet
from icenet.tools import stx
from icenet.tools import io
from icenet.tools import aux
from icenet.tools import aux_torch

from icenet.tools import plots
from icenet.tools import prints
from icenet.deep  import optimize


#from icenet.deep  import dev_dndt
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

from icefit import mine

from icenet.optim import adam
from icenet.optim import adamax
from icenet.optim import scheduler


# Raytune
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
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
    #elif conv_type == 'dndt':
    #    model = dev_dndt.DNDT(**netparam)
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

    print(model)

    return model


def getgraphmodel(conv_type, netparam):
    """
    Wrapper to return different graph networks
    """

    print(conv_type)
    print(netparam)
    """
    if conv_type == 'NN':
        model = graph.NNNet(**netparam)
    else:
    """
    model = graph.GNNGeneric(conv_type=conv_type, **netparam)
    
    print(model)
    
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
    try:
        num_node_features   = data_trn[0].x.size(-1)
    except:
        num_node_features   = 0

    try:
        num_edge_features   = data_trn[0].edge_attr.size(-1)
    except:
        num_edge_features   = 0

    try:
        num_global_features = len(data_trn[0].u)
    except:
        num_global_features = 0

    netparam = {
        'c_dim' : int(num_classes),
        'd_dim' : int(num_node_features),
        'e_dim' : int(num_edge_features),
        'u_dim' : int(num_global_features)
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
    num_samples    = args['raytune']['param']['num_samples']
    max_num_epochs = args['raytune']['param']['max_num_epochs']
    
    ### Construct hyperparameter config (setup) from yaml
    steer  = param['raytune']
    config = {}

    for key in args['raytune']['setup'][steer]['param']:

        rtp   = args['raytune']['setup'][steer]['param'][key]['type']
        value = args['raytune']['setup'][steer]['param'][key]['value']

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
    metric     = args['raytune']['setup'][steer]['search_metric']['metric']
    mode       = args['raytune']['setup'][steer]['search_metric']['mode']
    
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


def torch_loop(model, train_loader, test_loader, args, param, config={}, save_period=1):
    """
    Main training loop for all torch based models
    """

    DA_active = True if (hasattr(model, 'DA_active') and model.DA_active) else False
    
    trn_aucs  = []
    val_aucs  = []
    
    # Transfer to CPU / GPU
    model, device = optimize.model_to_cuda(model=model, device_type=param['device'])

    ### ** Optimization hyperparameters [possibly from Raytune] **
    opt_param = {}
    for key in param['opt_param'].keys():
        opt_param[key]       = config[key] if key in config.keys() else param['opt_param'][key]
    
    scheduler_param = {}
    for key in param['scheduler_param'].keys():
        scheduler_param[key] = config[key] if key in config.keys() else param['scheduler_param'][key]
    
    # Create optimizer
    if   opt_param['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),  lr=opt_param['lr'], weight_decay=opt_param['weight_decay'])
    elif opt_param['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt_param['lr'], weight_decay=opt_param['weight_decay'])
    else:
        raise Exception(__name__ + f".torch_loop: Unknown optimizer <{opt_param['optimizer']}> chosen")
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_param['step_size'], gamma=scheduler_param['gamma'])
    
    cprint(__name__ + f'.torch_loop: Number of free model parameters = {aux_torch.count_parameters_torch(model)}', 'yellow')
    
    # --------------------------------------------------------------------
    ## Mutual information regularization
    if 'MI_reg_param' in param:

        # Create network and set parameters
        MI         = copy.deepcopy(param['MI_reg_param']) # ! important
        input_size = param['MI_reg_param']['x_dim']

        index      = MI['y_index']
        if   index is None:
            input_size += model.C
        elif type(index) is str:
            input_size += eval(index)
        elif type(index) is list:
            input_size += len(index)

        MI['model']     = []
        MI['MI_lb']     = []

        for k in range(len(MI['classes'])):
            MI_model         = mine.MINENet(input_size=input_size, **MI)
            MI_model, device = optimize.model_to_cuda(model=MI_model, device_type=param['device'])
            MI_model.train() # !

            MI['model'].append(MI_model)

        all_parameters = list()
        for k in range(len(MI['classes'])):
            all_parameters += list(MI['model'][k].parameters())

        MI['optimizer'] = torch.optim.Adam(all_parameters, lr=MI['lr'], weight_decay=MI['weight_decay'])
    else:
        MI = None
    # --------------------------------------------------------------------

    # Training loop
    loss_history = {}

    for epoch in range(opt_param['epochs']):

        if MI is not None: # Reset diagnostics
            MI['MI_lb'] = np.zeros(len(MI['classes']))

        loss = optimize.train(model=model, loader=train_loader, optimizer=optimizer, device=device, opt_param=opt_param, MI=MI)

        if (epoch % save_period) == 0:
            train_acc, train_auc       = optimize.test(model=model, loader=train_loader, optimizer=optimizer, device=device)
            validate_acc, validate_auc = optimize.test(model=model, loader=test_loader,  optimizer=optimizer, device=device)
        
        ## Save values
        optimize.trackloss(loss=loss, loss_history=loss_history)
        trn_aucs.append(train_auc)
        val_aucs.append(validate_auc)

        print(__name__)
        print(f'.torch_loop: Epoch {epoch+1:03d} / {opt_param["epochs"]:03d} | Loss: {optimize.printloss(loss)} Train: {train_acc:.4f} (acc), {train_auc:.4f} (AUC) | Validate: {validate_acc:.4f} (acc), {validate_auc:.4f} (AUC) | lr = {scheduler.get_last_lr()}')
        if MI is not None:
            print(f'.torch_loop: Final MI network_loss = {MI["network_loss"]:0.4f}')
            for k in range(len(MI['classes'])):
                print(f'.torch_loop: k = {k}: MI_lb value = {MI["MI_lb"][k]:0.4f}')

        # Update scheduler
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
        plotdir  = aux.makedir(f'{args["plotdir"]}/train/loss')
        fig,ax   = plots.plot_train_evolution_multi(loss_history, trn_aucs, val_aucs, param['label'])
        plt.savefig(f"{plotdir}/{param['label']}--evolution.pdf", bbox_inches='tight'); plt.close()

        return model

    return # No return value for raytune


def train_torch_graph(config={}, data_trn=None, data_val=None, args=None, param=None, y_soft=None, save_period=5):
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
    print(__name__ + f'.train_torch_graph: Training <{param["label"]}> classifier ...')
    
    # Construct model
    netparam, conv_type = getgraphparam(data_trn=data_trn, num_classes=args['num_classes'], param=param, config=config)
    model               = getgraphmodel(conv_type=conv_type, netparam=netparam)    

    ### ** Optimization hyperparameters [possibly from Raytune] **
    opt_param = {}
    for key in param['opt_param'].keys():
        opt_param[key] = config[key] if key in config.keys() else param['opt_param'][key]

    # ** Set distillation training targets **
    if y_soft is not None:
        for i in range(len(data_trn)):
            data_trn[i].y = y_soft[i]

    # Data loaders
    train_loader = torch_geometric.loader.DataLoader(data_trn, batch_size=opt_param['batch_size'], shuffle=True)
    test_loader  = torch_geometric.loader.DataLoader(data_val, batch_size=512, shuffle=False)
    
    return torch_loop(model=model, train_loader=train_loader, test_loader=test_loader, \
                args=args, param=param, config=config)


def train_torch_generic(X_trn=None, Y_trn=None, X_val=None, Y_val=None,
    trn_weights=None, val_weights=None, X_trn_2D=None, X_val_2D=None, args=None, param=None, 
    Y_trn_DA=None, trn_weights_DA=None, Y_val_DA=None, val_weights_DA=None, y_soft=None, 
    data_trn_MI=None, data_val_MI=None, config={}):
    """
    Train generic neural model [R^d x (2D) -> output]
    
    Args:
        See other train_*
    
    Returns:
        trained model
    """
    print(__name__ + f'.train_torch_generic: Training <{param["label"]}> classifier ...')

    model, train_loader, test_loader = \
        torch_construct(X_trn=X_trn, Y_trn=Y_trn, X_val=X_val, Y_val=Y_val, X_trn_2D=X_trn_2D, X_val_2D=X_val_2D, \
                trn_weights=trn_weights, val_weights=val_weights, param=param, args=args, config=config, \
                y_soft=y_soft, Y_trn_DA=Y_trn_DA, trn_weights_DA=trn_weights_DA, Y_val_DA=Y_val_DA, val_weights_DA=val_weights_DA,
                data_trn_MI=data_trn_MI, data_val_MI=data_val_MI)
    
    # Set MI-regularization X-dimension
    if 'MI_reg_param' in param:
        if 'x_dim' not in param['MI_reg_param']:
            param['MI_reg_param']['x_dim'] = data_trn_MI.shape[1]
    
    return torch_loop(model=model, train_loader=train_loader, test_loader=test_loader, \
                args=args, param=param, config=config)


def torch_construct(X_trn, Y_trn, X_val, Y_val, X_trn_2D, X_val_2D, trn_weights, val_weights, param, args, 
    Y_trn_DA=None, trn_weights_DA=None, Y_val_DA=None, val_weights_DA=None, y_soft=None, 
    data_trn_MI=None, data_val_MI=None, config={}):
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
        training_set   = optimize.DualDataset(X=X_trn_2D, U=X_trn, Y=Y_trn if y_soft is None else y_soft, W=trn_weights, Y_DA=Y_trn_DA, W_DA=trn_weights_DA, X_MI=data_trn_MI)
        validation_set = optimize.DualDataset(X=X_val_2D, U=X_val, Y=Y_val, W=val_weights, Y_DA=Y_val_DA, W_DA=val_weights_DA, X_MI=data_val_MI)
    else:
        training_set   = optimize.Dataset(X=X_trn, Y=Y_trn if y_soft is None else y_soft, W=trn_weights, Y_DA=Y_trn_DA, W_DA=trn_weights_DA, X_MI=data_trn_MI)
        validation_set = optimize.Dataset(X=X_val, Y=Y_val, W=val_weights, Y_DA=Y_val_DA, W_DA=val_weights_DA, X_MI=data_val_MI)

    ### ** Optimization hyperparameters [possibly from Raytune] **
    opt_param = {}
    for key in param['opt_param'].keys():
        opt_param[key] = config[key] if key in config.keys() else param['opt_param'][key]

    params = {'batch_size'  : opt_param['batch_size'],
              'shuffle'     : True,
              'num_workers' : param['num_workers'],
              'pin_memory'  : True}

    train_loader = torch.utils.data.DataLoader(training_set,   **params)
    test_loader  = torch.utils.data.DataLoader(validation_set, **params)

    return model, train_loader, test_loader


def train_flr(config={}, data_trn=None, args=None, param=None):
    """
    Train factorized likelihood model

    Args:
        See other train_*

    Returns:
        trained model
    """
    print(__name__ + f'.train_flr: Training <{param["label"]}> classifier ...')

    b_pdfs, s_pdfs, bin_edges = flr.train(X = data_trn.x, y = data_trn.y, weights = data_trn.w, param = param)
    pickle.dump([b_pdfs, s_pdfs, bin_edges],
        open(args['modeldir'] + f'/{param["label"]}_' + str(0) + '_.dat', 'wb'))

    def func_predict(X):
        return flr.predict(X, b_pdfs, s_pdfs, bin_edges)

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

    print(__name__ + f'.train_flow: Training <{param["label"]}> classifier ...')
    
    for classid in range(args['num_classes']):
        param['model'] = 'class_' + str(classid)

        # Load datasets
        trn   = data_trn.classfilter(classid)
        val   = data_val.classfilter(classid)

        # Create model
        model = dbnf.create_model(param=param['model_param'], verbose = True)

        # Create optimizer & scheduler
        if   param['opt_param']['optimizer'] == 'Adam':
            optimizer = adam.Adam(model.parameters(), lr = param['opt_param']['lr'], \
                weight_decay = param['opt_param']['weight_decay'], polyak = param['opt_param']['polyak'])
        
        elif param['opt_param']['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr = param['opt_param']['lr'], \
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
            trn_x=trn.x, val_x=val.x, trn_weights=trn.w, val_weights=val.w, param=param, modeldir=args['modeldir'])
        
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

                model = train.torch_loop(model=model, train_loader=train_loader, test_loader=test_loader, \
                            args=args, param=param['model_param'])

            except:
                print(__name__ + f'.train_xtx: Problem in training with bin: {var0} = {range0}], {var1} = {range1}')

    return True
