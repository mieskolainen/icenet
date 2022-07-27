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


#from icenet.deep  import dev_dndt
from icenet.deep  import deps
from icenet.algo  import flr
from icenet.deep  import bnaf
from icenet.deep  import mlgr
from icenet.deep  import maxo
from icenet.deep  import dmlp
from icenet.deep  import dbnf
from icenet.deep  import vae
from icefit import mine

from icenet.deep  import cnn
from icenet.deep  import graph

from icenet.optim import adam
from icenet.optim import adamax
from icenet.optim import scheduler


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
            MI_model, device = dopt.model_to_cuda(model=MI_model, device_type=param['device'])
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

        loss = dopt.train(model=model, loader=train_loader, optimizer=optimizer, device=device, opt_param=opt_param, MI=MI)

        if (epoch % save_period) == 0:
            train_acc, train_auc       = dopt.test(model=model, loader=train_loader, optimizer=optimizer, device=device)
            validate_acc, validate_auc = dopt.test(model=model, loader=test_loader,  optimizer=optimizer, device=device)
        
        ## Save values
        dopt.trackloss(loss=loss, loss_history=loss_history)
        trn_aucs.append(train_auc)
        val_aucs.append(validate_auc)

        print(__name__)
        print(f'.torch_loop: Epoch {epoch+1:03d} / {opt_param["epochs"]:03d} | Loss: {dopt.printloss(loss)} Train: {train_acc:.4f} (acc), {train_auc:.4f} (AUC) | Validate: {validate_acc:.4f} (acc), {validate_auc:.4f} (AUC) | lr = {scheduler.get_last_lr()}')
        if MI is not None:
            print(f'.torch_loop: MI network_loss = {MI["network_loss"]:0.4f}')
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
        plt.savefig(f"{plotdir}/{param['label']}_evolution.pdf", bbox_inches='tight'); plt.close()

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
        training_set   = dopt.DualDataset(X=X_trn_2D, U=X_trn, Y=Y_trn if y_soft is None else y_soft, W=trn_weights, Y_DA=Y_trn_DA, W_DA=trn_weights_DA, X_MI=data_trn_MI)
        validation_set = dopt.DualDataset(X=X_val_2D, U=X_val, Y=Y_val, W=val_weights, Y_DA=Y_val_DA, W_DA=val_weights_DA, X_MI=data_val_MI)
    else:
        training_set   = dopt.Dataset(X=X_trn, Y=Y_trn if y_soft is None else y_soft, W=trn_weights, Y_DA=Y_trn_DA, W_DA=trn_weights_DA, X_MI=data_trn_MI)
        validation_set = dopt.Dataset(X=X_val, Y=Y_val, W=val_weights, Y_DA=Y_val_DA, W_DA=val_weights_DA, X_MI=data_val_MI)

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


def _binary_CE_with_MI(preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor=None, EPS=1E-30):
    """
    Custom binary cross entropy loss with mutual information regularization
    """
    global MI_x
    global MI_reg_param
    global loss_history
    
    if weights is None:
        w = torch.ones(len(preds)).to(preds.device)
    else:
        w = weights / torch.sum(weights)
    
    ## Squared hinge-loss type
    #targets = 2 * targets - 1
    #classifier_loss = w * torch.max(torch.zeros_like(preds), 1 - preds * targets) ** 2
    #classifier_loss = classifier_loss.sum()
    
    ## Sigmoid link function
    phat = 1 / (1 + torch.exp(-preds))
    
    ## Classifier binary Cross Entropy
    CE_loss  = - w * (targets*torch.log(phat + EPS) + (1-targets)*(torch.log(1 - phat + EPS)))
    CE_loss  = CE_loss.sum()
    
    #lossfunc = torch.nn.BCELoss(weight=w, reduction='sum')
    #classifier_loss = lossfunc(phat, targets)

    ## Domain Adaptation
    # [reservation] ...

    ## Regularization

    # Loop over chosen classes
    k       = 0
    MI_loss = 0
    MI_lb_values = []

    for c in MI_reg_param['classes']:

        # -----------
        reg_param = copy.deepcopy(MI_reg_param)
        reg_param['ma_eT'] = reg_param['ma_eT'][k] # Pick the one
        # -----------

        if c == None:
            ind = (targets != -1) # All classes
        else:
            ind = (targets == c)

        X     = torch.Tensor(MI_x).to(preds.device)
        Z     = preds
        
        # We need .detach() here for Z!
        model = mine.estimate(X=X[ind], Z=Z[ind].detach(), weights=weights[ind],
            return_model_only=True, device=X.device, **reg_param)

        # ------------------------------------------------------------
        # Now apply the MI estimator to the sample

        # No .detach() here, we need the gradients for Z!
        MI_lb = mine.apply_in_batches(X=X[ind], Z=Z[ind], weights=weights[ind], model=model)
        # ------------------------------------------------------------

        MI_loss = MI_loss + MI_reg_param['beta'][k] * MI_lb
        MI_lb_values.append(np.round(MI_lb.item(), 4))

        k += 1

    ## Total
    total_loss = CE_loss + MI_loss
    cprint(f'Total_loss = {total_loss:0.4f} | CE_loss = {CE_loss:0.4f} | MI_loss = {MI_loss:0.4f} | MI_lb = {MI_lb_values}', 'yellow')

    loss = {'sum': total_loss.item(), 'CE': CE_loss.item(), f'MI, $\\beta = {MI_reg_param["beta"]}$': MI_loss.item()}
    dopt.trackloss(loss=loss, loss_history=loss_history)

    # Scale finally to the total number of events (to conform with xgboost internal convention)
    return total_loss * len(preds)


def train_xgb(config={}, data_trn=None, data_val=None, y_soft=None, args=None, param=None, plot_importance=True,
    data_trn_MI=None, data_val_MI=None):
    """
    Train XGBoost model
    
    Args:
        See other train_*
    
    Returns:
        trained model
    """

    if 'MI_reg_param' in param:

        global MI_x
        global MI_reg_param
        global loss_history

        loss_history  = {}
        MI_reg_param  = copy.deepcopy(param['MI_reg_param']) #! important
    # ---------------------------------------------------

    if param['model_param']['tree_method'] == 'auto':
        param['model_param'].update({'tree_method': 'gpu_hist' if torch.cuda.is_available() else 'hist'})

    print(__name__ + f'.train_xgb: Training <{param["label"]}> classifier ...')

    ### ** Optimization hyperparameters [possibly from Raytune] **
    if config is not {}:
        for key in param['model_param'].keys():
            param['model_param'][key] = config[key] if key in config.keys() else param['model_param'][key]

    ### *********************************
    
    # Normalize weights to sum to number of events (xgboost library has no scale normalization)
    w_trn     = data_trn.w / np.sum(data_trn.w) * data_trn.w.shape[0]
    w_val     = data_val.w / np.sum(data_val.w) * data_val.w.shape[0]

    dtrain    = xgboost.DMatrix(data = data_trn.x, label = data_trn.y if y_soft is None else y_soft, weight = w_trn)
    deval     = xgboost.DMatrix(data = data_val.x, label = data_val.y,  weight = w_val)

    evallist  = [(dtrain, 'train'), (deval, 'eval')]
    print(param)

    trn_losses = []
    val_losses = []
    
    trn_aucs   = []
    val_aucs   = []

    # ---------------------------------------
    # Update the parameters
    model_param = copy.deepcopy(param['model_param'])
    
    if 'multi' in model_param['objective']:
        model_param.update({'num_class': args['num_classes']})

    del model_param['num_boost_round']
    # ---------------------------------------

    # Boosting iterations
    max_num_epochs = param['model_param']['num_boost_round']
    for epoch in range(max_num_epochs):

        results = dict()
        
        a = {'params':          copy.deepcopy(model_param),
             'dtrain':          dtrain,
             'num_boost_round': 1,
             'evals':           evallist,
             'evals_result':    results,
             'verbose_eval':    False}

        # == Custom loss ==
        if 'custom' in model_param['objective']:
            import icenet.deep.autogradxgb as autogradxgb

            strs   = model_param['objective'].split(':')
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'

            if strs[1] == 'binary_cross_entropy_with_MI':

                a['obj'] = autogradxgb.XgboostObjective(loss_func=_binary_CE_with_MI, skip_hessian=True, device=device)
                a['params']['disable_default_eval_metric'] = 1

                # ! Important to have here because we modify it in the eval below
                MI_x = copy.deepcopy(data_trn_MI)

                def eval_obj(mode='train'):
                    global MI_x #!
                    obj = autogradxgb.XgboostObjective(loss_func=_binary_CE_with_MI, mode='eval', device=device)
                    
                    if mode == 'train':
                        MI_x = copy.deepcopy(data_trn_MI)
                        loss = obj(preds=model.predict(dtrain), targets=dtrain)[1] / len(MI_x)
                    elif mode == 'eval':
                        MI_x = copy.deepcopy(data_val_MI)
                        loss = obj(preds=model.predict(deval), targets=deval)[1] / len(MI_x)
                    return loss
            else:
                raise Exception(__name__ + f'.train_xgb: Unknown custom loss {strs[1]}')

            #!
            del a['params']['eval_metric']
            del a['params']['objective']
        # -----------------

        if epoch > 0: # Continue from the previous epoch model
            a['xgb_model'] = model

        # Train it
        model = xgboost.train(**a)

        # ------- AUC values ------
        pred    = model.predict(dtrain)
        if len(pred.shape) > 1: pred = pred[:, args['signalclass']]
        metrics = aux.Metric(y_true=data_trn.y, y_pred=pred, weights=w_trn, num_classes=args['num_classes'], hist=False, verbose=True)
        trn_aucs.append(metrics.auc)
        
        pred    = model.predict(deval)
        if len(pred.shape) > 1: pred = pred[:, args['signalclass']]
        metrics = aux.Metric(y_true=data_val.y, y_pred=pred, weights=w_val, num_classes=args['num_classes'], hist=False, verbose=True)
        val_aucs.append(metrics.auc)
        # -------------------------

        # ------ Loss values ------
        if 'custom' in model_param['objective']:
            trn_losses.append(0)#eval_obj('train'))
            val_losses.append(0)#eval_obj('eval'))
        else:
            trn_losses.append(results['train'][model_param['eval_metric'][0]][0])
            val_losses.append(results['eval'][model_param['eval_metric'][0]][0])
        # -------------------------

        print(__name__ + f'.train_xgb: Tree {epoch+1:03d}/{max_num_epochs:03d} | Train: loss = {trn_losses[-1]:0.4f}, AUC = {trn_aucs[-1]:0.4f} | Eval: loss = {val_losses[-1]:0.4f}, AUC = {val_aucs[-1]:0.4f}')
        
        if args['__raytune_running__']:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                pickle.dump(model, open(path, 'wb'))

            tune.report(loss = trn_losses[-1], AUC = val_aucs[-1])
        else:
            ## Save
            filename = args['modeldir'] + f'/{param["label"]}_{epoch}'
            pickle.dump(model, open(filename + '.dat', 'wb'))
            model.save_model(filename + '.json')
            model.dump_model(filename + '.text', dump_format='text')
        
    if not args['__raytune_running__']:
        
        # Plot evolution
        plotdir  = aux.makedir(f'{args["plotdir"]}/train/loss')

        if 'custom' not in model_param['objective']:
            fig,ax   = plots.plot_train_evolution_multi(losses={'train': trn_losses, 'validate': val_losses},
                trn_aucs=trn_aucs, val_aucs=val_aucs, label=param["label"])
        else:
            fig,ax   = plots.plot_train_evolution_multi(losses=loss_history,
                trn_aucs=trn_aucs, val_aucs=val_aucs, label=param["label"])    
        plt.savefig(f'{plotdir}/{param["label"]}_evolution.pdf', bbox_inches='tight'); plt.close()
        
        ## Plot feature importance
        if plot_importance:
            
            for sort in [True, False]:
                fig,ax = plots.plot_xgb_importance(model=model, tick_label=data_trn.ids, label=param["label"], sort=sort)
                targetdir = aux.makedir(f'{args["plotdir"]}/train/xgb_importance')
                plt.savefig(f'{targetdir}/{param["label"]}_importance_sort_{sort}.pdf', bbox_inches='tight'); plt.close()
                
        ## Plot decision trees
        if ('plot_trees' in param) and param['plot_trees']:
            try:
                print(__name__ + f'.train_xgb: Plotting decision trees ...')
                model.feature_names = data_trn.ids
                for i in tqdm(range(max_num_epochs)):
                    xgboost.plot_tree(model, num_trees=i)
                    fig = plt.gcf(); fig.set_size_inches(60, 20) # Higher reso
                    path = aux.makedir(f'{targetdir}/trees_{param["label"]}')
                    plt.savefig(f'{path}/tree_{i}.pdf', bbox_inches='tight'); plt.close()
            except:
                print(__name__ + f'.train_xgb: Could not plot the decision trees (try: conda install python-graphviz)')
            
        model.feature_names = None # Set original default ones

        return model

    return # No return value for raytune


def train_graph_xgb(config={}, data_trn=None, data_val=None, trn_weights=None, val_weights=None,
    args=None, y_soft=None, param=None, feature_names=None):
    """
    Train graph model + xgb hybrid model
    
    Args:
        See other train_*

    Returns:
        trained model
    """
    if param['xgb']['model_param']['tree_method'] == 'auto':
        param['xgb']['model_param'].update({'tree_method' : 'gpu_hist' if torch.cuda.is_available() else 'hist'})
    
    print(__name__ + f'.train_graph_xgb: Training <{param["label"]}> classifier ...')

    # --------------------------------------------------------------------
    ### Train GNN
    graph_model = train_torch_graph(data_trn=data_trn, data_val=data_val, args=args, param=param['graph'], y_soft=y_soft)
    graph_model = graph_model.to('cpu:0')    
    
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
    ## Evaluate GNN output

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

    # ------------------------------------------------------------------------------
    ## Train xgboost

    # Normalize weights to sum to the number of events (xgboost library has no scale normalization)
    w_trn     = trn_weights / np.sum(trn_weights) * trn_weights.shape[0]
    w_val     = val_weights / np.sum(val_weights) * val_weights.shape[0]

    dtrain    = xgboost.DMatrix(data = x_trn, label = y_trn if y_soft is None else y_soft.detach().cpu().numpy(), weight = w_trn)
    deval     = xgboost.DMatrix(data = x_val, label = y_val, weight = w_val)
    
    evallist  = [(dtrain, 'train'), (deval, 'eval')]
    results   = dict()

    trn_losses = []
    val_losses = []
    
    trn_aucs   = []
    val_aucs   = []

    # ---------------------------------------
    # Update the parameters
    model_param = copy.deepcopy(param['xgb']['model_param'])
    
    if 'multi' in model_param['objective']:
        model_param.update({'num_class': args['num_classes']})

    del model_param['num_boost_round']
    # ---------------------------------------

    # Boosting iterations
    max_num_epochs = param['xgb']['model_param']['num_boost_round']
    for epoch in range(max_num_epochs):
        
        results = dict()
        
        a = {'params':          model_param,
             'dtrain':          dtrain,
             'num_boost_round': 1,
             'evals':           evallist,
             'evals_result':    results,
             'verbose_eval':    False}

        if epoch > 0: # Continue from the previous epoch model
            a['xgb_model'] = model

        # Train it
        model = xgboost.train(**a)
        
        # AUC
        pred    = model.predict(dtrain)
        if len(pred.shape) > 1: pred = pred[:, args['signalclass']]
        metrics = aux.Metric(y_true=y_trn, y_pred=pred, weights=w_trn, num_classes=args['num_classes'], hist=False, verbose=True)
        trn_aucs.append(metrics.auc)

        pred    = model.predict(deval)
        if len(pred.shape) > 1: pred = pred[:, args['signalclass']]
        metrics = aux.Metric(y_true=y_val, y_pred=pred, weights=w_val, num_classes=args['num_classes'], hist=False, verbose=True)
        val_aucs.append(metrics.auc)

        # Loss
        trn_losses.append(results['train'][model_param['eval_metric'][0]][0])
        val_losses.append(results['eval'][model_param['eval_metric'][0]][0])

        ## Save
        pickle.dump(model, open(args['modeldir'] + f"/{param['xgb']['label']}_{epoch}.dat", 'wb'))
        
        print(__name__ + f'.train_graph_xgb: Tree {epoch+1:03d}/{max_num_epochs:03d} | Train: loss = {trn_losses[-1]:0.4f}, AUC = {trn_aucs[-1]:0.4f} | Eval: loss = {val_losses[-1]:0.4f}, AUC = {val_aucs[-1]:0.4f}')
    
    # ------------------------------------------------------------------------------
    # Plot evolution
    plotdir  = aux.makedir(f'{args["plotdir"]}/train/')
    fig,ax   = plots.plot_train_evolution(losses=np.array([trn_losses, val_losses]).T,
                    trn_aucs=trn_aucs, val_aucs=val_aucs, label=param['xgb']['label'])
    plt.savefig(f"{plotdir}/{param['xgb']['label']}_evolution.pdf", bbox_inches='tight'); plt.close()
    
    # -------------------------------------------
    ## Plot feature importance

    # Create all feature names
    ids = []
    for i in range(Z):                  # Graph-net latent dimension Z (message passing output) features
        ids.append(f'conv_Z_{i}')
    for i in range(len(data_trn[0].u)): # Xgboost high-level features
        ids.append(feature_names[i])
    
    for sort in [True, False]:
        fig,ax = plots.plot_xgb_importance(model=model, tick_label=ids, label=param["label"], sort=sort)
        targetdir = aux.makedir(f'{args["plotdir"]}/train/xgb_importance')
        plt.savefig(f'{targetdir}/{param["label"]}_importance_sort_{sort}.pdf', bbox_inches='tight'); plt.close()
        
    ## Plot decision trees
    if ('plot_trees' in param['xgb']) and param['xgb']['plot_trees']:
        try:
            print(__name__ + f'.train_graph_xgb: Plotting decision trees ...')
            model.feature_names = ids
            for i in tqdm(range(max_num_epochs)):
                xgboost.plot_tree(model, num_trees=i)
                fig = plt.gcf(); fig.set_size_inches(60, 20) # Higher reso
                path = aux.makedir(f'{targetdir}/trees_{param["label"]}')
                plt.savefig(f'{path}/tree_{i}.pdf', bbox_inches='tight'); plt.close()
        except:
            print(__name__ + f'.train_graph_xgb: Could not plot the decision trees (try: conda install python-graphviz)')
        
    model.feature_names = None # Set original default ones

    return model


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
