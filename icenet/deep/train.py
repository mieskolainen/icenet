# Generic model training wrapper functions [TBD; unify and simplify data structures further]
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import torch
import torch_geometric
import xgboost

import os
import pickle
import copy
import multiprocessing

from tqdm import tqdm
from matplotlib import pyplot as plt

# icenet
from icenet.tools import aux
from icenet.tools import aux_torch

from icenet.tools import plots
from icenet.deep  import optimize, predict, deeptools

from icenet.algo  import flr
from icenet.deep  import deps
from icenet.deep  import mlgr
from icenet.deep  import maxo
from icenet.deep  import dmlp
from icenet.deep  import lzmlp
from icenet.deep  import dbnf
from icenet.deep  import vae
from icenet.deep  import fastkan

from icenet.deep  import cnn
from icenet.deep  import graph

from icefit import mine

from icenet.optim import adam

# ------------------------------------------
from icenet import print
# ------------------------------------------

# Raytune
import ray
from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator
from ray.tune.search.hyperopt      import HyperOptSearch
from ray.tune.search.optuna        import OptunaSearch
from ray.tune.search.bayesopt      import BayesOptSearch

from ray.air.config import RunConfig, ScalingConfig
from ray.tune.schedulers import ASHAScheduler


def getgenericmodel(conv_type, netparam):
    """
    Wrapper to return different torch models
    """

    if   conv_type == 'lgr':
        model = mlgr.MLGR(**netparam)
    elif conv_type == 'dmlp':
        model = dmlp.DMLP(**netparam)
    elif conv_type == 'lzmlp':
        model = lzmlp.LZMLP(**netparam)
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
    elif conv_type == 'fastkan':
        model = fastkan.FastKAN(**netparam) 
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
    
    model = graph.GNNGeneric(conv_type=conv_type, **netparam)
    
    print(model)
    
    return model


def getgenericparam(param, D, num_classes, config={}):
    """
    Construct generic torch network parameters
    """
    netparam = {
        'C'       : int(num_classes),
        'D'       : int(D),
        'out_dim' : param['out_dim'] if 'out_dim' in param else None
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
        'C'       : int(num_classes),
        'd_dim'   : int(num_node_features),
        'e_dim'   : int(num_edge_features),
        'u_dim'   : int(num_global_features),
        'out_dim' : param['out_dim'] if 'out_dim' in param else None
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
    
    ### Construct hyperparameter config (setup) from yaml
    steer  = param['raytune']
    parameters = {}

    for key in args['raytune']['setup'][steer]['param']:
        
        rtp   = args['raytune']['setup'][steer]['param'][key]['type']
        print(f'{key}: {rtp}')
        parameters[key] = eval(rtp)
    
    # Raytune search algorithm
    algo     = args['raytune']['setup'][steer]['search_algo']
    metric   = args['raytune']['setup'][steer]['search_metric']['metric']
    mode     = args['raytune']['setup'][steer]['search_metric']['mode']
    
    # Hyperopt Bayesian
    if   algo == 'Basic':
        search_alg = BasicVariantGenerator()
    elif algo == 'HyperOpt':
        search_alg = HyperOptSearch()
    elif algo == 'Optuna':
        search_alg = OptunaSearch()
    elif algo == 'BayesOpt':
        search_alg = BayesOptSearch()
    else:
        raise Exception(__name__ + f".raytune_main: Unknown 'search_algo' (use 'Basic', 'HyperOpt', 'Optuna', 'BayesOpt')")
    
    print(f'Optimization algorithm: {algo}', 'yellow')
    
    # Raytune scheduler
    scheduler = ASHAScheduler()
    
    ## Flag raytune on for training functions
    inputs['args']['__raytune_running__'] = True
    
    # Raytune main setup
    print(f'Launching tune ...')
    
    param_space = {
        "scaling_config": ScalingConfig(
            num_workers          = multiprocessing.cpu_count(),
            resources_per_worker = {"cpu": 1, "gpu": 1 if torch.cuda.is_available() else 0}
        ),
    }
    param_space['params'] = parameters # Set hyperparameters
    
    tuner = ray.tune.Tuner(
        ray.tune.with_parameters(train_func, **inputs),
        tune_config=ray.tune.TuneConfig(
            search_alg          = search_alg,
            scheduler           = scheduler,
            metric              = metric,
            mode                = mode,
            num_samples         = num_samples
        ),
        run_config  = RunConfig(name="icenet_raytune", local_dir=os.getcwd() + "/tmp"),
        param_space = param_space,
    )
    results = tuner.fit()
    
    # Get the best config
    best_result = results.get_best_result(metric=metric, mode=mode)
    
    print('-----------------')
    print(f'Best result config: \n\n{best_result.config}',  'green')
    print('')
    print(f'Best result metrics:\n\n{best_result.metrics}', 'green')
    print('-----------------')
    
    # Set the best config, training functions will update the parameters
    inputs['config'] = {} # Create empty
    inputs['config']['params'] = best_result.config['params']
    inputs['args']['__raytune_running__'] = False
    
    # Train finally once more with the best parameters
    best_trained_model = train_func(**inputs)
    
    return best_trained_model


def torch_loop(model, train_loader, test_loader, args, param, config={'params': {}}, ids=None):
    """
    Main training loop for all torch based models
    """
    
    savedir  = aux.makedir(f'{args["modeldir"]}/{param["label"]}')
    
    # Transfer to CPU / GPU
    model, device = optimize.model_to_cuda(model=model, device_type=param['device'])

    ### ** Optimization hyperparameters [possibly from Raytune] **
    opt_param       = aux.replace_param(default=param['opt_param'], raytune=config['params'])
    scheduler_param = aux.replace_param(default=param['scheduler_param'], raytune=config['params'])
    
    # Create optimizer
    if   opt_param['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),  lr=opt_param['lr'], weight_decay=opt_param['weight_decay'])
    elif opt_param['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt_param['lr'], weight_decay=opt_param['weight_decay'])
    else:
        raise Exception(__name__ + f".torch_loop: Unknown optimizer <{opt_param['optimizer']}> chosen")
    
    # Create scheduler
    scheduler = deeptools.set_scheduler(optimizer=optimizer, param=scheduler_param)
    
    print(f'Number of free model parameters = {aux_torch.count_parameters_torch(model)}', 'yellow')
    
    
    # --------------------------------------------------------------------
    ## Mutual information regularization
    if 'MI_param' in param:
        
        print(f'Using MI regularization', 'yellow')
        
        # Create network and set parameters
        MI         = copy.deepcopy(param['MI_param']) # ! important
        input_size = param['MI_param']['x_dim']

        y_dim      = MI['y_dim']
        if   y_dim is None:
            input_size += model.C
        elif type(y_dim) is str:
            input_size += eval(y_dim)
        elif type(y_dim) is list:
            input_size += len(y_dim)

        MI['model'] = []
        MI['MI_lb'] = []
        
        print(f'MINE estimator input_size: {input_size}')
        
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
    
    # TensorboardX
    if not args['__raytune_running__'] and 'tensorboard' in param and param['tensorboard']:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(os.path.join(args['modeldir'], param['label']))
    
    # --------------------------------------------------------------------
    # Training loop
    
    loss_history_train = {}
    loss_history_eval  = {}
    
    trn_losses = []
    val_losses = []
    
    trn_aucs   = []
    val_aucs   = []
    
    for epoch in range(0, opt_param['epochs']):

        if MI is not None: # Reset diagnostics
            MI['MI_lb'] = np.zeros(len(MI['classes']))
        
        # Set current epoch (for special scheduling reasons)
        opt_param['current_epoch'] = epoch
        
        # Train
        loss = optimize.train(model=model, loader=train_loader, optimizer=optimizer, device=device, opt_param=opt_param, MI=MI)
        
        if epoch == 0 or ((epoch+1) % param['savemode']) == 0 or args['__raytune_running__']:
            _, train_acc, train_auc                   = optimize.test(model=model, loader=train_loader, device=device, opt_param=opt_param, MI=MI, compute_loss=False)
            validate_loss, validate_acc, validate_auc = optimize.test(model=model, loader=test_loader,  device=device, opt_param=opt_param, MI=MI, compute_loss=True)
            
            # Temperature calibration
            try:
                from icenet.deep import tempscale
                tt = tempscale.ModelWithTemperature(model=model, device=device, mode='softmax' if model.out_dim > 1 else 'binary')
                tt.calibrate(valid_loader=test_loader)
            except Exception as e:
                print(e)
                print('Could not evaluate temperature scaling -- skip')
        
        ## ** Save values **
        optimize.trackloss(loss=loss, loss_history=loss_history_train)
        optimize.trackloss(loss=validate_loss, loss_history=loss_history_eval)
        
        trn_losses.append(loss_history_train['sum'][-1])
        val_losses.append(loss_history_eval['sum'][-1])
        
        trn_aucs.append(train_auc)
        val_aucs.append(validate_auc)
        
        print(__name__)
        print(f'[{param["label"]}] Epoch {epoch+1:03d} / {opt_param["epochs"]:03d} | Train: {optimize.printloss(loss)} (loss) {train_acc:.4f} (acc) {train_auc:.4f} (AUC) | Eval: {optimize.printloss(validate_loss)} (loss) {validate_acc:.4f} (acc) {validate_auc:.4f} (AUC) | lr = {scheduler.get_last_lr()[0]:0.4E}', 'yellow')
        if MI is not None:
            print(f'Final MI network_loss = {MI["network_loss"]:0.4f}')
            for k in range(len(MI['classes'])):
                print(f'k = {k}: MI_lb value = {MI["MI_lb"][k]:0.4f}')
        
        # Update scheduler
        scheduler.step()
        
        if not args['__raytune_running__'] and 'tensorboard' in param and param['tensorboard']:
            writer.add_scalar('lr', scheduler.get_last_lr()[0],  epoch)
            writer.add_scalar('loss/validation', val_losses[-1], epoch)
            writer.add_scalar('loss/train',      trn_losses[-1], epoch)
            writer.add_scalar('AUC/validation',  val_aucs[-1],   epoch)
            writer.add_scalar('AUC/train',       trn_aucs[-1],   epoch)
        
        if not args['__raytune_running__']:
            
            ## Save the model
            filename = savedir + f'/{param["label"]}_{epoch}.pth'
            
            losses   = {'trn_losses':         trn_losses,
                        'val_losses':         val_losses,
                        'trn_aucs':           trn_aucs,
                        'val_aucs:':          val_aucs,
                        'loss_history_train': loss_history_train,
                        'loss_history_eval':  loss_history_eval}
            
            checkpoint = {'model': model, 'state_dict': model.state_dict(),
                          'ids': ids, 'losses': losses, 'epoch': epoch, 'param': param}
            torch.save(checkpoint, filename)
            
        else:
            ray.train.report({'loss': val_losses[-1], 'AUC': validate_auc})
    
    if not args['__raytune_running__']:
        
        # Plot evolution
        plotdir = aux.makedir(f'{args["plotdir"]}/train/loss/{param["label"]}')
        
        ltr = {f'train: {k}': v for k, v in loss_history_train.items()}
        lev = {f'eval:  {k}': v for k, v in loss_history_eval.items()}
        
        losses_ = ltr | lev
        
        for yscale in ['linear', 'log']:
            for xscale in ['linear', 'log']:
                
                fig,ax = plots.plot_train_evolution_multi(losses=losses_, trn_aucs=trn_aucs, val_aucs=val_aucs,
                                                          label=param["label"], yscale=yscale, xscale=xscale)
                
                plt.savefig(f"{plotdir}/{param['label']}_losses_yscale_{yscale}_xscale_{xscale}.pdf", bbox_inches='tight')
                plt.close(fig)
        
        return model

    return # No return value for raytune


def train_torch_graph(config={'params': {}}, data_trn=None, data_val=None, args=None, param=None, y_soft=None):
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
    print(f'Training <{param["label"]}> classifier ...')
    print(config)
    
    # Construct model
    netparam, conv_type = getgraphparam(data_trn=data_trn, num_classes=len(args['primary_classes']), param=param, config=config['params'])
    model               = getgraphmodel(conv_type=conv_type, netparam=netparam)

    ### ** Optimization hyperparameters [possibly from Raytune] **
    opt_param = aux.replace_param(default=param['opt_param'], raytune=config['params'])
    
    # ** Set distillation training targets **
    if y_soft is not None:
        for i in range(len(data_trn)):
            data_trn[i].y = y_soft[i]

    # Data loaders
    train_loader = torch_geometric.loader.DataLoader(data_trn, batch_size=opt_param['batch_size'],  shuffle=True)
    test_loader  = torch_geometric.loader.DataLoader(data_val, batch_size=param['eval_batch_size'], shuffle=True)
    
    return torch_loop(model=model, train_loader=train_loader, test_loader=test_loader, \
                args=args, param=param, config=config)


def train_torch_generic(X_trn=None, Y_trn=None, X_val=None, Y_val=None,
    trn_weights=None, val_weights=None, X_trn_2D=None, X_val_2D=None, args=None, param=None, 
    Y_trn_DA=None, trn_weights_DA=None, Y_val_DA=None, val_weights_DA=None, y_soft=None, 
    data_trn_MI=None, data_val_MI=None, ids=None, config={'params': {}}):
    """
    Train generic neural model [R^d x (2D) -> output]
    
    Args:
        See other train_*
    
    Returns:
        trained model
    """
    print(f'Training <{param["label"]}> classifier ...')

    model, train_loader, test_loader = \
        torch_construct(X_trn=X_trn, Y_trn=Y_trn, X_val=X_val, Y_val=Y_val, X_trn_2D=X_trn_2D, X_val_2D=X_val_2D, \
                trn_weights=trn_weights, val_weights=val_weights, param=param, args=args, config=config, \
                y_soft=y_soft, Y_trn_DA=Y_trn_DA, trn_weights_DA=trn_weights_DA, Y_val_DA=Y_val_DA, val_weights_DA=val_weights_DA,
                data_trn_MI=data_trn_MI, data_val_MI=data_val_MI)
    
    # Set MI-regularization X-dimension
    if 'MI_param' in param:
        if 'x_dim' not in param['MI_param']:
            param['MI_param']['x_dim'] = data_trn_MI.shape[1]
    
    return torch_loop(model=model, train_loader=train_loader, test_loader=test_loader, \
                args=args, param=param, config=config, ids=ids)


def torch_construct(X_trn, Y_trn, X_val, Y_val, X_trn_2D, X_val_2D, trn_weights, val_weights, param, args, 
    Y_trn_DA=None, trn_weights_DA=None, Y_val_DA=None, val_weights_DA=None, y_soft=None, 
    data_trn_MI=None, data_val_MI=None, config={'params': {}}):
    """
    Torch model and data loader constructor

    Args:
        See other train_*
    
    Returns:
        model, train_loader, test_loader
    """

    ## ------------------------
    ### Construct model
    netparam, conv_type = getgenericparam(config=config['params'], param=param, D=X_trn.shape[-1], num_classes=len(args['primary_classes']))
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
    opt_param = aux.replace_param(default=param['opt_param'], raytune=config['params'])
    
    # N.B. We use 'sampler' with 'BatchSampler', which loads a set of events using multiple event indices (faster) than the default
    # one which takes events one-by-one and concatenates the results (slow).
    
    ## ------------------------
    # If True, then all batches are the same size (i.e. the last small one is skipped)
    if 'drop_last' in opt_param:
        drop_last = opt_param['drop_last']
    else:
        drop_last = False
        
    params_train = {'batch_size'  : None,
                    'num_workers' : param['num_workers'],
                    'sampler'     : torch.utils.data.BatchSampler(
                        torch.utils.data.RandomSampler(training_set), opt_param['batch_size'], drop_last=drop_last
                    ),
                    'pin_memory'  : True}
    
    params_test  = {'batch_size'  : None,
                    'num_workers' : param['num_workers'],
                    'sampler'     : torch.utils.data.BatchSampler(
                        torch.utils.data.RandomSampler(validation_set), param['eval_batch_size'], drop_last=drop_last
                    ),
                    'pin_memory'  : True}
        
    train_loader = torch.utils.data.DataLoader(training_set,   **params_train)
    test_loader  = torch.utils.data.DataLoader(validation_set, **params_test)
    
    return model, train_loader, test_loader


def train_cutset(config={'params': {}}, data_trn=None, data_val=None, args=None, param=None):
    """
    Train cutset model

    Args:
        See other train_*

    Returns:
        Trained model
    """
    print(f'Training <{param["label"]}> classifier ...')
    print(config)
    
    model_param = aux.replace_param(default=param['model_param'], raytune=config['params'])
    
    new_param   = copy.deepcopy(param)
    new_param['model_param'] = model_param
    
    x         = data_trn.x
    y_true    = data_trn.y
    weights   = data_trn.w
    
    pred_func = predict.pred_cutset(ids=data_trn.ids, param=new_param)
    
    # Apply cutset
    y_pred    = pred_func(x)
    
    # Metrics
    metrics   = aux.Metric(y_true=y_true, y_pred=y_pred, class_ids=args['primary_classes'], weights=weights, hist=False, verbose=True)
    
    # ------------------------------------------------------
    # Compute loss
    for p in param['opt_param']['lossfunc_var'].keys():
        exec(f"{p} = param['opt_param']['lossfunc_var']['{p}']")
    
    y_pred = y_pred.astype(int)
    
    # Efficiency
    eff_s = np.sum(weights[np.logical_and(y_pred == 1, y_true == 1)]) / np.sum(weights[y_true == 1])
    eff_b = np.sum(weights[np.logical_and(y_pred == 0, y_true == 0)]) / np.sum(weights[y_true == 0])
    
    loss  = eval(param['opt_param']['lossfunc'])
    # -------------------------------------------------------
    
    print(f'(eff_s: {eff_s:0.3E}, eff_b: {eff_b:0.3E}) | loss: {loss:0.3f} | AUC = {metrics.auc:0.4f}', 'yellow')
    
    if args['__raytune_running__']:
        #with ray.tune.checkpoint_dir(epoch) as checkpoint_dir:
        #    path = os.path.join(checkpoint_dir, "checkpoint")
        #    torch.save((model.state_dict(), optimizer.state_dict()), path)
        ray.train.report({'loss': loss, 'AUC': metrics.auc})
    else:
        ## Save
        True
        #checkpoint = {'model': model, 'state_dict': model.state_dict()}
        #torch.save(checkpoint, args['modeldir'] + f'/{param["label"]}_' + str(epoch) + '.pth')
    
    if not args['__raytune_running__']:
        
        return model_param

    return # No return value for raytune


def train_flr(config={'params': {}}, data_trn=None, args=None, param=None):
    """
    Train factorized likelihood model

    Args:
        See other train_*

    Returns:
        trained model
    """
    print(f'Training <{param["label"]}> classifier ...')

    savedir = aux.makedir(f'{args["modeldir"]}/{param["label"]}')
    
    b_pdfs, s_pdfs, bin_edges = flr.train(X = data_trn.x, y = data_trn.y, weights = data_trn.w, param = param)
    
    with open(f'{savedir}/{param["label"]}_0.pkl', 'wb') as file:
        data = {'b_pdfs': b_pdfs, 's_pdfs': s_pdfs, 'bin_edges': bin_edges}
        pickle.dump(data, file)

    return (b_pdfs, s_pdfs)


def train_flow(config={'params': {}}, data_trn=None, data_val=None, args=None, param=None):
    """
    Train normalizing flow (BNAF) neural model

    Args:
        See other train_*

    Returns:
        trained model
    """
    
    # Set input dimensions
    param['model_param']['n_dims'] = data_trn.x.shape[1]

    print(f'Training [{param["label"]}] classifier ...')
    
    for classid in range(len(args['primary_classes'])):
        
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
        
        # Custom wrapper
        from icenet.optim import scheduler
        sched = scheduler.ReduceLROnPlateau(optimizer,
                                      factor   = param['scheduler_param']['factor'],
                                      patience = param['scheduler_param']['patience'],
                                      cooldown = param['scheduler_param']['cooldown'],
                                      min_lr   = param['scheduler_param']['min_lr'],
                                      early_stopping = param['scheduler_param']['early_stopping'],
                                      threshold_mode = 'abs')
        
        print(f'Training density for class = {classid}', 'magenta')
        
        modeldir  = aux.makedir(f"{args['modeldir']}/{param['label']}")
        save_name = f'{param["label"]}_class_{classid}'
        
        dbnf.train(model=model, optimizer=optimizer, scheduler=sched,
            trn_x=trn.x, val_x=val.x, trn_weights=trn.w, val_weights=val.w,
            param=param, modeldir=modeldir, save_name=save_name)
    
    return True


def train_graph_xgb(config={'params': {}}, data_trn=None, data_val=None, trn_weights=None, val_weights=None,
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
    
    print(f'Training [{param["label"]}] classifier ...')

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
        print(f'Latent z-space dimension = {Z} auto-detected')
    
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
    
    print(f'After extension: {x_trn.shape}')

    # Create all feature names
    ids = []
    for i in range(Z):                  # Graph-net latent dimension Z (message passing output) features
        ids.append(f'conv_Z_{i}')
    for i in range(len(data_trn[0].u)): # Xgboost high-level features
        ids.append(feature_names[i])
    
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
        model_param.update({'num_class': len(args['primary_classes'])})

    del model_param['num_boost_round']
    # ---------------------------------------

    # Boosting iterations
    num_epochs = param['xgb']['model_param']['num_boost_round']
    
    savedir = aux.makedir(f"{args['modeldir']}/{param['xgb']['label']}")
    
    for epoch in range(0, num_epochs):
        
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
        
        if len(args['primary_classes']) >= 2:
            
            # AUC
            pred    = model.predict(dtrain)
            if len(pred.shape) > 1: pred = pred[:, args['signal_class']]
            metrics = aux.Metric(y_true=y_trn, y_pred=pred, weights=w_trn, class_ids=args['primary_classes'], hist=False, verbose=True)
            trn_aucs.append(metrics.auc)

            pred    = model.predict(deval)
            if len(pred.shape) > 1: pred = pred[:, args['signal_class']]
            metrics = aux.Metric(y_true=y_val, y_pred=pred, weights=w_val, class_ids=args['primary_classes'], hist=False, verbose=True)
            val_aucs.append(metrics.auc)

        # Loss
        trn_losses.append(results['train'][model_param['eval_metric'][0]][0])
        val_losses.append(results['eval'][model_param['eval_metric'][0]][0])

        ## Save
        losses = {'trn_losses': trn_losses, 'val_losses': val_losses, 'trn_aucs': trn_aucs, 'val_aucs': val_aucs}
        
        with open(savedir + f"{param['xgb']['label']}_{epoch}.pkl", 'wb') as file:
            pickle.dump({'model': model, 'ids': ids, 'losses': losses, 'param': param}, file, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f'Tree {epoch:03d}/{num_epochs:03d} | Train: loss = {trn_losses[-1]:0.4f}, AUC = {trn_aucs[-1]:0.4f} | Eval: loss = {val_losses[-1]:0.4f}, AUC = {val_aucs[-1]:0.4f}')
    
    # -------------------------------------------
    # Plot evolution
    plotdir  = aux.makedir(f'{args["plotdir"]}/train/{param["xgb"]["label"]}')
    fig,ax   = plots.plot_train_evolution_multi(losses={'train': trn_losses, 'validate': val_losses},
                    trn_aucs=trn_aucs, val_aucs=val_aucs, label=param['xgb']['label'])
    plt.savefig(f"{plotdir}/{param['xgb']['label']}--evolution.pdf", bbox_inches='tight'); plt.close()
    
    # -------------------------------------------
    ## Plot feature importance
    targetdir = aux.makedir(f'{args["plotdir"]}/train/xgboost-importance/{param["xgb"]["label"]}')
    
    for sort in [True, False]:
        fig,ax = plots.plot_xgb_importance(model=model, tick_label=ids, label=param['xgb']['label'], sort=sort)
        
        plt.savefig(f'{targetdir}/{param["xgb"]["label"]}--importance--sort-{sort}.pdf', bbox_inches='tight'); plt.close()
    
    # -------------------------------------------
    ## Plot decision trees
    
    if ('plot_trees' in param['xgb']) and param['xgb']['plot_trees']:
        
        targetdir = aux.makedir(f'{args["plotdir"]}/train/xgboost-treeviz/{param["xgb"]["label"]}')
        
        try:
            print(f'Plotting decision trees ...')
            model.feature_names = ids
            for i in tqdm(range(num_epochs)):
                xgboost.plot_tree(model, num_trees=i)
                fig = plt.gcf()
                fig.set_size_inches(60, 20) # Higher reso
                plt.savefig(f'{targetdir}/tree_{i}.pdf', bbox_inches='tight'); plt.close()
        except:
            print(f'Could not plot the decision trees (try: conda install python-graphviz)')
        
    model.feature_names = None # Set original default ones

    return model
