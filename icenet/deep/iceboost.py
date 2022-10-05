# iceboost == xgboost + torch autograd based extensions
#
# m.mieskolainen@imperial.ac.uk, 2022

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric
import xgboost
import copy
from termcolor import cprint
from tqdm import tqdm
import pickle

# icenet
from icenet.tools import stx
from icenet.tools import io
from icenet.tools import aux
from icenet.tools import aux_torch
from icenet.tools import plots
from icenet.tools import prints

from icenet.deep import autogradxgb
from icenet.deep import optimize
from icenet.deep.train import train_torch_graph

from icefit import mine

# Raytune
from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial


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
    MI_loss = torch.tensor(0.0).to(preds.device)
    MI_lb_values = []

    for c in MI_reg_param['classes']:

        # -----------
        reg_param = copy.deepcopy(MI_reg_param)
        reg_param['ma_eT'] = reg_param['ma_eT'][k] # Pick the one
        # -----------

        # Pick class indices
        cind  = (targets != None) if c == None else (targets == c)

        X     = torch.Tensor(MI_x).to(preds.device)[cind]
        Z     = preds[cind]
        W     = weights[cind]

        ### Now loop over all powerset categories
        mask  = reg_param['evt_mask'][c]
        N_cat = mask.shape[0]

        MI_loss_this = torch.tensor(0.0).to(preds.device)
        MI_lb        = torch.tensor(0.0).to(preds.device)
        total_ww     = 0.0

        for m in range(N_cat):

            mm_ = mask[m,:] # Pick index mask

            if np.sum(mm_) > 32: # Minimum number of events per category cutoff

                # We need .detach() here for Z!
                model = mine.estimate(X=X[mm_], Z=Z[mm_].detach(), weights=W[mm_],
                    return_model_only=True, device=X.device, **reg_param)
                
                # ------------------------------------------------------------
                # Now apply the MI estimator to the sample
                
                # No .detach() here, we need the gradients for Z!
                MI_lb = mine.apply_in_batches(X=X[mm_], Z=Z[mm_], weights=W[mm_], model=model, losstype=reg_param['losstype'])
                # ------------------------------------------------------------
                
                # Significance N/sqrt(N) = sqrt(N) weights based on Poisson stats
                cat_ww       = np.sqrt(np.sum(mm_))
                
                # Save
                MI_loss_this = MI_loss_this + MI_reg_param['beta'][k] * MI_lb * cat_ww
                total_ww    += cat_ww

            MI_lb_values.append(np.round(MI_lb.item(), 4))

        # Finally add this to the total loss
        MI_loss = MI_loss + MI_loss_this / total_ww

        k += 1

    ## Total loss
    total_loss = CE_loss + MI_loss
    cprint(f'Loss: Total = {total_loss:0.4f} | CE = {CE_loss:0.4f} | MI x beta {reg_param["beta"]} = {MI_loss:0.4f}', 'yellow')
    cprint(f'MI_lb = {MI_lb_values}', 'yellow')
    
    loss = {'sum': total_loss.item(), 'CE': CE_loss.item(), f'MI x $\\beta = {MI_reg_param["beta"]}$': MI_loss.item()}
    optimize.trackloss(loss=loss, loss_history=loss_history)
    
    # Scale finally to the total number of events (to conform with xgboost internal convention)
    return total_loss * len(preds)


def train_xgb(config={}, data_trn=None, data_val=None, y_soft=None, args=None, param=None, plot_importance=True,
    data_trn_MI=None, data_val_MI=None):
    """
    Train XGBoost model
    
    Args:
        See other train_* under train.py
    
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
        # Create powerset masks
        MI_reg_param['evt_mask'] = [None]*len(MI_reg_param['classes'])

        for c in MI_reg_param['classes']:
            cind = (data_trn.y != None) if c == None else (data_trn.y == c)

            # Per powerset category
            if 'set_filter' in MI_reg_param:
                mask, text, path = stx.filter_constructor(
                    filters=MI_reg_param['set_filter'], X=data_trn.x[cind,...], ids=data_trn.ids)
            # All inclusive
            else:
                mask = np.ones((1, len(data_trn.x[cind,...])), dtype=int)
            
            # Save the mask
            MI_reg_param['evt_mask'][c] = copy.deepcopy(mask)

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

    X_trn, ids_trn = aux.red(data_trn.x, data_trn.ids, param)
    dtrain    = xgboost.DMatrix(data=X_trn, label = data_trn.y if y_soft is None else y_soft, weight = w_trn, feature_names=ids_trn)
    
    X_val, ids_val = aux.red(data_val.x, data_val.ids, param)
    deval     = xgboost.DMatrix(data=X_val, label = data_val.y,  weight = w_val, feature_names=ids_val)
    
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
        plt.savefig(f'{plotdir}/{param["label"]}--evolution.pdf', bbox_inches='tight'); plt.close()
        
        ## Plot feature importance
        if plot_importance:
            for sort in [True, False]:
                fig,ax = plots.plot_xgb_importance(model=model, tick_label=aux.red(data_trn.x, data_trn.ids, param, 'ids'), label=param["label"], sort=sort)
                targetdir = aux.makedir(f'{args["plotdir"]}/train/xgboost-importance')
                plt.savefig(f'{targetdir}/{param["label"]}--importance--sort-{sort}.pdf', bbox_inches='tight'); plt.close()
        
        ## Plot decision trees
        if ('plot_trees' in param) and param['plot_trees']:
            try:
                print(__name__ + f'.train_xgb: Plotting decision trees ...')
                model.feature_names = aux.red(data_trn.x, data_trn.ids, param, 'ids')
                for i in tqdm(range(max_num_epochs)):
                    xgboost.plot_tree(model, num_trees=i)
                    fig = plt.gcf(); fig.set_size_inches(60, 20) # Higher reso
                    path = aux.makedir(f'{targetdir}/trees_{param["label"]}')
                    plt.savefig(f'{path}/tree-{i}.pdf', bbox_inches='tight'); plt.close()
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
    fig,ax   = plots.plot_train_evolution_multi(losses={'train': trn_losses, 'validate': val_losses},
                    trn_aucs=trn_aucs, val_aucs=val_aucs, label=param['xgb']['label'])
    plt.savefig(f"{plotdir}/{param['xgb']['label']}--evolution.pdf", bbox_inches='tight'); plt.close()
    
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
        targetdir = aux.makedir(f'{args["plotdir"]}/train/xgboost-importance')
        plt.savefig(f'{targetdir}/{param["label"]}--importance--sort-{sort}.pdf', bbox_inches='tight'); plt.close()
        
    ## Plot decision trees
    if ('plot_trees' in param['xgb']) and param['xgb']['plot_trees']:
        try:
            print(__name__ + f'.train_graph_xgb: Plotting decision trees ...')
            model.feature_names = ids
            for i in tqdm(range(max_num_epochs)):
                xgboost.plot_tree(model, num_trees=i)
                fig = plt.gcf(); fig.set_size_inches(60, 20) # Higher reso
                path = aux.makedir(f'{targetdir}/trees_{param["label"]}')
                plt.savefig(f'{path}/tree-{i}.pdf', bbox_inches='tight'); plt.close()
        except:
            print(__name__ + f'.train_graph_xgb: Could not plot the decision trees (try: conda install python-graphviz)')
        
    model.feature_names = None # Set original default ones

    return model
