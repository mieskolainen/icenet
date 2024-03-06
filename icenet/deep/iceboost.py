# iceboost == xgboost + torch autograd based extensions
#
# m.mieskolainen@imperial.ac.uk, 2024

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import xgboost
import copy
from termcolor import cprint
from tqdm import tqdm
import pickle

# icenet
from icenet.tools import stx
from icenet.tools import aux
from icenet.tools import plots

from icenet.deep import autogradxgb
from icenet.deep import optimize

from icefit import mine, cortools

import ray
from ray import tune


def _hinge_loss(preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor=None):

    device = preds.device
    
    if weights is None:
        w = torch.ones(len(preds))
        w = w / torch.sum(w)
    else:
        w = weights / torch.sum(weights)
    
    # Set computing device
    targets = targets.to(device)
    w       = w.to(device)
    
    targets = 2 * targets - 1
    loss    = w * torch.max(torch.zeros_like(preds), 1 - preds * targets) ** 2
    
    return loss.sum() * len(preds)

def _binary_cross_entropy(preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor=None, EPS=1E-12):
    """
    Custom binary cross entropy loss with
    domain adaptation (DA) and mutual information (MI) regularization
    """
    global loss_mode
    global MI_x
    global BCE_param
    global MI_param
    global loss_history
    
    device = preds.device
    
    if weights is None:
        w = torch.ones(len(preds))
        w = w / torch.sum(w)
    else:
        w = weights / torch.sum(weights)
    
    # Set computing device
    targets = targets.type(torch.int32).to(device)
    w       = w.to(device)
    
    track_loss = {}
    loss_str   = ''
    
    # --------------------------------------------------------------------
    ## Sigmoid
    phat = torch.clip(1 / (1 + torch.exp(-preds)), min=EPS, max=1-EPS)
    
    ## Binary Cross Entropy terms
    BCE_loss = torch.tensor(0.0).to(device)
    
    for key in BCE_param.keys():
        param = BCE_param[key]
            
        # Set labels
        t0 = (targets == param['classes'][0])
        t1 = (targets == param['classes'][1])
        
        targets_CE = targets.clone()
        targets_CE[t0] = 0
        targets_CE[t1] = 1
        
        ### Now loop over all filter categories
        mask0 = param[f'evt_mask_{loss_mode}'][param['classes'][0]]
        mask1 = param[f'evt_mask_{loss_mode}'][param['classes'][1]]
        N_cat = mask0.shape[0]
        
        loss = torch.tensor(0.0).to(device)
        
        for m in range(N_cat):
            
            m0   = torch.tensor(mask0[m,:], dtype=torch.bool).to(device) # Pick index mask
            m1   = torch.tensor(mask1[m,:], dtype=torch.bool).to(device)
            
            wnew = torch.zeros_like(w).to(device) # ! important
            
            wnew[t0] = m0 * w[t0]; wnew[t0] = wnew[t0] / torch.sum(wnew[t0])
            wnew[t1] = m1 * w[t1]; wnew[t1] = wnew[t1] / torch.sum(wnew[t1])
            wnew     = wnew / torch.sum(wnew)
            
            # BCE
            CE   = - wnew * ((1-targets_CE)*torch.log(1-phat) + targets_CE*torch.log(phat))
            loss = loss + CE

        loss     = param["beta"] * loss.sum()        
        BCE_loss = BCE_loss + loss
        
        txt = f'{key} [$\\beta$ = {param["beta"]}]'
        track_loss[txt] = loss.item()
        loss_str       += f'{txt} = {loss.item():0.5f} | '
    
    # --------------------------------------------------------------------
    ## MI Regularization
    
    MI_loss = torch.tensor(0.0).to(device)
    
    if MI_param is not None:
        
        # Loop over chosen classes
        k      = 0
        values = []
        
        for c in MI_param['classes']:
            
            # -----------
            reg_param = copy.deepcopy(MI_param)
            reg_param['ma_eT'] = reg_param['ma_eT'][k] # Pick the one
            # -----------

            # Pick class indices
            cind  = (targets != None) if c == None else (targets == c)

            X     = torch.Tensor(MI_x).to(device)[cind].squeeze()
            Z     = preds[cind].squeeze()
            W     = weights[cind]
            mask  = reg_param[f'evt_mask_{loss_mode}'][c]
            
            # Total maximum is limited, pick subsample
            if reg_param['max_N'] is not None and X.shape[0] > reg_param['max_N']:
                X    = X[0:reg_param['max_N']]
                Z    = Z[0:reg_param['max_N']]
                W    = W[0:reg_param['max_N']]
                mask = mask[:,0:reg_param['max_N']]
            
            ### Now loop over all filter categories
            N_cat     = mask.shape[0]
            
            loss_this = torch.tensor(0.0).to(device)
            value     = torch.tensor(0.0).to(device)
            total_ww  = 0.0
            
            for m in range(N_cat):
                
                mm_ = torch.from_numpy(mask[m,:]).to(device) # Pick index mask

                # Apply target threshold (e.g. we are interested only in high score region)
                # First iterations might not yield any events passing this, 'min_count' will take care
                if reg_param['min_score'] is not None:
                    mm_ = mm_ & (Z > reg_param['min_score']) 
                
                # Minimum number of events per category cutoff
                if reg_param['min_count'] is not None and torch.sum(mm_) < reg_param['min_count']:
                    continue
                
                ## Non-Linear Distance Correlation
                if   reg_param['losstype'] == 'DCORR':
                    
                    value = value + cortools.distance_corr_torch(x=X[mm_], y=Z[mm_], weights=W[mm_])
                
                ## Linear Pearson Correlation (only for DEBUG)
                elif reg_param['losstype'] == 'PEARSON':
                    
                    if len(X.shape) > 1: # if multidim X
                        
                        for j in range(X.shape[-1]): # dim-by-dim against Z (BDT output)
                            
                            rho   = cortools.corrcoeff_weighted_torch(x=X[mm_, j], y=Z[mm_], weights=W[mm_])
                            triag = torch.triu(rho, diagonal=1) # upper triangle without diagonal
                            L     = torch.sum(torch.abs(triag))
                            value = value + L
                    else:
                        rho   = cortools.corrcoeff_weighted_torch(x=X[mm_], y=Z[mm_], weights=W[mm_])
                        triag = torch.triu(rho, diagonal=1)
                        L     = torch.sum(torch.abs(triag))
                        value = value + L

                ## Neural Mutual Information
                else:
                    
                    # We need .detach() here for Z!
                    model = mine.estimate(X=X[mm_], Z=Z[mm_].detach(), weights=W[mm_],
                        return_model_only=True, device=device, **reg_param)

                    # ------------------------------------------------------------
                    # Now (re)-apply the MI estimator to the sample
                    
                    # No .detach() here, we need the gradients wrt Z!
                    value = mine.apply_mine_batched(X=X[mm_], Z=Z[mm_], weights=W[mm_], model=model,
                                                losstype=reg_param['losstype'], batch_size=reg_param['eval_batch_size'])
                
                # Significance N/sqrt(N) = sqrt(N) weights based on Poisson stats
                if MI_param['poisson_weight']:
                    cat_ww = torch.sqrt(torch.sum(mm_))
                else:
                    cat_ww = 1.0
                
                if not torch.isfinite(value): # First boost iteration might yield bad values
                    value = torch.tensor(0.0).to(device)
                
                # Save
                loss_this = loss_this + MI_param['beta'][k] * value * cat_ww
                total_ww    += cat_ww

                values.append(np.round(value.item(), 6))
            
            # Finally add this to the total loss
            if total_ww > 0:
                MI_loss = MI_loss + loss_this / total_ww
            
            k += 1
        
        cprint(f'RAW {reg_param["losstype"]} = {values}', 'yellow')
        
        txt = f'{reg_param["losstype"]} [$\\beta$ = {MI_param["beta"]}]'
        track_loss[txt] = MI_loss.item()
        loss_str       += f'{txt} = {MI_loss.item():0.5f}'
    
    # --------------------------------------------------------------------
    # Total loss
    
    total_loss        = BCE_loss + MI_loss
    track_loss['sum'] = total_loss.item()
    
    # --------------------------------------------------------------------
    # Track losses
    
    loss_str = f'Loss[{loss_mode}]: sum: {total_loss:0.5f} | ' + loss_str
    cprint(loss_str, 'yellow')
    optimize.trackloss(loss=track_loss, loss_history=loss_history)
    
    # --------------------------------------------------------------------
    
    # Scale finally to the total number of events (to conform with xgboost internal convention)
    return total_loss * len(preds)


def create_filters(param, data_trn, data_val):
    
    # Create filter masks
    param['evt_mask_train'] = {}
    param['evt_mask_eval']  = {}

    for c in param['classes']: # per chosen class
        
        for mode in ['train', 'eval']:
        
            print(f'class[{c}] ({mode}):')
            
            if mode == 'train':
                data = data_trn
            else:
                data = data_val
            
            # Pick class indices
            cind = (data.y != None) if c == None else (data.y == c)
            
            # Per filter category
            if 'set_filter' in param:
                mask, text, path = stx.filter_constructor(
                    filters=param['set_filter'],
                    X=data.x[cind,...],
                    ids=data_trn.ids,
                    y=data.y[cind])
            
            # All inclusive
            else:
                mask = np.ones((1, len(data.x[cind,...])), dtype=int)
                text = ['inclusive']
            
            stx.print_stats(mask=mask, text=text)
                
            # Save the mask
            param[f'evt_mask_{mode}'][c] = copy.deepcopy(mask)    
    
    return param


def train_xgb(config={'params': {}}, data_trn=None, data_val=None, y_soft=None, args=None, param=None,
              plot_importance=True, data_trn_MI=None, data_val_MI=None):
    """
    Train XGBoost model
    
    Args:
        See other train_* under train.py
    
    Returns:
        trained model
    """
    
    global MI_x
    global BCE_param
    global MI_param    
    global loss_mode
    global loss_history
    
    MI_x         = None
    BCE_param    = None
    MI_param     = None
    loss_mode    = None
    loss_history = {}
    
    loss_history_train = {}
    loss_history_eval  = {}

    if 'BCE_param' in param:
        
        BCE_param = {}
        
        for key in param['BCE_param'].keys():
            
            cprint(__name__ + f'.train_xgb: Setting BCE event filters [{key}]', 'green')

            BCE_param[key] = create_filters(param=param['BCE_param'][key], data_trn=data_trn, data_val=data_val)

    if 'MI_param' in param:
        
        cprint(__name__ + f'.train_xgb: Setting MI event filters', 'green')
        
        MI_param = copy.deepcopy(param['MI_param']) #! important
        MI_param = create_filters(param=MI_param, data_trn=data_trn, data_val=data_val)
    
    # ---------------------------------------------------

    if param['model_param']['device'] == 'auto':
        param['model_param'].update({'device': 'cuda' if torch.cuda.is_available() else 'cpu'})
    
    print(__name__ + f'.train_xgb: Training <{param["label"]}> classifier ...')

    ### ** Optimization hyperparameters [possibly from Raytune] **
    param['model_param'] = aux.replace_param(default=param['model_param'], raytune=config['params'])
    
    ### *********************************
    
    # Normalize weights to sum to number of events (xgboost library has no scale normalization)
    w_trn     = data_trn.w / np.sum(data_trn.w) * data_trn.w.shape[0]
    w_val     = data_val.w / np.sum(data_val.w) * data_val.w.shape[0]

    X_trn, ids_trn = aux.red(data_trn.x, data_trn.ids, param) # variable reduction
    dtrain    = xgboost.DMatrix(data=X_trn, label = data_trn.y if y_soft is None else y_soft, weight = w_trn, feature_names=ids_trn)
    
    X_val, ids_val = aux.red(data_val.x, data_val.ids, param) # variable reduction
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
        model_param.update({'num_class': len(args['primary_classes'])})
    
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
            
            strs   = model_param['objective'].split(':')
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu:0')

            # !
            MI_x         = copy.deepcopy(data_trn_MI)
            loss_history = copy.deepcopy(loss_history_train)
            
            if strs[1] == 'binary_cross_entropy':
                
                a['obj'] = autogradxgb.XgboostObjective(loss_func=_binary_cross_entropy, skip_hessian=True, device=device)
                a['params']['disable_default_eval_metric'] = 1
            
            else:
                raise Exception(__name__ + f'.train_xgb: Unknown custom loss {strs[1]}')

            #!
            del a['params']['eval_metric']
            del a['params']['objective']
        # -----------------

        if epoch > 0: # Continue from the previous epoch model
            a['xgb_model'] = model

        # Train
        loss_mode = 'train'
        model = xgboost.train(**a)
        loss_history_train = copy.deepcopy(loss_history)
        
        # Validate
        if 'custom' in model_param['objective']:
            loss_mode    = 'eval'
            MI_x         = copy.deepcopy(data_val_MI)
            loss_history = copy.deepcopy(loss_history_eval)
            
            # Loss history is updated inside the loss
            loss              = a['obj'](preds=model.predict(deval), targets=deval)[1] / len(data_val.x)
            loss_history_eval = copy.deepcopy(loss_history)
        
        # ------ Loss values ------
        if 'custom' in model_param['objective']:
            trn_losses.append(loss_history_train['sum'][-1])
            val_losses.append(loss_history_eval['sum'][-1])
        else:
            trn_losses.append(results['train'][model_param['eval_metric'][0]][0])
            val_losses.append(results['eval'][model_param['eval_metric'][0]][0])
        # -------------------------
        
        # ------- AUC values ------
        
        pred = model.predict(dtrain)
        if len(pred.shape) > 1: pred = pred[:, args['signal_class']]
        metrics = aux.Metric(y_true=data_trn.y, y_pred=pred, weights=w_trn, class_ids=args['primary_classes'], hist=False, verbose=True)
        trn_aucs.append(metrics.auc)
        
        pred = model.predict(deval)
        if len(pred.shape) > 1: pred = pred[:, args['signal_class']]
        metrics = aux.Metric(y_true=data_val.y, y_pred=pred, weights=w_val, class_ids=args['primary_classes'], hist=False, verbose=True)
        val_aucs.append(metrics.auc)
        # -------------------------
        
        print(__name__ + f'.train_xgb: Tree {epoch+1:03d}/{max_num_epochs:03d} | Train: loss = {trn_losses[-1]:0.4f}, AUC = {trn_aucs[-1]:0.4f} | Eval: loss = {val_losses[-1]:0.4f}, AUC = {val_aucs[-1]:0.4f}')
        
        if args['__raytune_running__']:
            #with tune.checkpoint_dir(epoch) as checkpoint_dir:
            #    path = os.path.join(checkpoint_dir, "checkpoint")
            #    pickle.dump(model, open(path, 'wb'))
            ray.train.report({'loss': trn_losses[-1], 'AUC': val_aucs[-1]})
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
            
            ltr = {f'train:    {k}': v for k, v in loss_history_train.items()}
            lev = {f'validate: {k}': v for k, v in loss_history_eval.items()}
            
            fig,ax = plots.plot_train_evolution_multi(losses=ltr | lev, trn_aucs=trn_aucs, val_aucs=val_aucs, label=param["label"])
        plt.savefig(f'{plotdir}/{param["label"]}--evolution.pdf', bbox_inches='tight'); plt.close()
        
        ## Plot feature importance
        if plot_importance:
            for sort in [True, False]:
                for importance_type in ['weight', 'gain', 'cover', 'total_gain', 'total_cover']:
                    fig,ax = plots.plot_xgb_importance(model=model, 
                        tick_label=aux.red(data_trn.x, data_trn.ids, param, 'ids'),
                        label=param["label"], importance_type=importance_type, sort=sort)
                    targetdir = aux.makedir(f'{args["plotdir"]}/train/xgboost-importance')
                    plt.savefig(f'{targetdir}/{param["label"]}--type_{importance_type}--sort-{sort}.pdf', bbox_inches='tight');
                    plt.close()
        
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
