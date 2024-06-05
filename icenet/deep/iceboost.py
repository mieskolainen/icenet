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
from icenet.tools import reweight

from icenet.deep import autogradxgb
from icenet.deep import optimize
from icenet.deep import tempscale

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


def BCE_loss_with_logits(input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
    """
    Numerically stable BCE loss with logits
    https://medium.com/@sahilcarterr/why-nn-bcewithlogitsloss-numerically-stable-6a04f3052967
    """
    max_val = (-input).clamp_min(0)
    loss    = (1 - target).mul(input).add(max_val).add((-max_val).exp().add((-input - max_val).exp()).log())

    if weights is not None:
        loss.mul_(weights)
    
    return loss


def _binary_cross_entropy(preds: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor=None, EPS=1E-12):
    """
    Custom binary cross entropy loss with
    domain adaptation (DA) and mutual information (MI) regularization
    
    Negative weights are supported via 'out_weights' variable (update it for each train / eval)
    """
    global loss_mode
    global MI_x
    global BCE_param
    global MI_param
    global track_loss
    global out_weights

    device = preds.device
    
    if  out_weights is not None: # Feed in weights outside
        w = torch.from_numpy(out_weights)
        w = w / torch.sum(w)
    elif weights is None:
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
    ## Binary Cross Entropy terms
    
    BCE_loss = torch.tensor(0.0).to(device)
    
    for key in BCE_param.keys():
        
        w_this = w.clone()
        
        param  = BCE_param[key]
        
        # Set labels
        t0 = (targets == param['classes'][0])
        t1 = (targets == param['classes'][1])
        
        # Check that there are some events
        if torch.sum(t0) < 1:
            print(__name__  + f": BCE[{key}] No events from class [{param['classes'][0]}] - skip loss term")
            continue
        
        if torch.sum(t1) < 1:
            print(__name__  + f": BCE[{key}] No events from class [{param['classes'][1]}] - skip loss term")
            continue
        
        # Now re-weight the target(s) based on the model predictions
        # (a specific type of regularization -- experimental)
        if 'AIRW' in param:
            
            for i in param['AIRW']['classes']:
                
                t_i     = (targets == param['AIRW']['classes'][i])
                RW_mode = param['AIRW']['RW_modes'][i]
                
                # Multiply existing weights
                p_map        = reweight.rw_transform_with_logits(preds[t_i], mode=RW_mode)
                w_this[t_i] *= p_map
        
        targets_CE = targets.clone()
        targets_CE[t0] = 0
        targets_CE[t1] = 1
        
        ### Now loop over all filter categories
        mask0 = param[f'evt_mask_{loss_mode}'][param['classes'][0]]
        mask1 = param[f'evt_mask_{loss_mode}'][param['classes'][1]]
        N_cat = mask0.shape[0]
        
        loss  = torch.tensor(0.0).to(device)
        
        for m in range(N_cat):
            
            # Pick 0-1 hot index mask
            m0     = torch.tensor(mask0[m,:], dtype=torch.int32).to(device)
            m1     = torch.tensor(mask1[m,:], dtype=torch.int32).to(device)
            wfinal = torch.zeros_like(w).to(device) # ! important to init with zeros
            
            # Event weights
            wfinal[t0] = m0 * w_this[t0]; wfinal[t0] = wfinal[t0] / torch.sum(wfinal[t0])
            wfinal[t1] = m1 * w_this[t1]; wfinal[t1] = wfinal[t1] / torch.sum(wfinal[t1])
            wfinal     = wfinal / torch.sum(wfinal)
            
            # BCE
            CE   = BCE_loss_with_logits(input=preds, target=targets_CE, weights=wfinal)
            loss = loss + CE
        
        loss     = param["beta"] * loss.sum()        
        BCE_loss = BCE_loss + loss
        
        txt = f'{key} [$\\beta$ = {param["beta"]}]'
        track_loss[txt] = loss.item()
        loss_str       += f'{txt} = {loss.item():0.5f} | '
    
    # ------------
    # Temperature post-calibration
    if loss_mode == 'eval':
        
        ts = tempscale.LogitsWithTemperature(mode='binary', device=device)
        ts.set_temperature(logits=preds, labels=targets.to(torch.float32), weights=w)
    
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
            
            mask  = reg_param[f'evt_mask_{loss_mode}'][c]
            
            # Pick class indices
            cind  = (targets != None) if c == None else (targets == c)

            # Map predictions to [0,1]
            Z     = torch.clip(torch.sigmoid(preds[cind]).squeeze(), EPS, 1-EPS)
            X     = torch.Tensor(MI_x).to(device)[cind].squeeze()
            W     = w[cind]
            
            # Total maximum is limited (for DCORR), pick random subsample
            if reg_param['losstype'] == 'DCORR' and reg_param['max_N'] is not None and X.shape[0] > reg_param['max_N']:
                r    = np.random.choice(len(X), reg_param['max_N'], replace=False)
                X    = X[r]
                Z    = Z[r]
                W    = W[r]
                mask = mask[:,r]
            
            ### Now loop over all filter categories
            N_cat     = mask.shape[0]
            
            loss_this = torch.tensor(0.0).to(device)
            value     = torch.tensor(0.0).to(device)
            total_ww  = 0.0
            
            for m in range(N_cat):
                
                # This needs to be done, otherwise torch doesn't understand indexing!
                mm_ = np.array(mask[m,:], dtype=bool)
                
                # Apply target threshold (e.g. we are interested only in high score region)
                # First iterations might not yield any events passing this, 'min_count' will take care
                if reg_param['min_score'] is not None:
                    mm_ = mm_ & (Z.detach().cpu().numpy() > reg_param['min_score']) 
                
                # Minimum number of events per category cutoff
                if reg_param['min_count'] is not None and np.sum(mm_) < reg_param['min_count']:
                    cprint(__name__ + f" MI_reg: {np.sum(mm_)} < {reg_param['min_count']} = reg_param['min_count'] (class [{c}] | category [{m}]) -- skip", 'red')
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
                    cat_ww = np.sqrt(np.sum(mm_))
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
    
    # Print
    loss_str = f'Loss[{loss_mode}]: sum: {total_loss.item():0.5f} | ' + loss_str
    cprint(loss_str, 'yellow')
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
                mask = np.ones((1, len(data.x[cind,...])), dtype=bool)
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
    global track_loss
    global out_weights

    MI_x        = None
    BCE_param   = None
    MI_param    = None
    loss_mode   = None
    out_weights = None
    
    loss_history_train = {}
    loss_history_eval  = {}

    # TensorboardX
    if not args['__raytune_running__'] and param['tensorboard']:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(os.path.join(args['modeldir'], param['label']))
    
    
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

    # Activate custom-loss mode
    use_custom = True if 'custom' in param['model_param']['objective'] else False
    
    ### *********************************
    
    # Normalize weights to sum to the number of events (xgboost library has no scale normalization)
    w_trn = data_trn.w / np.sum(data_trn.w) * data_trn.w.shape[0]
    w_val = data_val.w / np.sum(data_val.w) * data_val.w.shape[0]

    # ---------------------------------------------------------
    # Choose weight mode
    if np.min(w_trn) < 0.0 or np.min(w_val) < 0.0:
        print(__name__ + f'.train_xgb: Negative weights in the sample -- handled via custom loss')
        out_weights_on = True
        
        if not use_custom:
            raise Exception(__name__ + f'.train_xgb: Need to use custom with negative weights, e.g. "custom:binary_cross_entropy". Change your parameters.')
    else:
        out_weights_on = False
    # ---------------------------------------------------------
    
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
    num_epochs = param['model_param']['num_boost_round']
    
    # Prepare input
    X_trn, ids_trn = aux.red(X=data_trn.x, ids=data_trn.ids, param=param, verbose=True)  # variable reduction
    X_val, ids_val = aux.red(X=data_val.x, ids=data_val.ids, param=param, verbose=False) # variable reduction
    
    for epoch in range(0, num_epochs):
        
        # Create input xgboost frames
        dtrain = xgboost.DMatrix(data=X_trn, label = data_trn.y if y_soft is None else y_soft, weight = w_trn if not out_weights_on else None, feature_names=ids_trn)    
        deval  = xgboost.DMatrix(data=X_val, label = data_val.y,  weight = w_val if not out_weights_on else None, feature_names=ids_val)
        
        ## What to evaluate
        if epoch == 0 or (epoch % param['evalmode']) == 0 or args['__raytune_running__']:
            evallist = [(dtrain, 'train'), (deval, 'eval')]
        else:
            evallist = [(dtrain, 'train')]
        
        ## Prepare parameters
        results = dict()
        
        a = {'params':          copy.deepcopy(model_param),
             'dtrain':          dtrain,
             'num_boost_round': 1,
             'evals':           evallist,
             'evals_result':    results,
             'verbose_eval':    False}
        
        # ==============================================
        ## Train
        
        if use_custom:
            
            strs   = model_param['objective'].split(':')
            device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu:0')

            # !
            loss_mode = 'train'
            MI_x      = copy.deepcopy(data_trn_MI)
            
            if strs[1] == 'binary_cross_entropy':
                
                if len(strs) == 3 and 'hessian' in strs[2]:
                    print('Using Hessian with custom loss')
                    skip_hessian = False
                else:
                    skip_hessian = True

                a['obj'] = autogradxgb.XgboostObjective(loss_func=_binary_cross_entropy, skip_hessian=skip_hessian, device=device)
                a['params']['disable_default_eval_metric'] = 1
            
            else:
                raise Exception(__name__ + f'.train_xgb: Unknown custom loss {strs[1]}')
            
            #!
            del a['params']['eval_metric']
            del a['params']['objective']
        # -----------------

        if epoch > 0: # Continue from the previous epoch model
            a['xgb_model'] = model
        
        if out_weights_on:
            out_weights = copy.deepcopy(w_trn)

        model = xgboost.train(**a)
        
        if use_custom:
            track_loss_train = copy.deepcopy(track_loss) # track_loss from custom loss
        else:
            train_loss = results['train'][model_param['eval_metric'][0]][0]
        
        # ==============================================
        ## Validate
        if epoch == 0 or (epoch % param['evalmode']) == 0 or args['__raytune_running__']:
            
            # ------- AUC values ------
            if len(args['primary_classes']) >= 2:
                
                preds_train = model.predict(dtrain)
                if len(preds_train.shape) > 1: preds_train = preds_train[:, args['signal_class']]
                metrics_train = aux.Metric(y_true=data_trn.y, y_pred=preds_train, weights=w_trn, class_ids=args['primary_classes'], hist=False, verbose=True)
                
                preds_eval = model.predict(deval)
                if len(preds_eval.shape) > 1: preds_eval = preds_eval[:, args['signal_class']]
                metrics_eval = aux.Metric(y_true=data_val.y, y_pred=preds_eval, weights=w_val, class_ids=args['primary_classes'], hist=False, verbose=True)
        
            # ------- Loss values ------
            if use_custom:
                if out_weights_on:
                    out_weights = copy.deepcopy(w_val)    
                
                # !
                loss_mode = 'eval'
                MI_x      = copy.deepcopy(data_val_MI)
                
                a['obj'](preds=preds_eval, targets=deval)[1] / len(data_val.x)
                track_loss_eval = copy.deepcopy(track_loss)  # track_loss from custom loss
            
            else:
                eval_loss = results['eval'][model_param['eval_metric'][0]][0] # Collect the value
        
        # ==============================================
        # Collect values
        if use_custom:
            optimize.trackloss(loss=track_loss_train, loss_history=loss_history_train)
            optimize.trackloss(loss=track_loss_eval,  loss_history=loss_history_eval)

            trn_losses.append(loss_history_train['sum'][-1]) # For raytune
            val_losses.append(loss_history_eval['sum'][-1])
        else:
            trn_losses.append(train_loss)
            val_losses.append(eval_loss)    
        
        if len(args['primary_classes']) >= 2:
            
            trn_aucs.append(metrics_train.auc)
            val_aucs.append(metrics_eval.auc)
        # ==============================================
        
        if not args['__raytune_running__'] and param['tensorboard']:
            #writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
            writer.add_scalar('loss/validation', val_losses[-1], epoch)
            writer.add_scalar('loss/train',      trn_losses[-1], epoch)
            writer.add_scalar('AUC/validation',  val_aucs[-1],   epoch)
            writer.add_scalar('AUC/train',       trn_aucs[-1],   epoch)
        
        print(__name__ + f'.train_xgb [{param["label"]}] Tree {epoch:03d}/{num_epochs:03d} | Train: loss = {trn_losses[-1]:0.4f}, AUC = {trn_aucs[-1]:0.4f} | Eval: loss = {val_losses[-1]:0.4f}, AUC = {val_aucs[-1]:0.4f}')
        
        if not args['__raytune_running__']:
            
            ## Save the model
            filename = args['modeldir'] + f'/{param["label"]}_{epoch}'
            
            model.save_model(filename + '.json')
            model.dump_model(filename + '.text', dump_format='text')

            losses = {'trn_losses':         trn_losses,
                      'val_losses':         val_losses,
                      'trn_aucs':           trn_aucs,
                      'val_aucs:':          val_aucs,
                      'loss_history_train': loss_history_train,
                      'loss_history_eval':  loss_history_eval}
            
            with open(filename + '.pkl', 'wb') as file:
                pickle.dump({'model': model, 'ids': ids_trn, 'losses': losses, 'epoch': epoch}, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    # Report only once after all boost iterations
    # otherwise early stopping may happen due to scheduler as with neural net epochs
    if args['__raytune_running__']:
        #with tune.checkpoint_dir(epoch) as checkpoint_dir:
        #    path = os.path.join(checkpoint_dir, "checkpoint")
        #    pickle.dump(model, open(path, 'wb'))
        ray.train.report({'loss': trn_losses[-1], 'AUC': val_aucs[-1]})
    
    if not args['__raytune_running__']:
        
        # Plot evolution
        plotdir = aux.makedir(f'{args["plotdir"]}/train/loss/{param["label"]}')
        
        if use_custom:
            ltr = {f'train: {k}': v for k, v in loss_history_train.items()}
            lev = {f'eval:  {k}': v for k, v in loss_history_eval.items()}

            losses_ = ltr | lev
        
        else:
            losses_ = {'train': trn_losses, 'eval': val_losses}
        
        for yscale in ['linear', 'log']:
            for xscale in ['linear', 'log']:
                
                fig,ax = plots.plot_train_evolution_multi(losses=losses_, trn_aucs=trn_aucs, val_aucs=val_aucs,
                                                          label=param["label"], yscale=yscale, xscale=xscale)
                
                plt.savefig(f"{plotdir}/{param['label']}_losses_yscale_{yscale}_xscale_{xscale}.pdf", bbox_inches='tight')
                plt.close(fig)
        
        ## Plot feature importance
        if plot_importance:
            for sort in [True, False]:
                for importance_type in ['weight', 'gain', 'cover', 'total_gain', 'total_cover']:
                    fig,ax = plots.plot_xgb_importance(model=model, tick_label=ids_trn,
                        label=param["label"], importance_type=importance_type, sort=sort)
                    targetdir = aux.makedir(f'{args["plotdir"]}/train/xgboost-importance')
                    plt.savefig(f'{targetdir}/{param["label"]}--type_{importance_type}--sort-{sort}.pdf', bbox_inches='tight');
                    plt.close()
        
        ## Plot decision trees
        if ('plot_trees' in param) and param['plot_trees']:
            try:
                print(__name__ + f'.train_xgb: Plotting decision trees ...')
                model.feature_names = ids_trn # Make it explicit
                for i in tqdm(range(num_epochs)):
                    xgboost.plot_tree(model, num_trees=i)
                    fig = plt.gcf(); fig.set_size_inches(60, 20) # Higher reso
                    path = aux.makedir(f'{targetdir}/trees_{param["label"]}')
                    plt.savefig(f'{path}/tree-{i}.pdf', bbox_inches='tight'); plt.close()
            except:
                print(__name__ + f'.train_xgb: Could not plot the decision trees (try: conda install python-graphviz)')
        
        #model.feature_names = None # Set original default ones
        
        return model
    
    return # No return value for raytune
