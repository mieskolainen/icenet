# Deep Learning optimization functions
# 
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
from termcolor import colored,cprint

import torch
import torch.nn as nn

from icenet.deep import losstools
from icenet.deep import deeptools
from icenet.tools import aux
from icenet.tools import io


class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, Y, W, Y_DA=None, W_DA=None, X_MI=None):
        """ Initialization """
        self.x = X
        self.y = Y
        self.w = W

        self.y_DA = Y_DA  # Domain adaptation
        self.w_DA = W_DA

        self.x_MI = X_MI  # MI reg

    def __len__(self):
        """ Return the total number of samples """
        return self.y.shape[0]

    def __getitem__(self, index):
        """ Generates one sample of data """

        # Use ellipsis ... to index over scalar [,:], vector [,:,:], tensor [,:,:,..,:] indices
        out = {'x': self.x[index,...], 'y': self.y[index, ...], 'w': self.w[index, ...]}

        if self.y_DA is not None:
            out['y_DA'] = self.y_DA[index, ...]
            out['w_DA'] = self.w_DA[index, ...]

        if self.x_MI is not None:
            out['x_MI'] = self.x_MI[index, ...]

        return out

class DualDataset(torch.utils.data.Dataset):

    def __init__(self, X, U, Y, W, Y_DA=None, W_DA=None, X_MI=None):
        """ Initialization """
        self.x = X  # e.g. image tensors
        self.u = U  # e.g. global features 
        
        self.y = Y
        self.w = W

        self.y_DA = Y_DA  # Domain adaptation
        self.w_DA = W_DA
    
        self.x_MI = X_MI  # MI reg

    def __len__(self):
        """ Return the total number of samples """
        return self.y.shape[0]

    def __getitem__(self, index):
        """ Generates one sample of data """
        # Use ellipsis ... to index over scalar [,:], vector [,:,:], tensor [,:,:,..,:] indices

        out = {'x': self.x[index,...], 'u': self.u[index,...], 'y': self.y[index, ...], 'w': self.w[index, ...]}

        if self.y_DA is not None:
            out['y_DA'] = self.y_DA[index, ...]
            out['w_DA'] = self.w_DA[index, ...]
        
        if self.x_MI is not None:
            out['x_MI'] = self.x_MI[index, ...]

        return out

def dict_batch_to_cuda(batch, device):
    """
    Transfer to (GPU) device memory
    """
    for key in batch.keys():
        batch[key] = batch[key].to(device, non_blocking=True)

    return batch

def batch2tensor(batch, device):
    """
    Transform batch objects to a correct device
    """

    # Dataset or Dualdataset
    if type(batch) is dict:
        batch = dict_batch_to_cuda(batch, device)
        return batch

    # Pytorch geometric type
    else:
        batch = batch.to(device)
        return batch

def printloss(loss, precision=5):
    out = ''
    for key in loss.keys():
        out += f'{key}: {np.round(loss[key], precision)}, '
    return out

def trackloss(loss, loss_history):
    for key in loss.keys():
        if key not in loss_history: # First iteration
            loss_history[key] = [loss[key]]
        else:
            loss_history[key].append(loss[key])

def train(model, loader, optimizer, device, opt_param, MI=None):
    """
    Pytorch based training routine.
    
    Args:
        model     : pytorch geometric model
        loader    : pytorch geometric dataloader
        optimizer : pytorch optimizer
        device    : 'cpu' or 'device'
        opt_param : optimization parameters
    
    Returns
        trained model (return implicit via input arguments)
    """

    DA_active = True if (hasattr(model, 'DA_active') and model.DA_active) else False

    component_losses = {}
    total_loss       = 0
    n_batches        = 0

    # --------------------------------------------------------------------
    ## First step: Main network training
    
    model.train() #!

    # At this stage, we evaluate MI models !!
    if MI is not None:
        for k in range(len(MI['model'])):
            MI['model'][k].eval()

    for i, batch in enumerate(loader):

        batch_ = batch2tensor(batch, device)

        # -----------------------------------------
        # Torch models
        if type(batch_) is dict:
            x,y,w = batch_['x'], batch_['y'], batch_['w']
            
            if 'u' in batch_: # Dual input models
                x = {'x': batch_['x'], 'u': batch_['u']}

            #if DA_active:
            #    y_DA,w_DA = batch_['y_DA'], batch_['w_DA']
            if MI is not None:
                MI['x'] = batch_['x_MI']

        # Torch-geometric models
        else:
            x,y,w = batch_, batch_.y, batch_.w
            #if DA_active:
            #    y_DA,w_DA = batch_.y_DA, batch_.w_DA
            if MI is not None:
                MI['x'] = batch_.x_MI
        # -----------------------------------------
        
        #if DA_active:
        #    loss_tuple = losstools.loss_wrapper(model=model, x=x, y=y, num_classes=model.C, weights=w, param=opt_param, y_DA=y_DA, w_DA=w_DA, MI=MI)
        #    loss       = l + l_DA
        #else:

        # Clear gradients
        optimizer.zero_grad() # !
        
        loss_tuple = losstools.loss_wrapper(model=model, x=x, y=y, num_classes=model.C, weights=w, param=opt_param, MI=MI)  

        ## Create combined loss
        loss = 0
        for key in loss_tuple.keys():
            loss = loss + loss_tuple[key]

        ## Propagate gradients
        loss.backward(retain_graph=False)
        
        ## Gradient norm-clipping for stability if maximizing MI (unbounded)
        # For details: http://proceedings.mlr.press/v28/pascanu13.pdf
        if MI is not None:
            for k in range(len(MI['classes'])):
                deeptools.adaptive_gradient_clipping_(model, MI['model'][k])
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt_param['clip_norm'])
        
        ## Step optimizer
        optimizer.step()
        
        ## Aggregate losses
        total_loss = total_loss + loss.item()

        for key in loss_tuple.keys():
            if key in component_losses:
                component_losses[key] += loss_tuple[key].item()
            else:
                component_losses[key]  = loss_tuple[key].item()

        n_batches += 1

    model.eval() #!

    # --------------------------------------------------------------------
    ## Second (possible) step: MI network training

    if MI is not None:

        # At this stage, we train MI model(s)
        for k in range(len(MI['classes'])):
            MI['model'][k].train() #!

        n_batches    = 0
        network_loss = 0
        MI_lb = np.zeros(len(MI['classes']))

        for i, batch in enumerate(loader):

            batch_ = batch2tensor(batch, device)

            # -----------------------------------------
            if type(batch_) is dict:
                x,y,w = batch_['x'], batch_['y'], batch_['w']
                
                if 'u' in batch_: # Dual input models
                    x = {'x': batch_['x'], 'u': batch_['u']}
            # Torch-geometric models
            else:
                x,y,w   = batch_, batch_.y, batch_.w

            MI['x'] = batch_['x_MI']
            # -----------------------------------------

            MI['optimizer'].zero_grad() # !

            loss_tuple = losstools.loss_wrapper(model=model, x=x, y=y, num_classes=model.C, weights=w, param=opt_param, MI=MI)  

            MI['network_loss'].backward()
            for k in range(len(MI['classes'])):
                torch.nn.utils.clip_grad_norm_(MI['model'][k].parameters(), MI['clip_norm'])
            MI['optimizer'].step()

            # Accumulate for diagnostics
            network_loss += MI['network_loss'].item()
            for k in range(len(MI['classes'])):
                MI_lb[k] += MI['MI_lb'][k]

            n_batches += 1

        MI['network_loss'] = network_loss / n_batches
        MI['MI_lb']        = MI_lb / n_batches

        for k in range(len(MI['classes'])):
            MI['model'][k].eval() #!

    # Mean
    for key in component_losses.keys():
        component_losses[key] /= n_batches

    return {'sum': total_loss / n_batches, **component_losses}


def test(model, loader, optimizer, device):
    """
    Pytorch based testing routine.
    
    Args:
        model    : pytorch geometric model
        loader   : pytorch geometric dataloader
        optimizer: pytorch optimizer
        device   : 'cpu' or 'device'
    
    Returns
        accuracy, AUC
    """

    model.eval()

    DA_active  = True if (hasattr(model, 'DA_active') and model.DA_active) else False

    accsum = 0
    aucsum = 0
    k = 0

    for i, batch in enumerate(loader):

        batch_ = batch2tensor(batch, device)
        
        # -----------------------------------------
        # Torch models
        if type(batch_) is dict:
            x,y,w = batch_['x'], batch_['y'], batch_['w']

            if 'u' in batch_: # Dual models
                x = {'x': batch_['x'], 'u': batch_['u']}

        # Torch-geometric
        else:
            x,y,w = batch_, batch_.y, batch_.w
        # -----------------------------------------
        
        with torch.no_grad():
            pred = model.softpredict(x)
        
        weights = w.detach().cpu().numpy()
        y_true  = y.detach().cpu().numpy()
        y_pred  = pred.detach().cpu().numpy()
        
        # Classification metrics
        N       = len(y_true)
        metrics = aux.Metric(y_true=y_true, y_pred=y_pred, weights=weights, class_ids=None, hist=False, verbose=True)
        
        if metrics.auc > -1: # Bad batch protection
            aucsum += (metrics.auc * N)
            accsum += (metrics.acc * N)
            k += N

    if k > 0:
        return accsum / k, aucsum / k
    else:
        return accsum, aucsum


def model_to_cuda(model, device_type='auto'):
    """ Wrapper function to handle CPU/GPU setup
    """
    GPU_chosen = False

    if device_type == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            GPU_chosen = True
        else:
            device = torch.device('cpu:0')
    else:
        device = device_type

    model = model.to(device, non_blocking=True)
    
    # Try special map (.to() does not map all variables)
    try:
        model = model.to_device(device=device)
        print(__name__ + f'.model_to_cuda: Mapping special to <{device}>')
    except:
        True
    
    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(__name__ + f'.model_to_cuda: Multi-GPU {torch.cuda.device_count()}')
        model = nn.DataParallel(model)

    print(__name__ + f'.model_to_cuda: Computing device <{device}> chosen')
    
    if GPU_chosen:
        used  = io.get_gpu_memory_map()[0]
        total = io.torch_cuda_total_memory(device)
        cprint(__name__ + f'.model_to_cuda: device <{device}> VRAM in use: {used:0.2f} / {total:0.2f} GB', 'yellow')
        print('')

    return model, device
