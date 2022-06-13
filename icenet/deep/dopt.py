# Deep Learning optimization functions
# 
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

import torch
import uproot
import numpy as np
import sklearn
import psutil
from termcolor import colored,cprint

from matplotlib import pyplot as plt
import pdb
from tqdm import tqdm, trange

import torch.optim as optim
from   torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


from icenet.deep import losstools
from icenet.deep.tempscale import ModelWithTemperature

from icenet.tools import aux
from icenet.tools import aux_torch
from icenet.tools import io
from icefit import mine

from ray import tune


init_funcs = {
    1: lambda x: torch.nn.init.normal_(x, mean=0.0, std=1.0),  # bias terms
    2: lambda x: torch.nn.init.xavier_normal_(x,   gain=1.0),  # weight terms
    3: lambda x: torch.nn.init.xavier_uniform_(x,  gain=1.0),  # conv1D filter
    4: lambda x: torch.nn.init.xavier_uniform_(x,  gain=1.0),  # conv2D filter
    "default": lambda x: torch.nn.init.constant(x, 1.0),       # others
}


def weights_init_all(model, init_funcs):
    """
    Examples:
        model = MyNet()
        weights_init_all(model, init_funcs)
    """
    for p in model.parameters():
        init_func = init_funcs.get(len(p.shape), init_funcs["default"])
        init_func(p)


def weights_init_uniform_rule(m):
    """ Initializes module weights from uniform [-a,a]
    """
    classname = m.__class__.__name__

    # Linear layers
    if classname.find('Linear') != -1:
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def weights_init_normal(m):
    """ Initializes module weights from normal distribution
    with a rule sigma ~ 1/sqrt(n)
    """
    classname = m.__class__.__name__
    
    # Linear layers
    if classname.find('Linear') != -1:
        y = m.in_features
        m.weight.data.normal_(0.0, 1/np.sqrt(y))
        m.bias.data.fill_(0)


class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, Y, W):
        """ Initialization """
        self.x = X
        self.y = Y
        self.w = W

    def __len__(self):
        """ Return the total number of samples """
        return self.y.shape[0]

    def __getitem__(self, index):
        """ Generates one sample of data """

        # Use ellipsis ... to index over scalar [,:], vector [,:,:], tensor [,:,:,..,:] indices
        return {'x': self.x[index,...], 'y': self.y[index, ...], 'w': self.w[index, ...]}


class DualDataset(torch.utils.data.Dataset):

    def __init__(self, X, U, Y, W):
        """ Initialization """
        self.x = X  # e.g. image tensors
        self.u = U  # e.g. global features 
        
        self.y = Y
        self.w = W
    
    def __len__(self):
        """ Return the total number of samples """
        return self.y.shape[0]

    def __getitem__(self, index):
        """ Generates one sample of data """
        # Use ellipsis ... to index over scalar [,:], vector [,:,:], tensor [,:,:,..,:] indices

        return {'x': self.x[index,...], 'u': self.u[index,...], 'y': self.y[index, ...], 'w': self.w[index, ...]}


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
        if 'u' in batch.keys():
            x = {'x': batch['x'], 'u': batch['u']}
        else:
            x = batch['x']
        w = batch['w']
        y = batch['y']

    # Pytorch geometric type
    else:
        batch = batch.to(device)
        x = batch # this contains all graph (node and edge) information
        y = batch.y
        w = batch.w

    return x,y,w


def train(model, loader, optimizer, device, opt_param):
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
    model.train()
    
    total_loss = 0
    n = 0
    for i, batch in enumerate(loader):

        x,y,w = batch2tensor(batch, device)

        optimizer.zero_grad() # !

        # Compute loss
        weights = aux_torch.weight2onehot(weights=w, y=y, num_classes=model.C)
        loss    = losstools.loss_wrapper(model=model, x=x, y=y, num_classes=model.C, weights=weights, param=opt_param)
        
        # Propagate gradients
        loss.backward()
        
        if type(batch) is dict: # DualDataset or Dataset
            total_loss += loss.item()
            n += y.shape[0]
        else:
            total_loss += loss.item() * batch.num_graphs # torch-geometric
            n += batch.num_graphs

        optimizer.step()

    return total_loss / n


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

    accsum = 0
    aucsum = 0
    k = 0

    for i, batch in enumerate(loader):
        
        x,y,w = batch2tensor(batch, device)

        with torch.no_grad():
            phat = model.softpredict(x) # Probability
            #pred = phat.argmax(dim=1)  # Maximum probability class (index)
        
        weights = w.detach().cpu().numpy()
        y_true  = y.detach().cpu().numpy()
        y_soft  = phat.detach().cpu().numpy()
        
        # Classification metrics
        N       = len(y_true)
        metrics = aux.Metric(y_true=y_true, y_soft=y_soft, weights=weights, num_classes=model.C, hist=False)

        if metrics.auc > -1: # Bad batch protection
            aucsum += (metrics.auc * N)
            accsum += (metrics.acc * N)
            k += N

    return accsum / k, aucsum / k


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
        device = param['device']

    model = model.to(device, non_blocking=True)

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


"""
def train_loop():

    ### Weight initialization
    #print(__name__ + f'.train: Weight initialization')
    #model.apply(weights_init_normal_rule)
    
    # Class fractions
    frac = [sum(Y_trn.cpu().numpy() == i) / Y_trn.cpu().numpy().size for i in range(model.C)]
    
    # ------------------------------------------------------------
    # Mutual information regularization for the output independence w.r.t the target variable
    
    if param['opt_param']['MI_reg_on'] == True:

        # Input variables, make output orthogonal with respect to 'ortho' variable
        x1 = log_phat
        x2 = batch_x['ortho']

        MI, MI_err = mine.estimate(X=x1, Z=x2, num_iter=2000, loss=method)

        # Adaptive gradient clipping
        # ...

        # Add to the loss
        loss += param['opt_param']['MI_reg_beta'] * MI
    
    if clip_gradients:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Clip gradient for NaN problems
"""
