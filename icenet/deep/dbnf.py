# Block Neural Autoregressive Flow (BNAF)
# Generative (Neural Likelihood) Functions
#
# https://arxiv.org/abs/1904.04676
# https://github.com/nicola-decao/BNAF (MIT license)
#
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk


import os
import json
import pprint
import datetime

import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from torch.utils import data


from icenet.deep  import dopt

from . bnaf import *
from .. tools import aux


def compute_log_p_x(model, x):
    """ Model log-likelihood: log [ prod_{j=1}^N pdf(x_j)], where x ~ (iid) pdf(x)
    
    log p(x) = log p(z) + sum_{k=1}^K log|det J_{f_k}|
    
    Args:
        model : model object
        x     : N minibatch vectors
    Returns:
        log-likelihood value
    """

    # Evaluate the non-diagonal and the diagonal part
    y, log_diag = model(x)

    # Sum of log-likelihoods (log product)
    log_p_y = torch.distributions.Normal(torch.zeros_like(y), torch.ones_like(y)).log_prob(y).sum(dim=-1)
    
    return log_p_y + log_diag


def get_pdf(model, x) :
    """ Evaluate learned density (pdf) at point x
    
    Args:
        model :  model object
        x     :  input vector(s)
    Returns:
        pdf value

    Examples:
        > x = torch.tensor([[1.0, 2.0]])
        > l = get_pdf(model,x)
    """
    return (torch.exp(compute_log_p_x(model, x))).detach().numpy()


def predict(X, models, EPS=1E-12) :
    """
    2-class density ratio pdf(x,S) / pdf(x,B) for each vector x.
    
    Args:
        param    : input parameters
        X        : pytorch tensor of vectors
        models   : list of model objects
    Returns:
        LLR      : log-likelihood ratio
    """

    print(__name__ + f': Likelihood (density) ratio pdf(x,S)/pdf(x,B) for N = {X.shape[0]} events ...')

    sgn_likelihood = get_pdf(models[1], X)
    bgk_likelihood = get_pdf(models[0], X)
    
    LLR = sgn_likelihood / (bgk_likelihood + EPS)
    LLR[~np.isfinite(LLR)] = 0
    
    return LLR


class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, W):
        """ Initialization """
        self.X = X
        self.W = W

    def __len__(self):
        """ Return the total number of samples """
        return self.X.shape[0]

    def __getitem__(self, index):
        """ Generates one sample of data """
        # Use ellipsis ... to index over scalar [,:], vector [,:,:], tensor [,:,:,..,:] indices
        return self.X[index,...], self.W[index]


def train(model, optimizer, scheduler, trn_x, val_x, trn_weights, param, modeldir, EPS=1e-12):
    """ Train the model density.

    Args:
        model       : initialized model object
        optimizer   : optimizer object
        scheduler   : optimization scheduler
        trn_x       : training vectors
        val_x       : validation vectors
        trn_weights : training weights
        param       : parameters
        modeldir    : directory to save the model
    """
    
    model, device = dopt.model_to_cuda(model, param['device'])

    # TensorboardX
    if param['tensorboard']:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(os.path.join(param['tensorboard'], param['modelname']))
    
    epoch = param['start_epoch']

    params = {'batch_size': param['batch_size'],
            'shuffle': True,
            'num_workers': param['num_workers'],
            'pin_memory': True}

    # Training generator
    training_set         = Dataset(trn_x, trn_weights)
    training_generator   = torch.utils.data.DataLoader(training_set, **params)
    
    # Validation generator
    validation_set       = torch.utils.data.TensorDataset(torch.from_numpy(val_x).float().to(device))
    validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=param['batch_size'], shuffle=False)
    
    # Training loop
    for epoch in tqdm(range(param['start_epoch'], param['start_epoch'] + param['epochs']), ncols = 88):

        train_loss  = []
        permutation = torch.randperm((trn_x.shape[0]))

        for batch_x, batch_weights in training_generator:

            # Transfer to GPU
            batch_x       = batch_x.to(device, dtype=torch.float32, non_blocking=True)
            batch_weights = batch_weights.to(device, dtype=torch.float32, non_blocking=True)
            
            # Noise regularization
            if param['noise_reg'] > 0:
                noise   = torch.empty(batch_x.shape).normal_(mean=0, std=param['noise_reg']).to(device, dtype=torch.float32, non_blocking=True)
                batch_x = batch_x + noise
            
            # Per sample weights
            log_weights = torch.log(batch_weights + EPS)
            
            # Weighted negative log-likelihood loss
            lossvec = compute_log_p_x(model, batch_x)
            loss    = - (lossvec + log_weights).sum()
            
            # Zero gradients, calculate loss, calculate gradients and update parameters
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=param['clip_norm'])
            optimizer.step()

            train_loss.append(loss)

        train_loss = torch.stack(train_loss).mean()
        optimizer.swap()

        # Compute validation loss (without weighting)
        validation_loss = -torch.stack([compute_log_p_x(model, batch_x).mean().detach()
                                        for batch_x, in validation_generator], -1).mean()
        optimizer.swap()
        

        print('Epoch {:3}/{:3} -- train_loss: {:4.3f} -- validation_loss: {:4.3f}'.format(
            epoch + 1, param['start_epoch'] + param['epochs'], train_loss.item(), validation_loss.item()))

        stop = scheduler.step(validation_loss,
            callback_best   = aux.save_torch_model(model, optimizer, epoch + 1,
                modeldir + '/dbnf_' + param['model'] + '.pth'),
            callback_reduce = aux.load_torch_model(model, optimizer, param,
                modeldir + '/dbnf_' + param['model'] + '.pth'))
        
        if param['tensorboard']:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch + 1)
            writer.add_scalar('loss/validation', validation_loss.item(), epoch + 1)
            writer.add_scalar('loss/train', train_loss.item(), epoch + 1)
        
        if stop:
            break
    
    # Re-load the model        
    aux.load_torch_model(model, optimizer, param, modeldir + '/dbnf_' + param['model'] + '.pth')()

    optimizer.swap()
    validation_loss = - torch.stack([compute_log_p_x(model, x_mb).mean().detach()
                                     for x_mb, in validation_generator], -1).mean()

    print('###### Stop training after {} epochs!'.format(epoch + 1))
    print('Validation loss: {:4.3f}'.format(validation_loss.item()))


def create_model(param, verbose=False, rngseed=0):
    """ Construct the network object.
    
    Args:
        param : parameters
    Returns:
        model : model object
    """

    # For random permutations
    np.random.seed(rngseed)
    
    flows = []
    for f in range(param['flows']):

        # Layers of the flow block
        layers = []
        for _ in range(param['layers']):
            layers.append(MaskedWeight(param['n_dims'] * param['hidden_dim'],
                                       param['n_dims'] * param['hidden_dim'], dim=param['n_dims']))
            layers.append(Tanh())
        
        # Add this flow block
        flows.append(
            BNAF(*([MaskedWeight(param['n_dims'], param['n_dims'] * param['hidden_dim'], dim=param['n_dims']), Tanh()] + \
                   layers + \
                   [MaskedWeight(param['n_dims'] * param['hidden_dim'], param['n_dims'], dim=param['n_dims'])]),\
                 res=param['residual'] if f < param['flows'] - 1 else None
            )
        )

        # Flow permutations
        if f < param['flows'] - 1:
            flows.append(Permutation(param['n_dims'], param['perm']))

    # Create the model
    model  = Sequential(*flows)
    
    params = sum((p != 0).sum() if len(p.shape) > 1 else torch.tensor(p.shape).item()
                 for p in model.parameters()).item()

    # Print model information    
    print('{}'.format(model))
    print('Parameters={}, n_dims={}'.format(params, param['n_dims']))

    return model


def load_models(param, paths, modeldir):
    """ Load models from files
    """

    models = []
    for i in range(len(paths)):
        print(__name__ + f'.load_models: Loading model[{i}] from {paths[i]}')
        model = create_model(param, verbose=False)
        param['start_epoch'] = 0
        
        checkpoint = torch.load(modeldir + '/dbnf_' + paths[i] + '.pth')
        model.load_state_dict(checkpoint['model'])
        
        model.eval() # Turn on eval mode!
        models.append(model)

    return models
