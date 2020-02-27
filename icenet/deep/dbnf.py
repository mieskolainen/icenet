# Block Neural Autoregressive Flow (BNAF) Based Classifier
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
import numpy as np

from .bnaf import *

from tqdm import tqdm
from torch.utils import data
from ..tools import aux


# Log-likelihood
#
def compute_log_p_x(model, x_mb):

    y_mb, log_diag_j_mb = model(x_mb)
    log_p_y_mb = torch.distributions.Normal(torch.zeros_like(y_mb),
                    torch.ones_like(y_mb)).log_prob(y_mb).sum(-1)
    return log_p_y_mb + log_diag_j_mb


# Evaluate likelihood at point x
#
# Example:
# x = torch.tensor([[1.0, 2.0]])
# l = get_pdf(model,x)
#
def get_pdf(model, x) :

    return torch.exp(compute_log_p_x(model, x)).item()


# Train the model density
#
def train(model, optimizer, scheduler, trn_x, val_x, trn_weights, param, modeldir):

    # Pytorch loaders
    dataset_valid     = torch.utils.data.TensorDataset(torch.from_numpy(val_x).float().to(param['device']))
    data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size=param['batch_size'], shuffle=False)
    
    # TensorboardX
    if param['tensorboard']:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(os.path.join(param['tensorboard'], param['modelname']))
    
    epoch = param['start_epoch']

    # Training loop
    for epoch in tqdm(range(param['start_epoch'], param['start_epoch'] + param['epochs']), ncols = 88):

        train_loss = []
        permutation = torch.randperm((trn_x.shape[0]))

        # Minibatch loop
        for i in range(0, trn_x.shape[0], param['batch_size']):

            # Get batch
            indices = permutation[i:i + param['batch_size']]
            x_mb    = torch.tensor(trn_x[indices], dtype=torch.float32) # needs to be float32!
            
            # Per sample weights
            weights = trn_weights[indices]
            
            # Weighted loss
            lossvec = compute_log_p_x(model, x_mb)
            loss    = - (lossvec * torch.tensor(weights)).sum() / weights.sum()

            # Zero gradients, calculate loss, calculate gradients and update parameters
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=param['clip_norm'])
            optimizer.step()

            train_loss.append(loss)
            
        train_loss = torch.stack(train_loss).mean()
        optimizer.swap()
        validation_loss = - torch.stack([compute_log_p_x(model, x_mb).mean().detach()
                                         for x_mb, in data_loader_valid], -1).mean()
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
                                     for x_mb, in data_loader_valid], -1).mean()

    print('###### Stop training after {} epochs!'.format(epoch + 1))
    print('Validation loss: {:4.3f}'.format(validation_loss.item()))


# Construct the model
#
#
def create_model(param, verbose=False):

    flows = []
    for f in range(param['flows']):
        layers = []
        for _ in range(param['layers'] - 1):
            layers.append(MaskedWeight(param['n_dims'] * param['hidden_dim'],
                                       param['n_dims'] * param['hidden_dim'], dim=param['n_dims']))
            layers.append(Tanh())
            
        flows.append(
            BNAF(*([MaskedWeight(param['n_dims'], param['n_dims'] * param['hidden_dim'], dim=param['n_dims']), Tanh()] + \
                   layers + \
                   [MaskedWeight(param['n_dims'] * param['hidden_dim'], param['n_dims'], dim=param['n_dims'])]),\
                 res=param['residual'] if f < param['flows'] - 1 else None
            )
        )

        if f < param['flows'] - 1:
            flows.append(Permutation(param['n_dims'], 'flip'))

    model = Sequential(*flows).to(param['device'])
    params = sum((p != 0).sum() if len(p.shape) > 1 else torch.tensor(p.shape).item()
                 for p in model.parameters()).item()

    # Print model information    
    print('{}'.format(model))
    print('Parameters={}, n_dims={}'.format(params, param['n_dims']))

    return model


# Input X is [# vectors x # dimensions]
#
# 2-class likelihood ratio predictions
#
def predict(param, X, paths, modeldir) :

    EPS = 1e-12

    print(__name__ + ': Loading background model from {}'.format(paths[0]))
    model_bgk = create_model(param, verbose=True)
    param['start_epoch'] = 0

    checkpoint = torch.load(modeldir + '/dbnf_' + paths[0] + '.pth')
    model_bgk.load_state_dict(checkpoint['model'])

    print(__name__ + ': Loading signal model from {}'.format(paths[1]))
    model_sgn = create_model(param, verbose=True)
    param['start_epoch'] = 0

    checkpoint = torch.load(modeldir + '/dbnf_' + paths[1] + '.pth')
    model_sgn.load_state_dict(checkpoint['model'])
    
    print(__name__ + ': Calculating likelihood ratios signal/background ...')
    LLR = np.zeros((X.shape[0]))

    for i in tqdm(range(X.shape[0]), ncols = 88):
        
        x = torch.tensor([X[i,:]], dtype=torch.float32)
        
        sgn_likelihood = get_pdf(model_sgn, x)
        bgk_likelihood = get_pdf(model_bgk, x)
        
        if ( sgn_likelihood > EPS) & (bgk_likelihood > EPS) :
            LLR[i] = sgn_likelihood / bgk_likelihood
    
    return LLR

