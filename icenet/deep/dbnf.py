# Block Neural Autoregressive Flow (BNAF)
# Generative (Neural Likelihood) Functions
#
# https://arxiv.org/abs/1904.04676
# https://github.com/nicola-decao/BNAF (MIT license)
#
#
# m.mieskolainen@imperial.ac.uk, 2024


import os
import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from torch.utils import data


from icenet.deep  import optimize

from . bnaf import *
from icenet.tools import aux
from icenet.tools import aux_torch


def compute_log_p_x(model, x):
    """ Model log-density value log[pdf(x), where x is the data vector]
    
    log p(x) = log p(z) + sum_{k=1}^K log|det J_{f_k}|
    
    Args:
        model : model object
        x     : N minibatch vectors
    Returns:
        log-likelihood value
    """

    # Evaluate the non-diagonal and the diagonal part
    y, log_diag = model(x)

    # Sum of log-likelihoods (log product) pushed through the flow, evaluated for the unit Gaussian
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
    return (torch.exp(compute_log_p_x(model=model, x=x))).detach().numpy()


def predict(X, models, return_prob=True, EPS=1E-12):
    """
    2-class density ratio pdf(x,S) / pdf(x,B) for each vector x.
    
    Args:
        param      : input parameters
        X          : pytorch tensor of vectors
        models     : list of model objects
        return_prob: return pdf(S) / (pdf(S)+pdf(B)), else pdf(S)/pdf(B)
    
    Returns:
        likelihood ratio (or alternatively probability)
    """
    
    print(__name__ + f'.predict: Computing density (likelihood) ratio for N = {X.shape[0]} events ...')
    
    bgk_pdf = get_pdf(models[0], X)
    sgn_pdf = get_pdf(models[1], X)
    
    if return_prob:
        out = sgn_pdf / np.clip(sgn_pdf + bgk_pdf, a_min=EPS, a_max=None)
    else:
        out = sgn_pdf / np.clip(bgk_pdf, a_min=EPS, a_max=None)
    
    out[~np.isfinite(out)] = 0
    
    return out


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


def train(model, optimizer, scheduler, trn_x, val_x, trn_weights, val_weights, param, modeldir):
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
    label = param['label']

    model, device = optimize.model_to_cuda(model, param['device'])

    # TensorboardX
    if param['tensorboard']:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(os.path.join(param['tensorboard'], param['modelname']))

    params = {'batch_size': param['opt_param']['batch_size'],
              'shuffle'     : True,
              'num_workers' : param['num_workers'],
              'pin_memory'  : True}
    
    # Training generator
    training_set         = Dataset(trn_x, trn_weights)
    training_generator   = torch.utils.data.DataLoader(training_set, **params)
    
    # Validation generator
    validation_set       = Dataset(val_x, val_weights)
    validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=512, shuffle=False)
    
    # Loss function
    """
    Note:
        log-likelihood functions can be weighted linearly, due to
        \\prod_i p_i(\\theta; x_i)**w_i ==\\log==> \\sum_i w_i \\log p_i(\\theta; x_i)
    """
    def lossfunc(model, x, weights):
        w = weights / torch.sum(weights, dim=0)
        lossvec = compute_log_p_x(model, x)
        return -(lossvec * w).sum(dim=0) # log-likelihood
    
    # Training loop
    for epoch in tqdm(range(param['opt_param']['start_epoch'], param['opt_param']['start_epoch'] + param['opt_param']['epochs']), ncols = 88):

        model.train() # !

        train_loss  = []
        permutation = torch.randperm((trn_x.shape[0]))

        for batch_x, batch_weights in training_generator:

            # Transfer to GPU
            batch_x       = batch_x.to(device, dtype=torch.float32, non_blocking=True)
            batch_weights = batch_weights.to(device, dtype=torch.float32, non_blocking=True)

            loss = lossfunc(model=model, x=batch_x, weights=batch_weights)
            
            # Zero gradients, calculate loss, calculate gradients and update parameters
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=param['opt_param']['clip_norm'])
            optimizer.step()

            train_loss.append(loss)

        train_loss = torch.stack(train_loss).mean()
        optimizer.swap()

        # ----------------------------------------------------------------
        # Compute validation loss
        model.eval() # !

        validation_loss = []
        for batch_x, batch_weights in validation_generator:

            # Transfer to GPU
            batch_x       = batch_x.to(device, dtype=torch.float32, non_blocking=True)
            batch_weights = batch_weights.to(device, dtype=torch.float32, non_blocking=True)
            
            loss = lossfunc(model=model, x=batch_x, weights=batch_weights)
            validation_loss.append(loss)

        validation_loss = torch.stack(validation_loss).mean()
        optimizer.swap()
        # ----------------------------------------------------------------

        print('Epoch {:3d}/{:3d} | Train: loss: {:4.3f} | Validation: loss: {:4.3f}'.format(
            epoch + 1, param['opt_param']['start_epoch'] + param['opt_param']['epochs'], train_loss.item(), validation_loss.item()))

        stop = scheduler.step(validation_loss,
            callback_best = aux_torch.save_torch_model(model=model, optimizer=optimizer, epoch=epoch,
                filename = modeldir + f'/{label}_' + param['model'] + '_' + str(epoch) + '.pth'),
            callback_reduce = aux_torch.load_torch_model(model=model, optimizer=optimizer,
                filename = modeldir + f'/{label}_' + param['model'] + '_' + str(epoch) + '.pth', device=device))
        
        if param['tensorboard']:
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('loss/validation', validation_loss.item(), epoch)
            writer.add_scalar('loss/train', train_loss.item(), epoch)
        
        if stop:
            break


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


def load_models(param, modelnames, modeldir, device='cpu'):
    """ Load models from files
    """
    
    models = []
    for i in range(len(modelnames)):
        print(__name__ + f'.load_models: Loading model[{i}] from {modelnames[i]}')

        model      = create_model(param['model_param'], verbose=False)
        param['opt_param']['start_epoch'] = 0

        filename   = aux.create_model_filename(path=modeldir, label=modelnames[i], \
            epoch=param['readmode'], filetype='.pth')
        checkpoint = torch.load(filename, map_location=device)
        
        model.load_state_dict(checkpoint['model'])
        model.eval() # Turn on eval mode!

        models.append(model)

    return models
