# Block Neural Autoregressive Flow (BNAF)
# Generative (Neural Likelihood) Functions
#
# https://arxiv.org/abs/1904.04676
# https://github.com/nicola-decao/BNAF (MIT license)
#
#
# m.mieskolainen@imperial.ac.uk, 2025

import os
import glob

import torch
import torch.nn.functional as F
from torch.utils import data
import numpy as np
from tqdm import tqdm

from icenet.deep  import optimize, deeptools

from . bnaf import *
from icenet.tools import aux, aux_torch


def compute_log_p_x(model, x):
    """ Model log-density value log[pdf(x), where x is the data vector]
    
    log p(x) = log p(z) + sum_{k=1}^K log|det J_{f_k}|
    
    Args:
        model : model object
        x     : N minibatch vectors
    Returns:
        log-likelihood (density) value
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
    return (torch.exp(compute_log_p_x(model=model, x=x))).detach().cpu().numpy()


def predict(X, models, return_prob=True, EPS=1E-9):
    """
    2-class density ratio pdf(x,S) / pdf(x,B) for each vector x.
    
    Args:
        param       : input parameters
        X           : pytorch tensor of vectors
        models      : list of model objects
        return_prob : return pdf(S) / (pdf(S) + pdf(B)), else pdf(S) / pdf(B)
    
    Returns:
        likelihood (density) ratio (or alternatively probability)
    """
    
    print(__name__ + f'.predict: Computing density (likelihood) ratio for N = {X.shape[0]} events | return_prob = {return_prob}')
    
    bgk_pdf = get_pdf(models[0], X)
    sgn_pdf = get_pdf(models[1], X)
    
    if return_prob:
        out = sgn_pdf / np.clip(sgn_pdf + bgk_pdf, EPS, None)
    else:
        out = sgn_pdf / np.clip(bgk_pdf, EPS, None)
    
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


def train(model, optimizer, scheduler, trn_x, val_x,
          trn_weights, val_weights, param, modeldir, save_name):
    """ Train the model density
    """
    
    opt_param = param['opt_param']
    
    model, device = optimize.model_to_cuda(model, param['device'])

    # TensorboardX
    if 'tensorboard' in param and param['tensorboard']:
        from tensorboardX import SummaryWriter
        
        for f in glob.glob(os.path.join(modeldir, 'events.out.tfevents.*')):
            os.remove(f) # Clean old logs
        
        writer = SummaryWriter(modeldir)

    if trn_weights is None:
        trn_weights = torch.ones(trn_x.shape[0], dtype=torch.float32)
    
    if val_weights is None:
        val_weights = torch.ones(val_x.shape[0], dtype=torch.float32)
    
    # Datasets
    training_set   = Dataset(trn_x, trn_weights)
    validation_set = Dataset(val_x, val_weights)
    
    # N.B. We use 'sampler' with 'BatchSampler', which loads a set of events using multiple event indices (faster) than the default
    # one which takes events one-by-one and concatenates the results (slow).
    params_train = {'batch_size'  : None,
                    'num_workers' : param['num_workers'],
                    'sampler'     : torch.utils.data.BatchSampler(
                        torch.utils.data.RandomSampler(training_set), param['opt_param']['batch_size'], drop_last=False
                    ),
                    'pin_memory'  : True}
    
    params_validate = {'batch_size'  : None,
                    'num_workers' : param['num_workers'],
                    'sampler'     : torch.utils.data.BatchSampler(
                        torch.utils.data.RandomSampler(validation_set), param['eval_batch_size'], drop_last=False
                    ),
                    'pin_memory'  : True}
    
    training_loader   = torch.utils.data.DataLoader(training_set,   **params_train)
    validation_loader = torch.utils.data.DataLoader(validation_set, **params_validate)
    
    # Loss function (NLL)
    """
    Note:
        log-likelihood functions can be weighted linearly, due to
        \\prod_i p_i(\\theta; x_i)**w_i ==\\log==> \\sum_i w_i \\log p_i(\\theta; x_i)
    """
    
    trn_losses = {}
    val_losses = {}
    trn_losses_list = []
    val_losses_list = []
    
    # Training loop
    for epoch in tqdm(range(param['opt_param']['start_epoch'], param['opt_param']['start_epoch'] + param['opt_param']['epochs']), ncols = 88):
        
        model.train() # !

        trn_loss = 0.0
        denom    = 0.0
        
        # Scheduled noise regularization
        sigma2 = None
        if 'noise_reg' in opt_param and opt_param['noise_reg'] > 0.0:
            noise_reg = opt_param['noise_reg']
            sigma2 = noise_reg * deeptools.sigmoid_schedule(t=epoch, N_max=opt_param['epochs'])
            
            print(f'Noise reg. sigma2 = {sigma2:0.3E}')
        
        for x, w in training_loader:
            
            # Transfer to GPU
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            w = w.to(device, dtype=torch.float32, non_blocking=True)

            # ---------------------------------------------------------------
            # Add scheduled noise regularization
            if sigma2 is not None:
                x = np.sqrt(1 - sigma2)*x + np.sqrt(sigma2)*torch.randn_like(x)
            # ---------------------------------------------------------------    

            # Compute loss [per batch w normalized]
            loss = -(w * compute_log_p_x(model, x)).sum() / w.sum()
            
            # Zero gradients, calculate loss, calculate gradients and update parameters
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=param['opt_param']['clip_norm'])
            optimizer.step()

            trn_loss += loss.item()
            denom    += 1

        trn_loss = trn_loss / denom
        
        optimizer.swap()

        # ----------------------------------------------------------------
        # Compute validation loss
        
        if epoch == 0 or (epoch % param['savemode']) == 0:
            
            model.eval() # !

            val_loss = 0.0
            denom    = 0.0
            
            with torch.no_grad():
            
                for x, w in validation_loader:

                    # Transfer to GPU
                    x = x.to(device, dtype=torch.float32, non_blocking=True)
                    w = w.to(device, dtype=torch.float32, non_blocking=True)

                    # Compute loss [per batch w normalized]
                    loss = -(w * compute_log_p_x(model, x)).sum()
                    
                    val_loss += loss.item()
                    denom    += 1
            
            val_loss = val_loss / denom

            optimizer.swap()
        
        # ----------------------------------------------------------------
        
        ## ** Save values **
        optimize.trackloss(loss={'NLL': trn_loss}, loss_history=trn_losses)
        optimize.trackloss(loss={'NLL': val_loss}, loss_history=val_losses)
        
        trn_losses_list.append(trn_loss)
        val_losses_list.append(val_loss)
        
        # Step scheduler
        stop = scheduler.step(val_loss)
        
        print('Epoch {:3d}/{:3d} | Train: loss: {:4.3f} | Validation: loss: {:4.3f} | lr: {:0.3E}'.format(
            epoch+1,
            param['opt_param']['start_epoch'] + param['opt_param']['epochs'],
            trn_loss,
            val_loss,
            scheduler.get_last_lr()[0])
        )
        
        # Save
        if epoch == 0 or (epoch % param['savemode']) == 0:
            
            filename = f'{modeldir}/{save_name}_{epoch}.pth'
            aux_torch.save_torch_model(
                model     = model,
                optimizer = optimizer,
                epoch     = epoch,
                losses    = {'trn_losses': trn_losses_list, 'val_losses': val_losses_list},
                filename  = filename
            )() # Note ()

        if 'tensorboard' in param and param['tensorboard']:
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
            writer.add_scalar('loss/validation', val_loss, epoch)
            writer.add_scalar('loss/train', trn_loss, epoch)
        
        if stop:
            break
    
    return trn_losses, val_losses


def create_model(param, verbose=False, rngseed=0):
    """ Construct the network object.
    
    Args:
        param : parameters
    Returns:
        model : model object
    """

    # For random permutations
    aux.set_random_seed(rngseed)
    
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
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print model information    
    print('{}'.format(model))
    print('Parameters={}, n_dims={}'.format(params, param['n_dims']))

    return model


def load_models(param, modelnames, modeldir, device):
    """ Load models from files
    """
    
    models = []
    for i in range(len(modelnames)):
        print(__name__ + f'.load_models: Loading model[{i}] from {modelnames[i]}')

        model      = create_model(param['model_param'], verbose=False)
        param['opt_param']['start_epoch'] = 0

        filename   = aux.create_model_filename(path=modeldir, label=modelnames[i], \
            epoch=param['readmode'], filetype='.pth')
        checkpoint = torch.load(filename, map_location='cpu')
        
        model.load_state_dict(checkpoint['model'])
        model, device = optimize.model_to_cuda(model, device_type=device)
        
        model.eval() # Turn on eval mode!

        models.append(model)

    return models, device
