# MINE: Mutual Information Neural Estimation
#
# https://arxiv.org/abs/1801.04062
#
# Use adaptive gradient clipping (see icenet/deep/deeptools.py)
# when using MINE as a regulator cost and maximizing MI (it has no strict upper bound).
# The minimum MI is bounded by zero, to remind.
# 
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

from icenet.tools import aux
from icenet.deep.dmlp import MLP
from tqdm import tqdm


class MINENet(nn.Module):
    """
    MINE network object
    
    input_size: total (summed) dimension of two random variables, can be different
    mlp_dim:    mlp inner dimensionality
    
    """
    def __init__(self, input_size=2, mlp_dim=[128, 128], noise_std=0.025,
            activation='relu', dropout=0.01, batch_norm=False, EPS=1E-12, **kwargs):
        super(MINENet, self).__init__()
        
        self.EPS = EPS
        
        print(__name__ + f'__init__: input_size = {input_size} | mlp_dim = {mlp_dim} | noise_std = {noise_std}')
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=noise_std)
                nn.init.constant_(m.bias, 0)

        self.mlp = MLP([input_size] + mlp_dim + [1],
            activation=activation, dropout=dropout, batch_norm=batch_norm)
        
        # Special initialize with noise
        self.mlp.apply(init_weights)

    def forward(self, x):
        
        return self.mlp(x).squeeze()


def sample_batch(X: torch.Tensor, Z: torch.Tensor, weights: torch.Tensor=None,
                 batch_size=None, device=torch.device('cpu:0')):
    """
    Sample batches of data, either from the joint or marginals
    
    Args:
        X           : input data (N x dim)
        Z           : input data (N x dim)
        weights     : input weights (N)
        batch_size  : batch size
    
    Returns:
        joint, marginal batches with size (N x [dim[X] + dim[Z]])
    """
    
    if batch_size is None:
        batch_size = X.shape[0]

    if batch_size > X.shape[0]:
        batch_size = X.shape[0]
    
    if weights is None:
        weights = torch.ones(batch_size)
    
    # Padd outer [] dimensions if dim 1 inputs
    if len(X.shape) == 1:
        X = X[..., None]
    if len(Z.shape) == 1:
        Z = Z[..., None]

    index    = np.random.choice(range(X.shape[0]), size=batch_size, replace=False)
    random   = np.random.choice(range(Z.shape[0]), size=batch_size, replace=False)

    # XZ and X(*)Z
    joint    = torch.cat((X[index,...],  Z[index,...]), dim=1).to(device)
    marginal = torch.cat((X[index,...], Z[random,...]), dim=1).to(device)
    w        = weights[index].to(device)

    return joint, marginal, w

def compute_mine(joint: torch.Tensor, marginal: torch.Tensor, w: torch.Tensor,
                 model: nn.Module, ma_eT: float = None, alpha=0.01, EPS=1e-12, losstype='MINE_EMA'):
    """
    Compute Mutual Information estimate
    
    Args:
        joint, marginal : input data
        w     : input data weights
        net   : network object
    
    Note that bias corrected loss is used only for the gradient descent,
    MI value is calculated without it.
    
    Notes:
        MI_lb is mutual information lower bound ("neural information measure")
        sup E_{P_{XZ}} [T_theta] - log(E_{P_X otimes P_Z}[exp(T_theta)])
    """
    
    # Normalize for the expectation E[] values
    w = w / w.sum()
    
    if losstype == 'MINE_EMA' or losstype == 'MINE':

        # Use the network, apply weights
        T, eT = w * model(joint), w * torch.exp(model(marginal))

        # MI lower bound (Donsker-Varadhan representation)
        sum_eT = torch.clip(torch.sum(eT), min=EPS)
        MI_lb  = torch.sum(T) - torch.log(sum_eT)

    # Unbiased (see orig. paper) estimate via exponentially moving average (FIR filter)
    # https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
    if   losstype == 'MINE_EMA':
        if ma_eT == None: ma_eT = torch.sum(eT).item() # First call
        
        ma_eT  = (1 - alpha)*torch.sum(eT) + alpha*ma_eT
        loss   = -(torch.sum(T) - torch.sum(eT) / torch.sum(ma_eT).detach())
        
    # (Slightly) biased gradient based directly on the local MI value
    elif losstype == 'MINE':
        loss   = -MI_lb

    # Density ratio trick based
    elif losstype == 'DENSITY':
        
        pred_1 = torch.clip(torch.sigmoid(model(joint)),    min=EPS, max=1-EPS)
        pred_0 = torch.clip(torch.sigmoid(model(marginal)), min=EPS, max=1-EPS)

        # Concatenate
        pred   = torch.cat((pred_1, pred_0), dim=0)
        true   = torch.cat((torch.ones_like(pred_1), torch.zeros_like(pred_0)), dim=0)
        ww     = torch.cat((w,w), dim=0); ww = ww / torch.sum(ww)
        
        ## Binary Cross Entropy
        loss   = - ww * (true*torch.log(pred) + (1-true)*(torch.log(1 - pred)))
        loss   = loss.sum()

        # Density (likelihood) ratio trick based
        MI_lb  = torch.sum(w * torch.log(pred_1 / (1 - pred_1)))
    else:
        raise Exception(__name__ + ".compute_mine: Unknown losstype chosen")

    if torch.isnan(MI_lb):
        raise Exception(__name__ + f'.compute_mine: MI_lb is NaN -- adjust SGD / network param and check the input')
    
    return MI_lb, ma_eT, loss


def apply_mine(X: torch.Tensor, Z: torch.Tensor, model: nn.Module, losstype: str, weights: torch.Tensor=None):
    """
    Compute Mutual Information by re-applying the trained model
    
    (trained using statistics as input ~ X and Z)
    """
    
    model.eval() # !
    
    joint, marginal, w = sample_batch(X=X, Z=Z, weights=weights, batch_size=None, device=X.device)                
    return compute_mine(joint=joint, marginal=marginal, w=w, model=model, losstype=losstype)


def apply_mine_batched(X: torch.Tensor, Z: torch.Tensor, model: nn.Module,
                         losstype: str, weights: torch.Tensor=None, batch_size=4096, EPS=1E-12):
    """
    Compute Mutual Information by re-applying the trained model in batches
    
    (trained using statistics as input ~ X and Z)
    """
    model.eval() # !

    print(__name__ + f'.compute_mine_batched: Using device "{X.device}"')

    if weights is None:
        weights = torch.ones(X.shape[0], dtype=torch.float32).to(X.device)

    # ** Normalize for the expectation E[] values here **
    weights   = (weights / torch.sum(weights)).squeeze()

    # -----------------------
    # Compute blocks
    N_batches = int(np.ceil(X.shape[0] / batch_size))
    batch_ind = aux.split_start_end(range(X.shape[0]), N_batches)
    # -----------------------

    sum_T     = 0.0
    sum_eT    = 0.0
    sum_MI_lb = 0.0

    for b in (pbar := tqdm(range(len(batch_ind)))):
        ind    = np.arange(batch_ind[b][0], batch_ind[b][-1])

        joint, marginal, w = sample_batch(X=X[ind], Z=Z[ind], weights=weights[ind], batch_size=None, device=X.device)

        if losstype == 'MINE_EMA' or losstype == 'MINE':

            # Use the network, apply weights
            # (no normalization of weights here, because we sum over all batches)
            T, eT  = w * model(joint), w * torch.exp(model(marginal))

            # Sum local batch values
            sum_T  = sum_T  + torch.sum(T)
            sum_eT = torch.clip(sum_eT + torch.sum(eT), min=EPS)

            # Update progress bar
            MI = sum_T - torch.log(sum_eT)
            pbar.set_description(f'MI = {MI.item():0.5f}')
            
        elif losstype == 'DENSITY':
            
            pred_1    = torch.clip(torch.sigmoid(model(joint)), min=EPS, max=1-EPS)
            sum_MI_lb = sum_MI_lb + torch.sum(w * torch.log(pred_1 / (1 - pred_1))) 
            
            # Update progress bar
            MI = sum_MI_lb
            pbar.set_description(f'MI = {MI.item():0.5f}')
            
        else:
            raise Exception(__name__ + ".apply_in_batches: Unknown losstype chosen")
    
    if   (losstype == 'MINE_EMA') or (losstype == 'MINE'):
        MI_lb = sum_T - torch.log(sum_eT)
    elif losstype == 'DENSITY':
        MI_lb = sum_MI_lb
    else:
        raise Exception(__name__ + ".apply_in_batches: Unknown losstype chosen")

    return MI_lb


def train_loop(X: torch.Tensor, Z: torch.Tensor, weights: torch.Tensor,
               model: nn.Module, opt, clip_norm, batch_size, epochs, alpha, losstype, device=torch.device('cpu:0')):
    """
    Train the network estimator

    Args:
        See estimate()
    
    Returns:
        mutual information estimates per iteration
    """
    
    if weights is None:
        weights = torch.ones(batch_size).to(device)
    
    # -----------------------
    # Compute blocks
    N_batches = int(np.ceil(X.shape[0] / batch_size))
    batch_ind = aux.split_start_end(range(X.shape[0]), N_batches)
    # -----------------------
    
    result   = torch.zeros(len(batch_ind) * epochs).to(device)

    print(__name__ + f'.train_loop: Using device "{device}"')
    
    model.train() #!

    ma_eT = None
    k     = 0
    
    for _ in (pbar := tqdm(range(epochs))):
        
        # Sample shuffled input
        joint, marginal, w = sample_batch(X=X, Z=Z, weights=weights, batch_size=None, device=device)

        for b in range(len(batch_ind)):
            
            ind = np.arange(batch_ind[b][0], batch_ind[b][-1])
            
            opt.zero_grad()

            # Compute it
            MI_lb, ma_eT, loss = compute_mine(joint=joint[ind],
                                              marginal=marginal[ind],
                                              w=w[ind],
                                              model=model, ma_eT=ma_eT, alpha=alpha, losstype=losstype)

            # Step gradient descent
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            opt.step()
            
            # Update progress bar
            pbar.set_description(f'loss = {loss.item():0.5f} | MI_lb = {MI_lb.item():0.5f}')

            result[k] = MI_lb
            k += 1
    
    return result


def estimate(X: torch.Tensor, Z: torch.Tensor, weights=None, epochs=50, alpha=0.01, losstype='MINE_EMA',
    batch_size=256, lr=1e-3, weight_decay=1e-5, clip_norm=1.0, 
    mlp_dim=[128, 128], dropout=0.01, activation='relu', batch_norm=False, noise_std=0.025,
    window_size=0.2, return_full=False, device=None, return_model_only=False, **kwargs):
    """
    Accurate Mutual Information Estimate via Neural Network
    
    Info:
        Input data X,Z can be random vectors (with different dimensions)
        or just scalar variables.
    
    Args:
        X          : input data variable 1 (N x dim1)
        Z          : input data variable 2 (N x dim2)
        weights    : input data weights (N) (set None if no weights)
    
    Params:
        batch_size  : optimization loop batch size
        num_iter    : number of iterations
        lr          : learning rate
        hidden_dim  : network hidden dimension
        loss        : estimator loss 'MINE_EMA' (default, unbiased), 'MINE'
        alpha       : exponentially moving average parameter
        window _size: iterations (tail) window size for the final estimate
    
    Return:
        MI estimate
        MI uncertainty
        model
    """

    # Transfer to CPU / GPU
    if device is None or device == 'auto':
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu:0')

    # Padd outer [] dimensions if dim 1 inputs
    if len(X.shape) == 1:
        X = X[..., None]
    if len(Z.shape) == 1:
        Z = Z[..., None]

    X = X.to(device)
    Z = Z.to(device)
    
    if weights is None:
        weights = torch.ones(X.shape[0]).to(device)
    
    # Create network
    input_size = X.shape[1] + Z.shape[1]
    model  = MINENet(input_size=input_size, mlp_dim=mlp_dim,
                    noise_std=noise_std, dropout=dropout, activation=activation, batch_norm=batch_norm)

    model  = model.to(device)
    
    opt    = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    result = train_loop(X=X, Z=Z, weights=weights, model=model, opt=opt, batch_size=batch_size,
        epochs=epochs, clip_norm=clip_norm, alpha=alpha, losstype=losstype, device=device)

    if return_model_only:
        return model
    
    if not return_full:
        
        # Take estimate from the tail
        N   = int(window_size * epochs)
        mu  = torch.mean(result[-N:])
        err = torch.std(result[-N:]) / np.sqrt(N) # standard error on the mean

        return mu, err, model
    else:
        return result, model
