# MINE: Mutual Information Neural Estimation
#
# https://arxiv.org/abs/1801.04062
#
# Use adaptive gradient clipping (see icenet/deep/deeptools.py)
# when using MINE as a regulator cost and maximizing MI (it has no strict upper bound).
# The minimum MI is bounded by zero, to remind.
# 
# m.mieskolainen@imperial.ac.uk, 2022

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
    """
    def __init__(self, input_size=2, mlp_dim=[128, 128],
            activation='relu', dropout=0.01, noise_std=0.025, batch_norm=False, **args):
        super(MINENet, self).__init__()
        
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


def apply_in_batches(X, Z, model, losstype, weights=None, batch_size=4096):
    """
    Compute Mutual Information estimate in batches (exact same result but memory friendly)
    for a pre-trained model (trained using the same statistics as X and Z follow)
    
    Args:
        X:        variable 
        Z:        variable 
        weights:  weights
        model:    MINE model

    Returns:
        MI estimate
    """
    model.eval() # !

    if weights is None:
        weights = torch.ones(len(X)).float().to(X.device)

    # Normalize
    weights   = (weights / torch.sum(weights)).squeeze()

    # -----------------------
    # Compute blocks
    N_batches = int(np.ceil(len(X) / batch_size))
    batch_ind = aux.split_start_end(range(len(X)), N_batches)
    # -----------------------

    sum_T     = 0.0
    sum_eT    = 0.0
    sum_MI_lb = 0.0

    for b in range(N_batches):
        ind    = np.arange(batch_ind[b][0], batch_ind[b][-1])

        joint, marginal, w = sample_batch(X=X[ind], Z=Z[ind], weights=weights[ind], batch_size=None, device=X.device)

        if losstype == 'MINE_EMA' or losstype == 'MINE':

            # Use the network, apply weights
            T, eT  = w * model(joint), w * torch.exp(model(marginal))

            # Sum local batch values
            sum_T  = sum_T  + torch.sum(T)
            sum_eT = sum_eT + torch.sum(eT)

        elif losstype == 'DENSITY':

            pred_1    = torch.sigmoid(model(joint))
            sum_MI_lb = sum_MI_lb + torch.sum(w * torch.log(pred_1 / (1 - pred_1))) 
        else:
            raise Exception(__name__ + ".apply_in_batches: Unknown losstype chosen")

    if   (losstype == 'MINE_EMA') or (losstype == 'MINE'):
        MI_lb = sum_T - torch.log(sum_eT)
    elif losstype == 'DENSITY':
        MI_lb = sum_MI_lb
    else:
        raise Exception(__name__ + ".apply_in_batches: Unknown losstype chosen")

    return MI_lb

def compute_mine(joint, marginal, w, model, ma_eT, alpha=0.01, losstype='MINE_EMA'):
    """
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
    # Normalize
    w     = (w / torch.sum(w)).squeeze()
    
    if losstype == 'MINE_EMA' or losstype == 'MINE':

        # Use the network, apply weights
        T, eT = w * model(joint), w * torch.exp(model(marginal))

        # MI lower bound (Donsker-Varadhan representation)
        MI_lb  = torch.sum(T) - torch.log(torch.sum(eT))

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
        EPS    = 1e-20

        pred_1 = torch.sigmoid(model(joint))
        pred_0 = torch.sigmoid(model(marginal))

        # Concatenate
        pred   = torch.cat((pred_1, pred_0), dim=0)
        true   = torch.cat((torch.ones_like(pred_1), torch.zeros_like(pred_0)), dim=0)
        ww     = torch.cat((w,w), dim=0); ww = ww / torch.sum(ww)

        ## Binary Cross Entropy
        loss   = - ww * (true*torch.log(pred + EPS) + (1-true)*(torch.log(1 - pred + EPS)))
        loss   = loss.sum()

        # Density (likelihood) ratio trick based
        MI_lb  = torch.sum(w * torch.log(pred_1 / (1 - pred_1)))
    else:
        raise Exception(__name__ + ".compute_mine: Unknown losstype chosen")

    return MI_lb, ma_eT, loss


def sample_batch(X, Z, weights, batch_size=None, device='cpu:0'):
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
        weights = torch.ones(batch_size).float()

    # Padd outer [] dimensions if dim 1 inputs
    if len(X.shape) == 1:
        X = X[..., None]
    if len(Z.shape) == 1:
        Z = Z[..., None]

    index    = np.random.choice(range(X.shape[0]), size=batch_size, replace=False)
    random   = np.random.choice(range(Z.shape[0]), size=batch_size, replace=False)

    # XZ and X(*)Z
    if type(X) is np.ndarray:
        joint    = torch.Tensor(np.concatenate([X[index,...],  Z[index,...]], axis=1)).to(device)
        marginal = torch.Tensor(np.concatenate([X[index,...], Z[random,...]], axis=1)).to(device)
        w        = torch.Tensor(weights[index]).to(device)
    else:
        joint    = torch.cat((X[index,...],  Z[index,...]), dim=1).to(device)
        marginal = torch.cat((X[index,...], Z[random,...]), dim=1).to(device)
        w        = weights[index].to(device)

    return joint, marginal, w


def train_loop(X, Z, weights, model, opt, clip_norm, batch_size, epochs, alpha, losstype, device='cpu:0'):
    """
    Train the network estimator

    Args:
        See estimate()
    
    Returns:
        mutual information estimates per iteration
    """
    num_iter = int(np.ceil(X.shape[0] / batch_size) * epochs)
    result   = torch.Tensor(np.zeros(num_iter, dtype=float)).to(device)

    model.train() #!

    ma_eT = None

    for i in tqdm(range(num_iter)):

        # Sample input
        joint, marginal, w = sample_batch(X=X, Z=Z, weights=weights, batch_size=batch_size, device=device)

        # Call it
        opt.zero_grad()

        MI_lb, ma_eT, loss = compute_mine(joint=joint, marginal=marginal, w=w, model=model, ma_eT=ma_eT, alpha=alpha, losstype=losstype)
        
        # Step gradient descent
        autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        opt.step()

        result[i] = MI_lb

    return result


def estimate(X, Z, weights=None, epochs=50, alpha=0.01, losstype='MINE_EMA',
    batch_size=256, lr=1e-3, weight_decay=0.0, clip_norm=1.0, 
    mlp_dim=[128, 128], dropout=0.01, activation='relu', noise_std=0.025,
    window_size=0.2, return_full=False, device=None, return_model_only=False, **args):
    """
    Accurate Mutual Information Estimate via Neural Network
    
    Info:
        Input data X,Z can be random vectors (with different dimensions)
        or just scalar variables.
    
    Args:
        X          : input data variable 1 (N x dim1) (either torch or numpy arrays)
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
        mutual information estimate, its uncertainty
    """

    # Padd outer [] dimensions if dim 1 inputs
    if len(X.shape) == 1:
        X = X[..., None]
    if len(Z.shape) == 1:
        Z = Z[..., None]

    if weights is None:
        if type(X) is np.ndarray:
            weights = np.ones(X.shape[0])
        else:
            weights = torch.Tensor(np.ones(X.shape[0])).float().to(device)

    # Create network
    input_size = X.shape[1] + Z.shape[1]
    model = MINENet(input_size=input_size,  mlp_dim=mlp_dim, dropout=dropout, activation=activation, noise_std=noise_std)

    # Transfer to CPU / GPU
    if device is None:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu:0')

    model  = model.to(device)
    opt    = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    result = train_loop(X=X, Z=Z, weights=weights, model=model, opt=opt, batch_size=batch_size,
        epochs=epochs, clip_norm=clip_norm, alpha=alpha, losstype=losstype, device=device)

    if return_model_only:
        return model

    if not return_full:

        # Take estimate from the tail
        N   = int(window_size * epochs)
        mu  = torch.mean(result[-N:])
        err = torch.std(result[-N:]) / np.sqrt(N) # standard error on the mean

        if type(X) is np.ndarray:
            return mu.detach().cpu().numpy(), err.detach().cpu().numpy()
        else:
            return mu, err
    else:

        if type(X) is np.ndarray:
            return result.detach().cpu().nympy()
        else:
            return result
