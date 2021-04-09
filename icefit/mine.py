# MINE: Mutual Information Neural Estimation
#
# https://arxiv.org/abs/1801.04062
#
# Use adaptive gradient clipping when using MINE as a regulator cost.
#
# m.mieskolainen@imperial.ac.uk, 2021


import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd

from tqdm import tqdm


class Mine(nn.Module):
    """
    MINE network object
    """
    def __init__(self, input_size=2, hidden_dim=128, noise_std=0.025):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        # Noise initialization
        nn.init.normal_(self.fc1.weight, std=noise_std)
        nn.init.constant_(self.fc1.bias, 0)
        
        nn.init.normal_(self.fc2.weight, std=noise_std)
        nn.init.constant_(self.fc2.bias, 0)
        
        nn.init.normal_(self.fc3.weight, std=noise_std)
        nn.init.constant_(self.fc3.bias, 0)


    def forward(self, input):
        output = F.relu(self.fc1(input))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)

        return output


def train_mine(joint, marginal, net, opt, ma_eT, alpha=0.01, loss='mine_ema'):
    """
    Args:
        batch : a tuple of (joint, marginal)
        net   : network object
        opt   : optimizer object

    Note that bias corrected loss is used only for the gradient descent,
    MI value is calculated without it.
    
    MINE:

    MI_lb    : mutual information lower bound ("neural information measure")
               sup E_{P_{XZ}} [T_\theta] - log(E_{P_X \\otimes P_Z}[exp(T_\\theta)])
    """

    T, eT = net(joint), torch.exp(net(marginal))
    MI_lb = torch.mean(T) - torch.log(torch.mean(eT))

    # Unbiased estimate via exponentially moving average (FIR filter)
    # https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
    if   loss == 'MINE_EMA':

        ma_eT = alpha*torch.mean(eT) + (1 - alpha)*ma_eT
        l  = -(torch.mean(T) - (1/ma_eT.mean()).detach()*torch.mean(eT))

    # (Slightly) biased gradient based directly on the local MI value
    elif loss == 'MINE':
        l  = -MI_lb
    else:
        raise Exception(__name__ + ".train_mine: Unknown loss chosen")

    opt.zero_grad()
    autograd.backward(l)
    opt.step()

    return MI_lb, ma_eT


def sample_batch(X, Z, batch_size):
    """
    Sample batches of data, either from the joint or marginals
    
    Args:
        X           : input data (N x dim)
        Z           : input data (N x dim)
        batch_size  : batch size
    
    Returns:
        joint, marginal batches with size (N x [dim[X] + dim[Z])
    """

    index  = np.random.choice(range(X.shape[0]), size=batch_size, replace=False)
    random = np.random.choice(range(Z.shape[0]), size=batch_size, replace=False)

    # XZ and X(*)Z
    joint    = np.concatenate([X[index,:].reshape(-1,1),  Z[index,:].reshape(-1,1)], axis=1)
    marginal = np.concatenate([X[index,:].reshape(-1,1), Z[random,:].reshape(-1,1)], axis=1)

    return joint, marginal


def train(X, Z, net, opt, batch_size, num_iter, alpha, loss):
    """
    Train the network estimator

    Args:
        See estimate()
    
    Returns:
        mutual information estimates per iteration
    """
    result = []
    ma_eT  = 1.0
    
    for i in tqdm(range(num_iter)):

        # Sample input
        joint, marginal = sample_batch(X=X, Z=Z, batch_size=batch_size)

        # Create torch variables
        joint    = torch.FloatTensor(joint)
        marginal = torch.FloatTensor(marginal)

        if next(net.parameters()).is_cuda:
            joint    = joint.cuda()
            marginal = marginal.cuda()

        # Train
        MI_lb, ma_eT = train_mine(joint=joint, marginal=marginal, \
            net=net, opt=opt, ma_eT=ma_eT, alpha=alpha, loss=loss)

        result.append(MI_lb.detach().cpu().numpy())

    return result


def estimate(X, Z, batch_size=100, num_iter=int(1e3), lr=1e-3, hidden_dim=96, \
    loss='MINE_EMA', alpha=0.01, window=200, use_cuda=True, return_full=False):
    """
    Accurate Mutual Information Estimate via Neural Network
    
    Info:
        Input data X,Z can be random vectors (with different dimensions)
        or just scalar variables.
    
    Args:
        X          : input data variable 1 (N x dim1)
        Z          : input data variable 2 (N x dim2)
    
    Params:
        batch_size : optimization loop batch size
        num_iter   : number of iterations
        lr         : learning rate
        hidden_dim : network hidden dimension
        loss       : estimator loss 'MINE_EMA' (default, unbiased), 'MINE'
        alpha      : exponentially moving average parameter
        window     : iterations (tail) window size for the final estimate
    
    Return:
        mutual information estimate, its uncertainty
    """

    if len(X.shape) == 1:
        X = X[..., None]
    if len(Z.shape) == 1:
        Z = Z[..., None]

    # Create network
    net     = Mine(input_size=X.shape[1] + Z.shape[1], hidden_dim=hidden_dim)

    if torch.cuda.is_available() and use_cuda:
        net = net.cuda()

    opt     = optim.Adam(net.parameters(), lr=lr)
    result  = train(X=X, Z=Z, net=net, opt=opt, batch_size=batch_size, num_iter=num_iter, alpha=alpha, loss=loss)

    if not return_full:

        # Take estimate from the tail
        mu  = np.mean(result[-window:])
        err = np.std(result[-window:]) / np.sqrt(window) # standard error on mean

        if not np.isfinite(mu):
            mu  = 0
            err = 0

        return mu,err

    else:
        return result
