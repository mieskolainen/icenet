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

from icenet.deep.da import GradientReversal

class MINENet(nn.Module):
    """
    MINE network object
    """
    def __init__(self, input_size=2, hidden_dim=128, noise_std=0.025):
        super(MINENet, self).__init__()
        
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


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


def train_mine(joint, marginal, w, net, ma_eT, alpha=0.01, losstype='MINE_EMA'):
    """
    Args:
        joint, marginal : input data
        w     : input data weights
        net   : network object
    
    Note that bias corrected loss is used only for the gradient descent,
    MI value is calculated without it.
    
    MINE:

    MI_lb    : mutual information lower bound ("neural information measure")
               sup E_{P_{XZ}} [T_\\theta] - log(E_{P_X \\otimes P_Z}[exp(T_\\theta)])
    """
    
    # Use the network
    T, eT = net(joint), torch.exp(net(marginal))

    # Apply event weights
    w   = w / w.sum() / len(w)
    T   = w*T
    eT  = w*eT

    # MI lower bound
    MI_lb = torch.sum(T) - torch.log(torch.sum(eT))

    # Unbiased estimate via exponentially moving average (FIR filter)
    # https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
    if   losstype == 'MINE_EMA':

        ma_eT = alpha*torch.sum(eT) + (1 - alpha)*ma_eT
        loss  = -(torch.sum(T) - (1/torch.sum(ma_eT)).detach()*torch.sum(eT))

    # (Slightly) biased gradient based directly on the local MI value
    elif losstype == 'MINE':
        loss  = -MI_lb
    else:
        raise Exception(__name__ + ".train_mine: Unknown loss chosen")

    return MI_lb, ma_eT, loss


def sample_batch(X, Z, weights, batch_size, dtype='numpy'):
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

    index    = np.random.choice(range(X.shape[0]), size=batch_size, replace=False)
    random   = np.random.choice(range(Z.shape[0]), size=batch_size, replace=False)

    # XZ and X(*)Z
    if dtype == 'numpy':
        joint    = np.concatenate([X[index,...],  Z[index,...]], axis=1)
        marginal = np.concatenate([X[index,...], Z[random,...]], axis=1)
    else:
        joint    = torch.cat((X[index,...],  Z[index,...]), dim=1)
        marginal = torch.cat((X[index,...], Z[random,...]), dim=1)

    return joint, marginal, weights[index]


def train(X, Z, weights, net, opt, batch_size, num_iter, alpha, losstype):
    """
    Train the network estimator

    Args:
        See estimate()
    
    Returns:
        mutual information estimates per iteration
    """
    result = []
    ma_eT  = 1.0
    
    if weights is None:
        weights = np.ones(X.shape[0])

    for i in tqdm(range(num_iter)):

        # Sample input
        joint, marginal, w = sample_batch(X=X, Z=Z, weights=weights, batch_size=batch_size)

        # Create torch variables
        joint    = torch.FloatTensor(joint)
        marginal = torch.FloatTensor(marginal)
        w        = torch.FloatTensor(w)

        if next(net.parameters()).is_cuda:
            joint    = joint.cuda()
            marginal = marginal.cuda()
            w        = w.cuda()
        
        # Call it
        MI_lb, ma_eT, l = train_mine(joint=joint, marginal=marginal, w=w, net=net, ma_eT=ma_eT, alpha=alpha, losstype=losstype)

        # Step gradient descent
        opt.zero_grad()
        autograd.backward(l)
        opt.step()

        result.append(MI_lb.detach().cpu().numpy())

    return result


def estimate(X, Z, weights=None, batch_size=100, num_iter=int(1e3), lr=1e-3, hidden_dim=96, \
    losstype='MINE_EMA', alpha=0.01, window=200, use_cuda=True, return_full=False):
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
    input_size = X.shape[1] + Z.shape[1]
    net     = MINENet(input_size=input_size, hidden_dim=hidden_dim)
    net.train() #!

    if torch.cuda.is_available() and use_cuda:
        net = net.cuda()

    opt     = optim.Adam(net.parameters(), lr=lr)
    result  = train(X=X, Z=Z, weights=weights, net=net, opt=opt, batch_size=batch_size, num_iter=num_iter, alpha=alpha, losstype=losstype)

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
