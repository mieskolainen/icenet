# Deep Learning loss functions and related tools
# 
# m.mieskolainen@imperial.ac.uk, 2024


import torch_geometric
import torch.nn as nn
import torch
import torch.nn.functional as F

from icenet.tools import aux_torch
from icefit import mine


def loss_wrapper(model, x, y, num_classes, weights, param, y_DA=None, weights_DA=None, MI=None, EPS=1e-12):
    """
    A wrapper function to loss functions
    
    Note:
        log-likelihood functions can be weighted linearly, due to
        \\prod_i p_i(x_i; \\theta)**w_i ==\\log==> \\sum_i w_i \\log p_i(x_i; \\theta)
    """

    # --------------------------------------------
    # Synthetic negative edge sampling
    if ('negative_sampling' in param) and param['negative_sampling']:
        
        neg_edge_index  = torch_geometric.utils.negative_sampling(
            edge_index      = x.edge_index,          num_nodes = x.x.shape[0],
            num_neg_samples = x.edge_index.shape[1], method='sparse'
        )
        
        # Construct new combined (artificial) graph
        x.edge_index = torch.cat([x.edge_index, neg_edge_index], dim=-1).to(x.x.device)
        x.y          = torch.cat([x.y, x.y.new_zeros(size=(neg_edge_index.shape[1],))], dim=0).to(x.x.device)

        y            = x.y
        weights      = None # TBD. Could re-compute a new set of edge weights 
    # --------------------------------------------
    
    def MI_helper(log_phat):
        if MI is not None:
            X = MI['x'].float()
            Z = torch.exp(log_phat[:, MI['y_dim']])  # Classifier output
            return {f'MI x $\\beta = {MI["beta"]}$': MI_loss(X=X, Z=Z, weights=weights, MI=MI, y=y)}
        else:
            return {}
    
    if  param['lossfunc'] == 'cross_entropy':
        log_phat = torch.log(torch.clip(model.softpredict(x), min=EPS))
        
        if num_classes > 2:
            loss = multiclass_cross_entropy_logprob(log_phat=log_phat, y=y, num_classes=num_classes, weights=weights)
        
        # This can handle scalar y values in [0,1]
        else:
            loss = binary_cross_entropy_logprob(log_phat_0=log_phat[:,0], log_phat_1=log_phat[:,1], y=y, weights=weights)

        loss  = {'CE': loss, **MI_helper(log_phat)}

    elif param['lossfunc'] == 'cross_entropy_with_DA':

        x, x_DA     = model.forward_with_DA(x)

        log_phat    = F.log_softmax(x,    dim=-1)
        log_phat_DA = F.log_softmax(x_DA, dim=-1)

        # https://arxiv.org/abs/1409.7495
        CE_loss    = multiclass_cross_entropy_logprob(log_phat=log_phat,    y=y,    num_classes=num_classes, weights=weights)
        CE_DA_loss = multiclass_cross_entropy_logprob(log_phat=log_phat_DA, y=y_DA, num_classes=2, weights=weights_DA)

        loss  = {'CE': CE_loss, 'DA': CE_DA_loss, **MI_helper(log_phat)}

    elif param['lossfunc'] == 'logit_norm_cross_entropy':
        logit = model.forward(x)
        loss  = multiclass_logit_norm_loss(logit=logit, y=y, num_classes=num_classes, weights=weights, t=param['temperature'])
        
        loss  = {'LNCE': loss, **MI_helper(log_phat)}

    elif param['lossfunc'] == 'focal_entropy':
        log_phat = torch.log(torch.clip(model.softpredict(x), min=EPS))
        loss = multiclass_focal_entropy_logprob(log_phat=log_phat, y=y, num_classes=num_classes, weights=weights, gamma=param['gamma'])
        
        loss  = {'FE': loss, **MI_helper(log_phat)}

    elif param['lossfunc'] == 'VAE_background_only':
        B_ind    = (y == 0) # Use only background to train
        xhat, z, mu, std = model.forward(x=x[B_ind, ...])
        log_loss = model.loss_kl_reco(x=x[B_ind, ...], xhat=xhat, z=z, mu=mu, std=std, beta=param['VAE_beta'])
        
        if weights is not None:
            loss = (log_loss*weights[B_ind]).sum(dim=0) / torch.sum(weights[B_ind])
        else:
            loss = log_loss.mean(dim=0)

        loss  = {'KL_RECO': loss}
        
    else:
        print(__name__ + f".loss_wrapper: Error with an unknown lossfunc {param['lossfunc']}")

    return loss


def MI_loss(X, Z, weights, MI, y):
    """
    Neural Mutual Information regularization
    """
    #if len(MI['classes']) != 1:
    #    # To extend this, we should have separate MI nets/models for each class
    #    raise Exception(__name__ + f'.MI_loss: Support currently for one class only (or all inclusive with = [None])')

    if weights is not None:
        weights = weights / torch.sum(weights)
    else:
        weights = torch.ones(len(X)).to(X.device)

    loss = 0
    MI['network_loss'] = 0

    for k in range(len(MI['classes'])):
        c = MI['classes'][k]

        if c == None:
            ind = (y != -1) # All classes
        else:
            ind = (y == c)

        joint, marginal, w             = mine.sample_batch(X=X[ind], Z=Z[ind], weights=weights[ind], batch_size=None, device=X.device)
        MI_lb, MI['ma_eT'][k], loss_MI = mine.compute_mine(joint=joint, marginal=marginal, w=w,
                                            model=MI['model'][k], ma_eT=MI['ma_eT'][k], alpha=MI['alpha'], losstype=MI['losstype'])
        
        # Used by the total optimizer
        loss  = loss + MI['beta'][k] * MI_lb
        
        # Used by the MI-net torch optimizer
        MI['network_loss'] = MI['network_loss'] + loss_MI

        # ** For diagnostics ** 
        MI['MI_lb'][k]     = MI_lb.item()

    # Used by the main optimizer optimizing total cost ~ main loss + MI + ...
    return loss


def binary_cross_entropy_logprob(log_phat_0, log_phat_1, y, weights=None):
    """ 
    Per instance weighted binary cross entropy loss (y can be a scalar between [0,1])
    (negative log-likelihood)
    
    Numerically more stable version.
    """
    if weights is None:
        w = 1.0
    else:
        w = weights

    loss = - w * (y*log_phat_1 + (1 - y) * log_phat_0)

    if weights is not None:
        return loss.sum() / torch.sum(weights)
    else:
        return loss.sum() / y.shape[0]


def logsumexp(x, dim=-1):
    """ 
    https://en.wikipedia.org/wiki/LogSumExp
    """
    xmax, idx = torch.max(x, dim=dim, keepdim=True)
    return xmax + torch.log(torch.sum(torch.exp(x - xmax), dim=dim, keepdim=True))


def log_softmax(x, dim=-1):
    """
    Log of softmax
    
    Args:
        x : network output without softmax
    Returns:
        logsoftmax values
    """
    log_z = logsumexp(x, dim=dim)
    y = x - log_z
    return y

def multiclass_logit_norm_loss(logit, y, num_classes, weights=None, t=1.0, EPS=1e-7):
    """
    https://arxiv.org/abs/2205.09310
    """
    norms = torch.clip(torch.norm(logit, p=2, dim=-1, keepdim=True), min=EPS)
    logit_norm = torch.div(logit, norms) / t
    log_phat = F.log_softmax(logit_norm, dim=-1) # Numerically more stable than pure softmax
    
    return multiclass_cross_entropy_logprob(log_phat=log_phat, y=y, num_classes=num_classes, weights=weights)

def multiclass_cross_entropy_logprob(log_phat, y, num_classes, weights=None):
    """ 
    Per instance weighted cross entropy loss
    (negative log-likelihood)
    
    Numerically more stable version.
    """
    if weights is not None:
        w = aux_torch.weight2onehot(weights=weights, y=y, num_classes=num_classes)
    else:
        w = 1.0

    y    = F.one_hot(y, num_classes)
    loss = - y*log_phat * w

    if weights is not None:
        return loss.sum() / torch.sum(weights)
    else:
        return loss.sum() / y.shape[0]

def multiclass_cross_entropy(phat, y, num_classes, weights=None, EPS=1e-12):
    """
    Per instance weighted cross entropy loss
    (negative log-likelihood)
    """
    if weights is not None:
        w = aux_torch.weight2onehot(weights=weights, y=y, num_classes=num_classes)
    else:
        w = 1.0

    y = F.one_hot(y, num_classes)

    # Protection
    loss = - y*torch.log(torch.clip(phat, min=EPS)) * w
    
    if weights is not None:
        return loss.sum() / torch.sum(weights)
    else:
        return loss.sum() / y.shape[0]

def multiclass_focal_entropy_logprob(log_phat, y, num_classes, gamma, weights=None) :
    """
    Per instance weighted 'focal entropy loss'
    https://arxiv.org/pdf/1708.02002.pdf
    """
    if weights is not None:
        w = aux_torch.weight2onehot(weights=weights, y=y, num_classes=num_classes)
    else:
        w = 1.0

    y = F.one_hot(y, num_classes)
    loss = -y * torch.pow(1 - torch.exp(log_phat), gamma) * log_phat * w
    
    if weights is not None:
        return loss.sum() / torch.sum(weights)
    else:
        return loss.sum() / y.shape[0]

def multiclass_focal_entropy(phat, y, num_classes, gamma, weights=None, EPS=1e-12) :
    """
    Per instance weighted 'focal entropy loss'
    https://arxiv.org/pdf/1708.02002.pdf
    """
    if weights is not None:
        w = aux_torch.weight2onehot(weights=weights, y=y, num_classes=num_classes)
    else:
        w = 1.0

    y    = F.one_hot(y, num_classes)
    loss = -y * torch.pow(1 - phat, gamma) * torch.log(torch.clip(phat, min=EPS)) * w

    if weights is not None:
        return loss.sum() / torch.sum(weights)
    else:
        return loss.sum() / y.shape[0]

def log_sum_exp(x):
    """ 
    http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    """

    b, _ = torch.max(x, 1)
    # b.size() = [N, ], unsqueeze() required
    y = b + torch.log(torch.exp(x - b.unsqueeze(dim=1).expand_as(x)).sum(1))
    # y.size() = [N, ], no need to squeeze()
    return y


class _ECELoss(nn.Module):
    """
    The Expected Calibration Error of a model.
    
    In each bin, compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    Return a weighted average of the gaps, based on the number of samples in each bin.
    
    Reference: Naeini et al.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." (2015)
    """
    def __init__(self, n_bins=15):
        """
        Args:
            n_bins: number of confidence interval bins (int)
        """
        super(_ECELoss, self).__init__()
        bin_bounds      = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_bounds[:-1]
        self.bin_uppers = bin_bounds[1:]

    def forward(self, logits, labels, t=1.0):
        """
        Args:
            logits: network output logits (not softmax probabilities)
            labels: ground truth labels
            t:      temperature parameter
        Returns:
            ece value
        """
        probs      = F.softmax(logits/t, dim=1)
        conf, pred = torch.max(probs, 1)
        acc = pred.eq(labels)
        ece = torch.zeros(1, device=logits.device)

        # Compute |confidence - accuracy| in each bin
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):

            in_bin = conf.gt(bin_lower.item()) * conf.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()

            if prop_in_bin.item() > 0:
                acc_in_bin = acc[in_bin].float().mean()
                avg_conf_in_bin = conf[in_bin].mean()
                ece += torch.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin

        return ece
