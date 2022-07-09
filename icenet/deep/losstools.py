# Deep Learning loss functions and related tools
# 
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk


import torch_geometric
import torch.nn as nn
import torch
import torch.nn.functional as F

from icenet.tools import aux_torch


def loss_wrapper(model, x, y, num_classes, weights, param):
    """
    A wrapper function to loss functions
    
    Note:
        log-likelihood functions can be weighted linearly, due to
        \\prod_i p_i(x_i; \\theta)**w_i ==\\log==> \\sum_i w_i \\log p_i(x_i; \\theta)
    """
    if   param['lossfunc'] == 'cross_entropy':
        log_phat = model.softpredict(x)
        y = y.type(torch.int64)
        return multiclass_cross_entropy_logprob(log_phat=log_phat, y=y, num_classes=num_classes, weights=weights)

    elif param['lossfunc'] == 'cross_entropy_per_edge':

        # --------------------------------------------
        # Synthetic negative edge sampling
        if param['negative_sampling']:
            
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

        log_phat = model.softpredict(x)
        return multiclass_cross_entropy_logprob(log_phat=log_phat, y=y, num_classes=num_classes, weights=weights)
        
    elif param['lossfunc'] == 'logit_norm_cross_entropy':
        logit = model.forward(x)
        y = y.type(torch.int64)
        return multiclass_logit_norm_loss(logit=logit, y=y, num_classes=num_classes, weights=weights, t=param['temperature'])
        
    elif param['lossfunc'] == 'focal_entropy':
        log_phat = model.softpredict(x)
        y = y.type(torch.int64)
        return multiclass_focal_entropy_logprob(log_phat=log_phat, y=y, num_classes=num_classes, weights=weights, gamma=param['gamma'])
        
    elif param['lossfunc'] == 'VAE_background_only':
        ind      = (y == 0) # Use only background to train
        xhat, z, mu, std = model.forward(x=x[ind, ...])
        log_loss = model.loss_kl_reco(x=x[ind, ...], xhat=xhat, z=z, mu=mu, std=std, beta=param['VAE_beta'])
        
        if weights is not None:
            return (log_loss*weights[ind]).sum(dim=0) / torch.sum(weights[ind])
        else:
            return log_loss.mean(dim=0)
    
    else:
        print(__name__ + f".loss_wrapper: Error with an unknown lossfunc {param['lossfunc']}")


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
    norms = torch.norm(logit, p=2, dim=-1, keepdim=True) + EPS
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

def multiclass_cross_entropy(phat, y, num_classes, weights=None, EPS=1e-30):
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
    loss = - y*torch.log(phat + EPS) * w

    if weights is not None:
        return loss.sum() / torch.sum(weights)
    else:
        return loss.sum() / y.shape[0]

def multiclass_focal_entropy_logprob(log_phat, y, num_classes, gamma, weights=None, EPS=1e-30) :
    """
    Per instance weighted 'focal entropy loss'
    https://arxiv.org/pdf/1708.02002.pdf
    """
    if weights is not None:
        w = aux_torch.weight2onehot(weights=weights, y=y, num_classes=num_classes)
    else:
        w = 1.0

    y = F.one_hot(y, num_classes)
    loss = -y * torch.pow(1 - phat, gamma) * torch.log(phat + EPS) * w

    if weights is not None:
        return loss.sum() / torch.sum(weights)
    else:
        return loss.sum() / y.shape[0]

def multiclass_focal_entropy(phat, y, num_classes, gamma, weights=None, EPS=1e-30) :
    """
    Per instance weighted 'focal entropy loss'
    https://arxiv.org/pdf/1708.02002.pdf
    """
    if weights is not None:
        w = aux_torch.weight2onehot(weights=weights, y=y, num_classes=num_classes)
    else:
        w = 1.0

    y = F.one_hot(y, num_classes)
    loss = -y * torch.pow(1 - phat, gamma) * torch.log(phat + EPS) * w

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
