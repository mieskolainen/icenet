# Deep Learning loss functions and related tools
# 
# m.mieskolainen@imperial.ac.uk, 2024


import torch_geometric
import torch.nn as nn
import torch
import torch.nn.functional as F

from icenet.tools import aux_torch
from icefit import mine, transport

class FocalWithLogitsLoss(nn.Module):
    """
    Focal Loss with logits as input
    
    https://arxiv.org/abs/1708.02002
    """
    def __init__(self, weight=None, gamma=2, reduction="mean"):
        super(FocalWithLogitsLoss, self).__init__()
        self.weight    = weight
        self.bce       = torch.nn.functional.binary_cross_entropy_with_logits
        self.gamma     = gamma
        self.reduction = reduction
    
    def forward(self, predicted, target):
        pt           = torch.exp(-self.bce(predicted, target, reduction='none'))
        entropy_loss = self.bce(predicted, target, weight=self.weight, reduction='none')
        focal_loss   = ((1-pt)**self.gamma)*entropy_loss
        
        if   self.reduction == "none":
            return focal_loss
        elif self.reduction == "mean":
            if self.weight is None:
                return focal_loss.mean()
            else:
                return focal_loss.sum() / torch.sum(self.weight)
        else:
            return focal_loss.sum()


class LqBernoulliWithLogitsLoss(nn.Module):
    """
    L_q likelihood for the Bernoulli case
    
    https://arxiv.org/pdf/1002.4533
    """
    def __init__(self, weight=None, q=1.0, reduction="mean"):
        super(LqBernoulliWithLogitsLoss, self).__init__()
        self.weight    = weight
        self.bce       = torch.nn.functional.binary_cross_entropy_with_logits
        self.q         = q
        self.reduction = reduction
    
    def L_q(self, x):
        """
        when q -> 1, then L_q -> log(x)
        """
        return (x**(1 - self.q) - 1) / (1 - self.q)
    
    def forward(self, predicted, target):
        p    = torch.sigmoid(predicted)
        loss = - (target*self.L_q(p) + (1 - target) * self.L_q(1 - p))
        
        if self.weight is not None:
            loss = loss * self.weight

        if   self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            if self.weight is None:
                return loss.mean()
            else:
                return loss.sum() / torch.sum(self.weight)
        else:
            return loss.sum()


def loss_wrapper(model, x, y, num_classes, weights, param, y_DA=None, w_DA=None, MI=None, EPS=1e-12):
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
    
    def SWD_helper(logits):
        """
        Sliced Wasserstein reweight regularization
        """
        if 'SWD_beta' in param and param['SWD_beta'] > 0:
            
            beta  = param['SWD_beta']
            value = beta * SWD_reweight_loss(logits=logits, x=x, y=y, weights=weights,
                                    p=param['SWD_p'], num_slices=param['SWD_num_slices'],
                                    norm_weights=param['SWD_norm_weights'],
                                    mode=param['SWD_mode'])

            return {f'SWD x $\\beta = {beta}$': value}
        else:
            return {}
    
    def MI_helper(output):
        """ 
        Mutual Information regularization
        """
        if MI is not None:
            X = MI['x'].float()
            
            # Model output
            if len(output.shape) > 1:    
                Z = output[:, MI['y_dim']]
            else:
                Z = output
            
            return {f'MI x $\\beta = {MI["beta"]}$': MI_loss(X=X, Z=Z, weights=weights, MI=MI, y=y)}
        else:
            return {}
    
    def LZ_helper():
        """
        Lipschitz continuity regularization
        """
        if 'lipschitz_beta' in param and param['lipschitz_beta'] > 0:
            LZ_loss = param['lipschitz_beta'] * model.get_lipschitz_loss()
            LZ = {f"LZ x $\\beta = {param['lipschitz_beta']}$": LZ_loss}
        else:
            LZ = {}
        return LZ
    
    def LM_helper(logits):
        """
        Logit magnitude L1-regularization
        """
        if 'logit_L1_beta' in param and param['logit_L1_beta'] > 0:
            L1_loss = LOGIT_L1_loss(logits=logits, logit_beta=param['logit_L1_beta'], weights=weights)
            LRL1 = {f"LML1 x $\\lambda = {param['logit_L1_beta']}$": L1_loss}
        else:
            LRL1 = {}
        
        """
        Logit magnitude L2-regularization
        """
        if 'logit_L2_beta' in param and param['logit_L2_beta'] > 0:
            L2_loss = LOGIT_L2_loss(logits=logits, logit_beta=param['logit_L2_beta'], weights=weights)
            LRL2 = {f"LML2 x $\\lambda = {param['logit_L2_beta']}$": L2_loss}
        else:
            LRL2 = {}

        return {**LRL1, **LRL2}
    
    ## Loss functions
    
    if   param['lossfunc'] == 'binary_cross_entropy':
        
        logits = model.forward(x)
        loss   = BCE_loss(logits=logits, y=y, weights=weights)
        
        loss = {'BCE': loss, **SWD_helper(logits), **LZ_helper(), **LM_helper(logits), **MI_helper(torch.sigmoid(logits))}

    elif param['lossfunc'] == 'binary_focal_entropy':
        
        logits = model.forward(x)
        loss   = binary_focal_loss(logits=logits, y=y, gamma=param['gamma'], weights=weights)
        
        loss = {f"FE ($\\gamma = {param['gamma']}$)": loss, **SWD_helper(logits), **LZ_helper(), **LM_helper(logits), **MI_helper(torch.sigmoid(logits))}

    elif param['lossfunc'] == 'binary_Lq_entropy':
        
        logits = model.forward(x)
        loss   = Lq_binary_loss(logits=logits, y=y, q=param['q'], weights=weights)
        
        loss = {f"LQ ($q = {param['q']}$)": loss, **SWD_helper(logits), **LZ_helper(), **LM_helper(logits), **MI_helper(torch.sigmoid(logits))}
    
    elif param['lossfunc'] == 'SWD':
        
        # Reweight transport u -> v loss
        
        logits = model.forward(x)
        
        loss = SWD_reweight_loss(logits=logits, x=x, y=y, weights=weights,
                                 p=param['SWD_p'], num_slices=param['SWD_num_slices'],
                                 mode=param['SWD_mode'])
        
        loss = {'SWD': loss, **LZ_helper(), **LM_helper(logits), **MI_helper(torch.sigmoid(logits))}

    elif param['lossfunc'] == 'MSE':
        
        y_hat = model.forward(x)
        loss  = MSE_loss(y_hat=y_hat, y=y, weights=weights)
        
        loss  = {'MSE': loss, **SWD_helper(logits), **LZ_helper(), **LM_helper(y_hat), **MI_helper(y_hat)}

    elif param['lossfunc'] == 'MSE_prob':
        
        logits = model.forward(x)
        y_hat  = torch.sigmoid(logits)
        loss   = MSE_loss(y_hat=y_hat, y=y, weights=weights)
        
        loss  = {'MSE': loss, **SWD_helper(logits), **LZ_helper(), **LM_helper(logits), **MI_helper(y_hat)}
    
    elif param['lossfunc'] == 'MAE':
        
        y_hat = model.forward(x)
        loss  = MSE_loss(y_hat=y_hat, y=y, weights=weights)
        
        loss  = {'MAE': loss, **SWD_helper(logits), **LZ_helper(), **LM_helper(y_hat), **MI_helper(y_hat)}
    
    elif param['lossfunc'] == 'cross_entropy':
        """
        One-hot multiclass
        """
        
        logits   = model.forward(x)
        log_phat = F.log_softmax(logits, dim=-1)
        
        if num_classes > 2:
            loss = multiclass_cross_entropy_logprob(log_phat=log_phat, y=y, num_classes=num_classes, weights=weights)
        
        # This can handle scalar y (target) values in [0,1]
        else:
            loss = binary_cross_entropy_logprob(log_phat_0=log_phat[:,0], log_phat_1=log_phat[:,1], y=y, weights=weights)
        
        loss  = {'CE': loss, **LZ_helper(), **LM_helper(logits), **MI_helper(torch.exp(log_phat))}
    
    elif param['lossfunc'] == 'cross_entropy_with_DA':
        """
        One-hot multiclass
        """
        
        logits, logits_DA = model.forward_with_DA(x)
        
        log_phat    = F.log_softmax(logits,    dim=-1)
        log_phat_DA = F.log_softmax(logits_DA, dim=-1)
        
        # https://arxiv.org/abs/1409.7495
        CE_loss    = multiclass_cross_entropy_logprob(log_phat=log_phat,    y=y,    num_classes=num_classes, weights=weights)
        CE_DA_loss = multiclass_cross_entropy_logprob(log_phat=log_phat_DA, y=y_DA, num_classes=2, weights=w_DA)

        CE_DA_loss *= param['DA_beta']
        
        loss  = {'CE': CE_loss, f"DA x $\\beta = {param['DA_beta']}$": CE_DA_loss, **LZ_helper(), **LM_helper(logits), **MI_helper(torch.exp(log_phat))}

    elif param['lossfunc'] == 'logit_norm_cross_entropy':
        """
        One-hot multiclass
        """
        
        logits = model.forward(x)
        loss   = multiclass_logit_norm_loss(logit=logits, y=y, num_classes=num_classes, weights=weights, t=param['temperature'])
        
        loss  = {'LNCE': loss, **LZ_helper(), **LM_helper(logits), **MI_helper(torch.exp(log_phat))}

    elif param['lossfunc'] == 'focal_entropy':
        """
        One-hot multiclass
        """
        
        logits   = model.forward(x)
        log_phat = F.log_softmax(logits, dim=-1)
        loss = multiclass_focal_entropy_logprob(log_phat=log_phat, y=y, num_classes=num_classes, weights=weights, gamma=param['gamma'])
        
        loss  = {f"FE ($\\gamma = {param['gamma']}$)": loss, **LZ_helper(), **LM_helper(logits), **MI_helper(torch.exp(log_phat))}
        
    elif param['lossfunc'] == 'VAE_background_only':
        
        B_ind    = (y == 0) # Use only background to train
        xhat, z, mu, std = model.forward(x=x[B_ind, ...])
        log_loss = model.loss_kl_reco(x=x[B_ind, ...], xhat=xhat, z=z, mu=mu, std=std, beta=param['VAE_beta'])
        
        if weights is not None:
            loss = (log_loss*weights[B_ind]).sum(dim=0) / torch.sum(weights[B_ind])
        else:
            loss = log_loss.mean(dim=0)

        loss  = {f"VAE ($\\beta = {param['VAE_beta']}$)": loss, **LM_helper(xhat), **LZ_helper()}
    
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


def SWD_reweight_loss(logits, x, y, weights=None, p=1, num_slices=1000, norm_weights=True, mode='SWD'):
    """
    # Sliced Wasserstein reweight U (y==0) -> V (y==1) transport
    """
    u_idx = (y == 0)
    v_idx = (y == 1)
    
    u_values = x[u_idx, :]
    v_values = x[v_idx, :]
    
    LR = torch.exp(logits[u_idx]).squeeze() # likelihood ratio
    
    if weights is not None:
        u_weights = weights[u_idx] * LR
        v_weights = weights[v_idx] # Network output for v not needed
    else:
        u_weights = LR
        v_weights = None
    
    loss_uv = transport.sliced_wasserstein_distance(u_values=u_values, v_values=v_values,
                                                    u_weights=u_weights, v_weights=v_weights,
                                                    p=p, num_slices=num_slices,
                                                    norm_weights=norm_weights,
                                                    mode=mode)
    
    # Overall statistics scale normalization (based on class = 1) done afterwards
    # (due to possible maxevent cutoff for the SWD loss vs BCE loss due to GPU VRAM limit)
    
    if (norm_weights == False):
        if v_weights is not None:
            loss_uv = loss_uv / torch.sum(v_weights)
        else:
            loss_uv = loss_uv / torch.sum(v_idx)
    
    return loss_uv


def Lq_binary_loss(logits, y, q, weights=None):
    """
    L_q Bernoulli loss
    """
    criterion = LqBernoulliWithLogitsLoss(reduction='none', q=q, weight=weights)
    loss      = criterion(logits.squeeze(), y.squeeze().float())

    if weights is not None:
        return loss.sum() / torch.sum(weights)
    else:
        return loss.sum() / y.shape[0]

def LOGIT_L1_loss(logits, logit_beta=1.0, weights=None):
    """
    Logit magnitude L1-regularization sum |z|
    """
    w    = 1.0 if weights is None else weights
    norm = torch.linalg.vector_norm(logits, 1, dim=1).squeeze()
    loss = w * logit_beta * norm
    
    if weights is not None:
        return loss.sum() / torch.sum(weights)
    else:
        return loss.sum() / logits.shape[0]

def LOGIT_L2_loss(logits, logit_beta=1.0, weights=None):
    """
    Logit magnitude L2-regularization sum |z|^2
    """
    w    = 1.0 if weights is None else weights
    norm = torch.linalg.vector_norm(logits, 2, dim=1).squeeze()
    loss = w * logit_beta * (norm**2)
    
    if weights is not None:
        return loss.sum() / torch.sum(weights)
    else:
        return loss.sum() / logits.shape[0]

def BCE_loss(logits, y, weights=None):
    """
    Binary Cross Entropy loss
    """
    
    criterion = nn.BCEWithLogitsLoss(reduction='none', weight=weights)
    loss      = criterion(logits.squeeze(), y.squeeze().float())

    if weights is not None:
        return loss.sum() / torch.sum(weights)
    else:
        return loss.sum() / y.shape[0]

def binary_focal_loss(logits, y, gamma=1.0, weights=None):
    """
    Focal Cross Entropy loss
    """
    
    criterion = FocalWithLogitsLoss(reduction='none', gamma=gamma, weight=weights)
    loss      = criterion(logits.squeeze(), y.squeeze().float())

    if weights is not None:
        return loss.sum() / torch.sum(weights)
    else:
        return loss.sum() / y.shape[0]

def MSE_loss(y_hat, y, weights=None):
    """
    Mean squared error loss
    """
    
    w = 1.0 if weights is None else weights

    loss = - w * ((y_hat - y)**2).squeeze()

    if weights is not None:
        return loss.sum() / torch.sum(weights)
    else:
        return loss.sum() / y.shape[0]

def MAE_loss(y_hat, y, weights=None):
    """
    Mean absolute error loss
    """
    
    w = 1.0 if weights is None else weights
    
    loss = - w * (torch.abs(y_hat - y)).squeeze()

    if weights is not None:
        return loss.sum() / torch.sum(weights)
    else:
        return loss.sum() / y.shape[0]

def binary_cross_entropy_logprob(log_phat_0, log_phat_1, y, weights=None):
    """ 
    Per instance weighted binary cross entropy loss (y can be a scalar between [0,1])
    (negative log-likelihood)
    """
    w = 1.0 if weights is None else weights

    loss = - w * (y*log_phat_1 + (1 - y) * log_phat_0).squeeze()

    if weights is not None:
        return loss.sum() / torch.sum(weights)
    else:
        return loss.sum() / y.shape[0]

def multiclass_logit_norm_loss(logit, y, num_classes, weights=None, t=1.0, EPS=1e-7):
    """
    https://arxiv.org/abs/2205.09310
    """
    norms = torch.clip(torch.norm(logit, p=2, dim=-1, keepdim=True), min=EPS)
    logit_norm = torch.div(logit, norms) / t
    log_phat = F.log_softmax(logit_norm, dim=-1)
    
    return multiclass_cross_entropy_logprob(log_phat=log_phat, y=y, num_classes=num_classes, weights=weights)

def multiclass_cross_entropy_logprob(log_phat, y, num_classes, weights=None):
    """ 
    Per instance weighted cross entropy loss
    (negative log-likelihood)
    """
    if weights is not None:
        w = aux_torch.weight2onehot(weights=weights, y=y, num_classes=num_classes)
    else:
        w = 1.0

    y    = F.one_hot(y, num_classes)
    loss = - y * log_phat * w

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

def logsumexp(x, dim=-1):
    """ 
    https://en.wikipedia.org/wiki/LogSumExp
    """
    xmax, idx = torch.max(x, dim=dim, keepdim=True)
    return xmax + torch.log(torch.sum(torch.exp(x - xmax), dim=dim, keepdim=True))

def log_softmax(x, dim=-1):
    """
    Log of Softmax
    
    Args:
        x : network output without softmax
    Returns:
        logsoftmax values
    """
    log_z = logsumexp(x, dim=dim)
    y = x - log_z
    return y
