# Deep Learning (training) aux tool functions
# 
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import torch
import torch.nn as nn
import math

class Multiply(nn.Module):
    """
    Multiplication with a non-learnable constant alpha
    """
    def __init__(self, alpha):
        super().__init__()
        self.alpha =  alpha
    
    def forward(self, x):
        x = torch.mul(x, self.alpha)
        return x


def grad_norm(module: nn.Module):
    """
    Compute the total (Frobenius) norm for the gradients of a torch network
    
    Args:
        module: torch network
    Returns:
        total gradient norm
    """
    parameters = module.parameters()
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = list(filter(lambda p: p.grad is not None, parameters))
    
    # Loop over all parameters
    norm = 0
    for p in parameters:
        norm += p.grad.data.norm(2).item() ** 2
    norm = np.sqrt(norm)
    
    return norm


def adaptive_gradient_clipping_(main_module: nn.Module, MI_module: nn.Module, EPS=1E-9):
    """
    Adaptively clip the gradient from the mutual information module, so that its Frobenius norm
    is at most that of the gradient from the main network.
    
    See: https://arxiv.org/abs/1801.04062 (Appendix)
    
    Args:
        generator_module:  Generator/classifier/regressor... main task network (nn.Module)
        mi_module:         MI regulator network (nn.Module)
    """
    norm_main = grad_norm(main_module)
    norm_MI   = grad_norm(MI_module)

    min_norm  = np.minimum(norm_main, norm_MI)
    
    parameters = list(filter(lambda p: p.grad is not None, MI_module.parameters()))
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    # Scale gradients
    scale = min_norm / (norm_MI + EPS)
    for p in parameters:
        p.grad.data.mul_(scale)


init_funcs = {
    1: lambda x: torch.nn.init.normal_(x, mean=0.0, std=1.0),  # bias terms
    2: lambda x: torch.nn.init.xavier_normal_(x,   gain=1.0),  # weight terms
    3: lambda x: torch.nn.init.xavier_uniform_(x,  gain=1.0),  # conv1D filter
    4: lambda x: torch.nn.init.xavier_uniform_(x,  gain=1.0),  # conv2D filter
    "default": lambda x: torch.nn.init.constant(x, 1.0),       # others
}


def weights_init_all(model, init_funcs):
    """
    Examples:
        model = MyNet()
        weights_init_all(model, init_funcs)
    """
    for p in model.parameters():
        init_func = init_funcs.get(len(p.shape), init_funcs["default"])
        init_func(p)


def weights_init_uniform_rule(m):
    """ Initializes module weights from uniform [-a,a]
    """
    classname = m.__class__.__name__

    # Linear layers
    if classname.find('Linear') != -1:
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def weights_init_normal(m):
    """ Initializes module weights from normal distribution
    with a rule sigma ~ 1/sqrt(n)
    """
    classname = m.__class__.__name__
    
    # Linear layers
    if classname.find('Linear') != -1:
        y = m.in_features
        m.weight.data.normal_(0.0, 1/np.sqrt(y))
        m.bias.data.fill_(0)


def set_scheduler(optimizer: dict, param: dict):
    """
    Args:
        optimizer: optimizers for different models
        param:     setup parameters
    
    Returns:
        torch scheduler
    """
    
    print(__name__ + f'.set_scheduler: ')
    print(param)
    print('')
    
    if param['type']== 'cos':
        
        period = param['period']
        
        # "Cosine cycle scheme"
        lf = lambda x: (((1 + math.cos(x * math.pi / period)) / 2) ** 1.0) * 0.9 + 0.1
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        
    elif param['type'] == 'warm-cos':
        
        T0      = param['period']
        eta_min = param['eta_min'] # ~ lr / 10
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T0, eta_min=eta_min)
    
    elif param['type'] == 'exp':
        
        gamma      = param['gamma']      # ~ (1 - 1e-5)
        last_epoch = param['last_epoch'] # -1
        
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=last_epoch)
    
    elif param['type'] == 'step':
        
        step_size = param['step_size']  # ~ 150
        gamma     = param['gamma']      # ~ 0.1
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    
    elif param['type'] == 'constant':
        
        factor      = param['factor']      # ~ 1
        total_iters = param['total_iters'] # ~ 10000
        
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor, total_iters)
    
    else:
        raise Exception(__name__ + f'.set_scheduler: Unknown scheduler_type: {param["type"]}')
    
    return scheduler
