# Deep Learning optimization functions
# 
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk


import torch
import uproot
import numpy as np
import sklearn

from matplotlib import pyplot as plt
import pdb
from tqdm import tqdm, trange

import torch.optim as optim
from   torch.autograd import Variable
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

from icenet.tools import aux



init_funcs = {
    1: lambda x: torch.nn.init.normal_(x, mean=0., std=1.), # can be bias
    2: lambda x: torch.nn.init.xavier_normal_(x, gain=1.),  # can be weight
    3: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.), # can be conv1D filter
    4: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.), # can be conv2D filter
    "default": lambda x: torch.nn.init.constant(x, 1.),     # everything else
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

    # Liner layers
    if classname.find('Linear') != -1:
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def weights_init_normal(m):
    """ Initializes module weights from normal distribution
    with ad-hoc rule sigma ~ 1/sqrt(n)
    """
    classname = m.__class__.__name__
    
    # Linear layers
    if classname.find('Linear') != -1:
        y = m.in_features
        m.weight.data.normal_(0.0, 1/np.sqrt(y))
        m.bias.data.fill_(0)


def multiclass_cross_entropy(p_hat, y, N_classes, weights, EPS = 1e-15) :
    """ Per instance weighted cross entropy loss
    (negative log-likelihood)
    """
    
    y = F.one_hot(y, N_classes)

    # Protection
    loss = -y*torch.log(p_hat + EPS) * weights
    loss = loss.sum() / y.shape[0]

    return loss


def multiclass_focal_entropy(p_hat, y, N_classes, weights, gamma, EPS = 1e-15) :
    """ Per instance weighted 'focal entropy loss'
    https://arxiv.org/pdf/1708.02002.pdf

    """

    y = F.one_hot(y, N_classes)
    loss = -y * torch.pow(1 - p_hat, gamma) * torch.log(p_hat + EPS) * weights
    loss = loss.sum() / y.shape[0]
    
    return loss


def log_sum_exp(x):
    """ 
    http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    """

    b, _ = torch.max(x, 1)
    # b.size() = [N, ], unsqueeze() required
    y = b + torch.log(torch.exp(x - b.unsqueeze(dim=1).expand_as(x)).sum(1))
    # y.size() = [N, ], no need to squeeze()
    return y


def train(model, X_trn, Y_trn, X_val, Y_val, trn_weights, param) :
    """
    Main training loop
    """
    
    if param['device'] == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
    else:
        device = param['device']
    print(__name__ + f'.train: computing device <{device}> chosen')    

    model  = model.to(device)
    X_trn, X_val = X_trn.to(device), X_val.to(device)
    Y_trn, Y_val = Y_trn.to(device), Y_val.to(device)
    
    
    # Input checks
    # more than 1 class sample required, will crash otherwise
    if len(np.unique(Y_trn.detach().numpy())) <= 1:
        raise Exception(__name__ + '.train: Number of classes in ''Y_trn'' <= 1')
    if len(np.unique(Y_val.detach().numpy())) <= 1:
        raise Exception(__name__ + '.train: Number of classes in ''Y_val'' <= 1')

    # --------------------------------------------------------------------

    print(__name__ + '.train: Training samples = {}, Validation samples = {}'.format(X_trn.shape[0], X_val.shape[0]))

    # Prints the weights and biases
    print(model)
        
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(__name__ + f'.train: Number of free parameters = {params}')
    
    # --------------------------------------------------------------------
    ### Weight initialization
    print(__name__ + f'.train: Weight initialization')
    #model.apply(weights_init_normal_rule)
    # --------------------------------------------------------------------

    print('')

    # Class fractions
    YY = Y_trn.numpy()
    frac = []
    for i in range(model.C):
        frac.append( sum(YY == i) / YY.size )

    print(__name__ + '.train: Class fractions in the training sample: ')

    ## Classes
    for i in range(len(frac)):
        print(f' {i:4d} : {frac[i]:5.6f} ({sum(YY == i)} counts)')
    print(__name__ + f'.train: Found {len(np.unique(Y_trn))} / {model.C} classes in the training sample')
    
    # Define the optimizer
    if   param['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = param['learning_rate'], amsgrad = True)
    elif param['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = param['learning_rate'])
    else:
        raise Exception(__name__ + f'.train: Unknown optimizer {param["optimizer"]} (use "Adam" or "SGD")')

    # List to store losses
    losses   = []
    trn_aucs = []
    val_aucs = []
    
    print(__name__ + '.train: Training loop ...')

    # Change the shape
    one_hot_weights = np.zeros((len(trn_weights), model.C))
    for i in range(model.C):
        one_hot_weights[Y_trn == i, i] = trn_weights[Y_trn == i]
    one_hot_weights = torch.tensor(one_hot_weights, dtype=torch.float32)

    # Epoch loop
    for epoch in tqdm(range(param['epochs']), ncols = 60):

        # X is a torch Variable
        permutation = torch.randperm(X_trn.size()[0])

        # Minibatch loop
        sumloss = 0
        nbatch  = 0
        for i in range(0, X_trn.size()[0], param['batch_size']):
            
            indices = permutation[i:i + param['batch_size']]
            
            # Vectors
            if (len(X_trn.shape) == 2):
                batch_x, batch_y = X_trn[indices,:], Y_trn[indices]
            
            # Matrices / Sets of vectors
            if (len(X_trn.shape) == 3):
                batch_x, batch_y = X_trn[indices,:,:], Y_trn[indices]                
            
            # Tensors / Sets of matrices
            if (len(X_trn.shape) == 4):
                batch_x, batch_y = X_trn[indices,:,:,:], Y_trn[indices]

            # Predict probabilities
            phat = model.softpredict(batch_x)

            # Evaluate loss
            loss = 0
            if   param['lossfunc'] == 'cross_entropy':
                loss = multiclass_cross_entropy(phat, batch_y, model.C, one_hot_weights[indices])
            elif param['lossfunc'] == 'focal_entropy':
                loss = multiclass_focal_entropy(phat, batch_y, model.C, one_hot_weights[indices], param['gamma'])
            elif param['lossfunc'] == 'inverse_focal':
                loss = multiclass_inverse_focal(phat, batch_y, model.C, one_hot_weights[indices], param['gamma'])
            else:
                print(__name__ + '.train: Error with unknown lossfunc ')

            # ------------------------------------------------------------
            optimizer.zero_grad() # Zero gradients
            loss.backward()       # Compute gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # Clip gradient for NaN problems

            # Update parameters
            optimizer.step()

            ### Save metrics
            sumloss += loss.item() # item() important for performance
            nbatch  += 1

        avgloss = sumloss / nbatch
        losses.append(avgloss)

        # ================================================================
        # TEST AUC PERFORMANCE SPARSILY (SLOW -- IMPROVE PERFORMANCE)
        if (epoch % 5 == 0) :

            if model.C == 2: # Only two class problems

                SIGNAL_ID = 1

                trn_metric = aux.Metric(y_true = Y_trn.detach().numpy(),
                    y_soft = model.softpredict(X_trn).detach().numpy()[:, SIGNAL_ID], valrange=[0,1])

                val_metric = aux.Metric(y_true = Y_val.detach().numpy(),
                    y_soft = model.softpredict(X_val).detach().numpy()[:, SIGNAL_ID], valrange=[0,1])

                trn_aucs.append(trn_metric.auc)
                val_aucs.append(val_metric.auc)

            if model.C  > 2: # Multiclass problems

                class_labels = np.arange(model.C)
                
                trn_metric_auc = sklearn.metrics.roc_auc_score(y_true = Y_trn.detach().numpy(),
                    y_score = model.softpredict(X_trn).detach().numpy(), average="weighted", multi_class='ovo', labels=class_labels)
                trn_aucs.append(trn_metric_auc)

                # -----------------------------------------------------------------

                val_metric_auc = sklearn.metrics.roc_auc_score(y_true = Y_val.detach().numpy(),
                    y_score = model.softpredict(X_val).detach().numpy(), average="weighted", multi_class='ovo', labels=class_labels)
                val_aucs.append(val_metric_auc)
                
                print('Epoch = {} : train loss = {:.3f} [trn AUC = {:.3f}, val AUC = {:.3f}]'. format(epoch, avgloss, trn_aucs[-1], val_aucs[-1]))
            
            # Just print the loss
            else:
                print('Epoch = {} : train loss = {:.3f}'. format(epoch, avgloss)) 

    return model, losses, trn_aucs, val_aucs
