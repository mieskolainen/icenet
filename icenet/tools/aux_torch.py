# Torch aux tools
#
# m.mieskolainen@imperial.ac.uk, 2025

import torch
import numpy as np

from icenet.tools import aux

# ------------------------------------------
from icenet import print
# ------------------------------------------


def weight2onehot(weights, y, num_classes):
    """
    Weights into one-hot encoding

    Args:
        weights     : array of weights (torch type)
        y           : targets (torch type)
        num_classes : number of classes
    """
    one_hot_weights = torch.zeros((len(weights), num_classes)).to(weights.device)
    for i in range(num_classes):
        try:
            one_hot_weights[y == i, i] = weights[y == i]
        except:
            print(f'Failed with class = {i} (zero samples)')
    return one_hot_weights


def count_parameters_torch(model):
    """
    Count the number of trainable pytorch model parameters
    """
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def load_torch_checkpoint(path='/', label='mynet', epoch=-1):
    """ Load pytorch checkpoint

    Args:
        path  : folder path
        label : model label name
        epoch : epoch index. Use -1 for the last epoch
    
    Returns:
        pytorch model
    """
    
    filename = aux.create_model_filename(path=path, label=label, epoch=epoch, filetype='.pth')
    
    # Load the model (always first to CPU memory)
    print(f'Loading model "{filename}" to CPU memory ...')

    checkpoint = torch.load(filename, map_location ='cpu')
    model      = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval() # Set it in the evaluation mode
    return model


def save_torch_model(model, optimizer, epoch, losses, filename):
    """ PyTorch model saver
    """
    def f():
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'losses': losses
        }, (filename))
    return f


def load_torch_model(model, optimizer, filename, device='cpu', param=None, load_start_epoch = False):
    """ PyTorch model loader
    """
    def f():
        print(f'Loading model to "{device}" memory ...')
        checkpoint = torch.load(filename, map_location = device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        if param is not None and load_start_epoch:
            param['start_epoch'] = checkpoint['epoch']
    
    return f
