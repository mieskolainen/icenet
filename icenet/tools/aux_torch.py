# Torch aux tools
#
# m.mieskolainen@imperial.ac.uk, 2021

import torch

from icenet.tools import aux

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
    print(__name__ + f'.load_torch_checkpoint: Loading model {filename} to CPU memory ...')

    checkpoint = torch.load(filename, map_location ='cpu')
    model      = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval() # Set it in the evaluation mode
    return model


def save_torch_model(model, optimizer, epoch, filename):
    """ PyTorch model saver
    """
    def f():
        print('Saving model..')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }, (filename))
    return f


def load_torch_model(model, optimizer, filename, load_start_epoch = False):
    """ PyTorch model loader
    """
    def f():
        print(__name__ + '.load_torch_model: Loading model to CPU memory ...')
        checkpoint = torch.load(filename, map_location = 'cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        if load_start_epoch:
            param.start_epoch = checkpoint['epoch']
    return f