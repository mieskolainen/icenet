# Derministic and Stochastic Flow Transport
#
# m.mieskolainen@imperial.ac.uk, 2025

import os
import glob
import torch
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from torch.utils import data

from icenet.deep  import optimize, flows, deeptools
from icenet.tools import aux, aux_torch

class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, C, Y, W):
        """ Initialization """
        self.X = X  # flow-space
        self.C = C  # conditional-space
        self.Y = Y  # class label
        self.W = W  # event weight

    def __len__(self):
        """ Return the total number of samples """
        return self.X.shape[0]

    def __getitem__(self, index):
        """ Generates one sample of data """
        
        if self.C is not None:
            # Use ellipsis ... to index over scalar [,:], vector [,:,:], tensor [,:,:,..,:] indices
            return self.X[index,...], self.C[index,...], self.Y[index], self.W[index]
        else:
            return self.X[index,...], None, self.Y[index], self.W[index]
    
def prepare_dataset(x: np.ndarray, c: np.ndarray, y: np.ndarray, w: np.ndarray):
    
    X = torch.tensor(x, dtype=torch.float)
    Y = torch.tensor(y, dtype=torch.long)
    W = torch.tensor(w, dtype=torch.float)
    
    # ---------------------------------------------------------------------
    # Append transport condition based on class (domain) label
    
    if c is not None:
        
        C = torch.tensor(c, dtype=torch.float)
        
        # Ensure it is column shaped and concatenate
        Y_col = Y.view(-1, 1).to(dtype=C.dtype)
        C = torch.cat([C, Y_col], dim=1)
    # ---------------------------------------------------------------------
    
    return Dataset(X=X, C=C, Y=Y, W=W)


def create_model(model_param, rngseed=0):
    """ Construct the network object.
    
    Args:
        param : parameters
    Returns:
        model : model object
    """
    
    # For random permutations
    aux.set_random_seed(rngseed)
    
    print(model_param)
    
    model  = flows.CouplingFlow(**model_param)
    
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Print model information    
    print('{}'.format(model))
    print(f'Parameters={params}')

    return model

def train(model, optimizer, scheduler, trn_loader, val_loader,
          param, modeldir, save_name):
    
    model, device = optimize.model_to_cuda(model, param['device'])

    opt_param = param['opt_param']
    
    # TensorboardX
    if 'tensorboard' in param and param['tensorboard']:
        from tensorboardX import SummaryWriter
        
        for f in glob.glob(os.path.join(modeldir, 'events.out.tfevents.*')):
            os.remove(f) # Clean old logs
        
        writer = SummaryWriter(modeldir)

    trn_losses = {}
    val_losses = {}
    trn_losses_list = []
    val_losses_list = []
    
    # Training loop
    for epoch in tqdm(range(opt_param['epochs']), ncols = 88):
        
        model.train() # !

        trn_loss = 0.0
        denom    = 0.0
        
        # Scheduled noise regularization
        sigma2 = None
        if 'noise_reg' in opt_param and opt_param['noise_reg'] > 0.0:
            noise_reg = opt_param['noise_reg']
            sigma2 = noise_reg * deeptools.sigmoid_schedule(t=epoch, N_max=opt_param['epochs'])
            
            print(f'Noise reg. sigma2 = {sigma2:0.3E}')
        
        for x, c, y, w in trn_loader:
            
            # Transfer to GPU
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            if c is not None:
                c = c.to(device, dtype=torch.float32, non_blocking=True)
            w = w.to(device, dtype=torch.float32, non_blocking=True)
            
            # ---------------------------------------------------------------
            # Add scheduled noise regularization
            
            if sigma2 is not None:
                x = np.sqrt(1 - sigma2)*x + np.sqrt(sigma2)*torch.randn_like(x)
            # ---------------------------------------------------------------    

            # ------------------------------------------------------------
            # Catenate 1D-random condition for Kantorovich map
            if param['stochastic'] and c is not None:
                c = torch.cat([c, torch.randn(c.shape[0], 1, device=device, dtype=torch.float32)], dim=1)
            # ------------------------------------------------------------
            
            # Compute loss [per batch w normalized]
            loss = (w * model.loss(x=x, cond=c)).sum() / w.sum()
            
            # Zero gradients, calculate loss, calculate gradients and update parameters
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=param['opt_param']['clip_norm'])
            optimizer.step()

            trn_loss += loss.item()
            denom    += 1
        
        trn_loss = trn_loss / denom

        # ----------------------------------------------------------------
        # Compute validation loss
        
        if epoch == 0 or (epoch % param['savemode']) == 0:
            
            model.eval() # !

            val_loss = 0.0
            denom    = 0.0
            
            with torch.no_grad():
            
                for x, c, y, w in val_loader:
                    
                    # Transfer to GPU
                    x = x.to(device, dtype=torch.float32, non_blocking=True)
                    if c is not None:
                        c = c.to(device, dtype=torch.float32, non_blocking=True)
                    w = w.to(device, dtype=torch.float32, non_blocking=True)
                    
                    # ------------------------------------------------------------
                    # Catenate 1D-random condition for Kantorovich map
                    if param['stochastic'] and c is not None:
                        c = torch.cat([c, torch.randn(c.shape[0], 1, device=device, dtype=torch.float32)], dim=1)
                    # ------------------------------------------------------------
                    
                    # Compute loss [per batch w normalized]
                    loss = (w  * model.loss(x=x, cond=c)).sum() / w.sum()
                    
                    val_loss += loss.item()
                    denom    += 1
            
            val_loss = val_loss / denom
        
        # ----------------------------------------------------------------
        
        # Save metrics
        
        ## ** Save values **
        optimize.trackloss(loss={'NLL': trn_loss}, loss_history=trn_losses)
        optimize.trackloss(loss={'NLL': val_loss}, loss_history=val_losses)
        
        trn_losses_list.append(trn_loss)
        val_losses_list.append(val_loss)
        
        # Step scheduler
        scheduler.step()
        
        print('Epoch {:3d}/{:3d} | Train: loss: {:4.3f} | Validation: loss: {:4.3f} | lr: {:0.3E}'.format(
            epoch+1,
            param['opt_param']['epochs'],
            trn_loss,
            val_loss,
            scheduler.get_last_lr()[0])
        )
        
        # Save
        if epoch == 0 or (epoch % param['savemode']) == 0:
            
            filename = f'{modeldir}/{save_name}_{epoch}.pth'
            aux_torch.save_torch_model(
                model     = model,
                optimizer = optimizer,
                epoch     = epoch,
                losses    = {'trn_losses': trn_losses_list, 'val_losses': val_losses_list},
                filename  = filename
            )() # Note ()

        if 'tensorboard' in param and param['tensorboard']:
            writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
            writer.add_scalar('loss/validation', val_loss, epoch)
            writer.add_scalar('loss/train', trn_loss, epoch)
        
    return trn_losses, val_losses

def load_model(param, modelname, modeldir, device):
    """ Load model from the disk
    """
    
    model      = create_model(param['model_param'])
    filename   = aux.create_model_filename(path=modeldir, label=modelname, \
        epoch=param['readmode'], filetype='.pth')
    checkpoint = torch.load(filename, map_location='cpu')
    
    model.load_state_dict(checkpoint['model'])
    model, device = optimize.model_to_cuda(model, device_type=device)
    
    model.eval() # Turn on eval mode!

    return model, device