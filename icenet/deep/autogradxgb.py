# Custom pytorch-driven autograd losses for XGBoost
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import torch
from torch import Tensor
import xgboost
from tqdm import tqdm

from typing import Callable, Sequence, List, Tuple

# ------------------------------------------
from icenet import print
# ------------------------------------------

class XgboostObjective():
    def __init__(self, loss_func: Callable[[Tensor, Tensor], Tensor], mode='train', 
                 flatten_grad=False, hessian_mode='constant', hessian_const=1.0, device='cpu'):

        self.mode          = mode
        self.loss_func     = loss_func
        self.device        = device
        self.hessian_mode  = hessian_mode
        self.hessian_const = hessian_const
        self.flatten_grad  = flatten_grad

        if self.hessian_mode == 'constant':
            print(f'Using device: {self.device} | hessian_mode = {self.hessian_mode} | hessian_const = {self.hessian_const}')
        else:
            print(f'Using device: {self.device} | hessian_mode = {self.hessian_mode}')
        
    def __call__(self, preds: np.ndarray, targets: xgboost.DMatrix):

        preds_, targets_, weights_ = self.torch_conversion(preds=preds, targets=targets)

        if   self.mode == 'train':
            loss = self.loss_func(preds=preds_, targets=targets_, weights=weights_)
            return self.derivatives(loss=loss, preds=preds_)
        elif self.mode == 'eval':
            loss = self.loss_func(preds=preds_, targets=targets_, weights=weights_)
            return 'custom', loss.detach().cpu().numpy()
        else:
            raise Exception('Unknown mode (set either "train" or "eval")')

    def torch_conversion(self, preds: np.ndarray, targets: xgboost.DMatrix):
        """ Conversion from xgboost.Dmatrix object
        """
        try:
            weights = targets.get_weight()
            weights = None if weights == [] else torch.FloatTensor(weights).to(self.device)
        except:
            weights = None

        preds   = torch.FloatTensor(preds).requires_grad_().to(self.device)
        targets = torch.FloatTensor(targets.get_label()).to(self.device)

        return preds, targets, weights

    def derivatives(self, loss: Tensor, preds: Tensor):
        """ Gradient and Hessian diagonal
        """
        
        ## Gradient
        grad1 = torch.autograd.grad(loss, preds, create_graph=True)[0]
        
        ## Diagonal elements of the Hessian matrix
        
        # Constant curvature
        if   self.hessian_mode == 'constant':
            grad2 = self.hessian_const * torch.ones_like(grad1)
        
        # Squared derivative based [uncontrolled] approximation (always positive curvature)
        elif self.hessian_mode == 'squared_approx':
            grad2 = grad1 * grad1
        
        # Exact autograd
        elif self.hessian_mode == 'exact':
            
            print('Computing Hessian diagonal with exact autograd ...')
            
            """
            for i in tqdm(range(len(preds))):
                grad2_i  = torch.autograd.grad(grad1[i], preds, retain_graph=True)[0]
                grad2[i] = grad2_i[i]
            """
            
            grad2 = torch.zeros_like(preds)
            
            for i in tqdm(range(len(preds))):
                
                # A basis vector
                e_i = torch.zeros_like(preds)
                e_i[i] = 1.0
                
                # Compute the Hessian-vector product H e_i
                grad2[i] = torch.autograd.grad(grad1, preds, grad_outputs=e_i, retain_graph=True)[0][i]
            
        else:
            raise Exception(f'Unknown "hessian_mode" {self.hessian_mode}')
        
        grad1, grad2 = grad1.detach().cpu().numpy(), grad2.detach().cpu().numpy()
        
        if self.flatten_grad:
            grad1, grad2 = grad1.flatten("F"), grad2.flatten("F")
        
        return grad1, grad2
