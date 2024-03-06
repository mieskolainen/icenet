# Custom pytorch-driven autograd losses for XGBoost
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import torch
from torch import Tensor
import xgboost
from tqdm import tqdm

from typing import Callable, Sequence, List, Tuple

class XgboostObjective():
    def __init__(self, loss_func: Callable[[Tensor, Tensor], Tensor], mode='train', loss_sign=1, 
                 flatten_grad=False, skip_hessian=False, device='cpu'):

        self.mode         = mode
        self.loss_func    = loss_func
        self.loss_sign    = loss_sign
        self.device       = device
        self.skip_hessian = skip_hessian
        self.flatten_grad = flatten_grad
    
        print(__name__ + f'.__init__: Using device: {device}')
    
    def __call__(self, preds: np.ndarray, targets: xgboost.DMatrix):

        preds_, targets_, weights_ = self.torch_conversion(preds=preds, targets=targets)

        if   self.mode == 'train':
            loss = self.loss_sign * self.loss_func(preds=preds_, targets=targets_, weights=weights_)
            return self.derivatives(loss=loss, preds=preds_)
        elif self.mode == 'eval':
            loss = self.loss_sign * self.loss_func(preds=preds_, targets=targets_, weights=weights_)
            return 'custom', loss.detach().cpu().numpy()
        else:
            raise Exception('XgboostObjective: Unknown mode (set either "train" or "eval")')

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

        # Gradient
        grad1 = torch.autograd.grad(loss, preds, create_graph=True)[0]
        
        # Diagonal elements of the Hessian matrix
        grad2 = torch.ones_like(grad1)
        if not self.skip_hessian:
            for i in tqdm(range(len(preds))): # Can be very slow
                grad2_i  = torch.autograd.grad(grad1[i], preds, retain_graph=True)[0]
                grad2[i] = grad2_i[i]

        grad1, grad2 = grad1.detach().cpu().numpy(), grad2.detach().cpu().numpy()

        if self.flatten_grad:
            grad1, grad2 = grad1.flatten("F"), grad2.flatten()
        
        return grad1, grad2
        