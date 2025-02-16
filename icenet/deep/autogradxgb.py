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
    """
    Args:
        loss_func:       Loss function handle
        mode:            'train' or 'eval'
        flatten_grad:    For vector valued model output [experimental]
        hessian_mode:    'constant', 'squared_approx', 'iterative', 'hutchinson', 'exact'
        hessian_const:   Scalar parameter 'constant 'hessian_mode'
        hessian_gamma:   Hessian momentum smoothing parameter for 'iterative' mode
        hessian_slices:  Hutchinson Hessian diagonal estimator MC slice sample size
        device:          Torch device
    """
    def __init__(self,
            loss_func: Callable[[Tensor, Tensor], Tensor],
            mode: str='train',
            flatten_grad: bool=False,
            hessian_mode: str='constant',
            hessian_const: float=1.0,
            hessian_gamma: float=0.9,
            hessian_slices: int=10,
            device: torch.device='cpu'
        ):
        
        self.mode           = mode
        self.loss_func      = loss_func
        self.device         = device
        self.hessian_mode   = hessian_mode
        self.hessian_const  = hessian_const
        self.hessian_gamma  = hessian_gamma
        self.hessian_slices = hessian_slices
        self.flatten_grad   = flatten_grad
        
        # For the optimization algorithms
        self.hess_diag  = None
        self.grad_prev  = None
        self.preds_prev = None
        
        txt = f'Using device: {self.device} | hessian_mode = {self.hessian_mode}'
        
        match self.hessian_mode:
            case 'constant':
                print(f'{txt} | hessian_const = {self.hessian_const}')
            case 'iterative':
                print(f'{txt} | hessian_gamma = {self.hessian_gamma}')
            case 'hutchinson':
                print(f'{txt} | hessian_slices = {self.hessian_slices}')
            case _:
                print(f'{txt}')

    def __call__(self, preds: np.ndarray, targets: xgboost.DMatrix):

        preds_, targets_, weights_ = self.torch_conversion(preds=preds, targets=targets)

        match self.mode:
            case 'train':
                loss = self.loss_func(preds=preds_, targets=targets_, weights=weights_)
                return self.derivatives(loss=loss, preds=preds_)
            case 'eval':
                loss = self.loss_func(preds=preds_, targets=targets_, weights=weights_)
                return 'custom', loss.detach().cpu().numpy()
            case _:
                raise Exception('Unknown mode (set either "train" or "eval")')

    def torch_conversion(self, preds: np.ndarray, targets: xgboost.DMatrix):
        """
        Conversion from xgboost.Dmatrix object
        """
        try:
            weights = targets.get_weight()
            weights = None if weights == [] else torch.FloatTensor(weights).to(self.device)
        except:
            weights = None

        preds   = torch.FloatTensor(preds).requires_grad_().to(self.device)
        targets = torch.FloatTensor(targets.get_label()).to(self.device)

        return preds, targets, weights

    def iterative_hessian_update(self,
            grad: Tensor, preds: Tensor, absMax: float=10, EPS: float=1e-8):
        """
        Iterative Hessian (diagonal) approximation update using finite differences
        
        [experimental]
        
        Args:
            grad:  Current gradient vector
            preds: Current prediction vector
        """
        
        if self.hess_diag is None:
            self.hess_diag = torch.ones_like(grad)
            hess_diag_new  = torch.ones_like(grad)
        
        # H_ii ~ difference in gradients / difference in predictions
        else:
            dg = grad  - self.grad_prev
            ds = preds - self.preds_prev
            
            hess_diag_new = dg / (ds + EPS)
            hess_diag_new = torch.clamp(hess_diag_new, min=-absMax, max=absMax)
        
        # Running smoothing update to stabilize
        self.hess_diag = self.hessian_gamma * self.hess_diag + (1-self.hessian_gamma) * hess_diag_new
        
        # Save the gradient vector and predictions
        self.grad_prev  = grad.clone()
        self.preds_prev = preds.clone()
    
    def derivatives(self, loss: Tensor, preds: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Gradient and Hessian diagonal
        
        Args:
            loss:  loss function values
            preds: model predictions
        
        Returns:
            gradient vector, hessian diagonal vector
        """
        
        ## Gradient
        grad1 = torch.autograd.grad(loss, preds, create_graph=True)[0]
        
        ## Diagonal elements of the Hessian matrix
        
        match self.hessian_mode:
            
            # Constant curvature
            case 'constant':
                grad2 = self.hessian_const * torch.ones_like(grad1)
        
            # Squared derivative based [uncontrolled] approximation (always positive curvature)
            case 'squared_approx':
                grad2 = grad1 * grad1
        
            # BFGS style iterative updates
            case 'iterative':
                self.iterative_hessian_update(grad=grad1, preds=preds)
                grad2 = self.hess_diag
            
            # Hutchinson MC approximator ~ O(slices)
            case 'hutchinson':
                
                print('Computing Hessian diagonal with approximate Hutchinson estimator ...')

                grad2 = torch.zeros_like(preds)

                for _ in tqdm(range(self.hessian_slices)):
                    
                    # Generate a Rademacher vector (each element +-1 with probability 0.5)
                    v = torch.empty_like(preds).uniform_(-1, 1)
                    v = torch.sign(v)

                    # Compute Hessian-vector product H * v
                    Hv = torch.autograd.grad(grad1, preds, grad_outputs=v, retain_graph=True)[0]

                    # Accumulate element-wise product v * Hv to get the diagonal
                    grad2 += v * Hv

                # Average over all samples
                grad2 /= self.hessian_slices
                
            # Exact autograd (slow)
            case 'exact':
                
                print('Computing Hessian diagonal with exact autograd ...')
                
                grad2 = torch.zeros_like(preds)
                
                for i in tqdm(range(len(preds))):
                    
                    # A basis vector
                    e_i = torch.zeros_like(preds)
                    e_i[i] = 1.0
                    
                    # Compute the Hessian-vector product H e_i
                    grad2[i] = torch.autograd.grad(grad1, preds, grad_outputs=e_i, retain_graph=True)[0][i]

            case _:
                raise Exception(f'Unknown "hessian_mode" {self.hessian_mode}')
        
        grad1, grad2 = grad1.detach().cpu().numpy(), grad2.detach().cpu().numpy()
        
        if self.flatten_grad:
            grad1, grad2 = grad1.flatten("F"), grad2.flatten("F")
        
        return grad1, grad2
