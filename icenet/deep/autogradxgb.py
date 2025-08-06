# Custom pytorch-driven autograd losses for XGBoost
# with various Hessian diagonal approaches.
#
# m.mieskolainen@imperial.ac.uk, 2025

import numpy as np
import torch
from torch import Tensor
import xgboost
import time
from tqdm import tqdm

from typing import Callable, Sequence, List, Tuple

# ------------------------------------------
from icenet import print
# ------------------------------------------

class XgboostObjective():
    """
    XGB custom loss driver class with torch (autograd)
    
    hessian_mode: 'hutchinson' (or 'iterative') may make the model converge
    significantly faster (or better) than 'constant' in some cases.
    
    N.B. Remember to call manually:
    
    obj.mode = 'train' or obj.mode = 'eval' while running the training boost iteration
    loop, otherwise .grad_prev, .preds_prev will get mixed with iterative hessian mode.
    
    Args:
        loss_func:       Loss function handle
        mode:            'train' or 'eval', see the comment above
        flatten_grad:    For vector valued model output [experimental]
        hessian_mode:    'constant', 'iterative', 'hutchinson', 'exact'
        hessian_const:   Scalar parameter constant 'hessian_mode'
        
        hessian_gamma:   Hessian EMA smoothing parameter for the 'iterative' mode
        hessian_eps:     Hessian estimate denominator regularization for the 'iterative'
        hessian_absmax:  Hessian absolute clip parameter for the 'iterative'
        
        hessian_slices:  Hutchinson MC estimator MC slice sample size for the 'hutchinson' mode
        device:          Torch device
    """
    
    def __init__(self,
            loss_func: Callable[[Tensor, Tensor], Tensor],
            mode: str='train',
            flatten_grad:   bool=False,
            hessian_mode:   str='hutchinson',
            hessian_const:  float=1.0,
            hessian_gamma:  float=0.9,
            hessian_eps:    float=1e-8,
            hessian_limit:  list=[1e-2, 20],
            hessian_slices: int=10,
            device: torch.device='cpu'
        ):
        
        self.mode           = mode
        self.loss_func      = loss_func
        self.device         = device
        self.hessian_mode   = hessian_mode
        self.hessian_const  = hessian_const
        
        # Iterative mode
        self.hessian_gamma  = hessian_gamma
        self.hessian_eps    = hessian_eps
        self.hessian_limit  = hessian_limit
        
        # Hutchinson mode
        self.hessian_slices = int(hessian_slices)
        
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

    def regulate_hess(self, hess: Tensor):
        """
        Regulate to be positive definite (H_ii > hessian_min)
        as required by second order gradient descent.
        
        Do not clip to zero, as that might result in zero denominators
        in the Hessian routines inside xgboost.
        """
        hess = torch.abs(hess) # ~ negative weights
        hess = torch.clamp(hess, min=self.hessian_limit[0], max=self.hessian_limit[1])
        
        return hess
        
    @torch.no_grad
    def iterative_hessian_update(self, grad: Tensor, preds: Tensor):
        """
        Iterative approximation of the Hessian diagonal using finite differences
        based on a previous boost iteration Hessian (full batch training ~ only one Hessian stored)
        
        Args:
            grad:  Current gradient vector
            preds: Current prediction vector
        """
        print(f'Computing Hessian diag with iterative finite difference (gamma = {self.hessian_gamma})')
        
        # Initialize to unit curvature as a neutral default
        # (if sigma_i^2 = 1/H_ii, then this is a Gaussian N(0,1) prior)
        if self.hess_diag is None:
            self.hess_diag = torch.ones_like(grad)
            hess_diag_new  = torch.ones_like(grad)
        
        # H_ii ~ difference in gradients / difference in predictions
        else:
            dg = grad  - self.grad_prev
            ds = preds - self.preds_prev
            
            hess_diag_new = dg / (ds + self.hessian_eps)
            hess_diag_new = self.regulate_hess(hess_diag_new) # regulate
        
        # Exponential Moving Average (EMA), approx filter size ~ 1 / (1 - gamma) steps
        self.hess_diag = self.hessian_gamma * self.hess_diag + \
                         (1 - self.hessian_gamma) * hess_diag_new
        
        # Save the gradient vector and predictions
        self.grad_prev  = grad.clone().detach()
        self.preds_prev = preds.clone().detach()
        
    def hessian_hutchinson(self, grad: Tensor, preds: Tensor):
        """
        Hutchinson MC estimator for the Hessian diagonal ~ O(slices) (time)
        """
        tic = time.time()
        print(f'Computing Hessian diag with Hutchinson MC (slices = {self.hessian_slices}) ... ')

        hess = torch.zeros_like(preds)
        
        for _ in range(self.hessian_slices):
            
            # Generate a Rademacher vector (each element +-1 with probability 0.5)
            v = torch.empty_like(preds).uniform_(-1, 1)
            v = torch.sign(v)

            # Compute Hessian-vector product H * v
            Hv = torch.autograd.grad(grad, preds, grad_outputs=v, retain_graph=True)[0]

            # Accumulate element-wise product v * Hv to get the diagonal
            hess += v * Hv

        # Average over all samples
        hess = hess / self.hessian_slices
        hess = self.regulate_hess(hess) # regulate
        
        print(f'Took {time.time()-tic:.2f} sec')

        return hess
        
    def hessian_exact(self, grad: Tensor, preds: Tensor):
        """
        Hessian diagonal with exact autograd ~ O(data points) (time)
        """
        tic = time.time()
        print('Computing Hessian diagonal with exact autograd ... ')

        hess = torch.zeros_like(preds)
        
        for i in tqdm(range(len(preds))):
            
            # A basis vector
            e_i = torch.zeros_like(preds)
            e_i[i] = 1.0
            
            # Compute the Hessian-vector product H e_i
            hess[i] = torch.autograd.grad(grad, preds, grad_outputs=e_i, retain_graph=True)[0][i]

        hess = self.regulate_hess(hess) # regulate
        
        print(f'Took {time.time()-tic:.2f} sec')
        
        return hess
        
    def derivatives(self, loss: Tensor, preds: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gradient and Hessian diagonal
        
        Args:
            loss:  loss function values
            preds: model predictions
        
        Returns:
            gradient vector, hessian diagonal vector as numpy arrays
        """
        
        ## Gradient
        grad1 = torch.autograd.grad(loss, preds, create_graph=True)[0]
        
        ## Diagonal elements of the Hessian matrix
        match self.hessian_mode:
            
            # Constant curvature
            case 'constant':
                print(f'Setting Hessian diagonal using a constant (hessian_const = {self.hessian_const})')
                grad2 = self.hessian_const * torch.ones_like(grad1)

            # BFGS style iterative updates
            case 'iterative':
                self.iterative_hessian_update(grad=grad1, preds=preds)
                grad2 = self.hess_diag

            # Hutchinson based MC estimator
            case 'hutchinson':
                grad2 = self.hessian_hutchinson(grad=grad1, preds=preds)
            
            # Exact autograd (slow)
            case 'exact':
                grad2 = self.hessian_exact(grad=grad1, preds=preds)

            # Squared derivative based [uncontrolled] approximation (always positive curvature)
            case 'squared_approx':
                print(f'Setting Hessian diagonal using grad^2 [DEBUG ONLY]')
                grad2 = grad1 * grad1
            
            case _:
                raise Exception(f'Unknown "hessian_mode" {self.hessian_mode}')
        
        # Return numpy arrays
        grad1, grad2 = grad1.detach().cpu().numpy(), grad2.detach().cpu().numpy()
        
        if self.flatten_grad:
            grad1, grad2 = grad1.flatten("F"), grad2.flatten("F")
        
        return grad1, grad2
