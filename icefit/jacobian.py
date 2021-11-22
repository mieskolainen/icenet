# Torch autograd Jacobians, Hessians and related
# 
# m.mieskolainen@imperial.ac.uk, 2021

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd.functional import jacobian


def get_gradient(inputs, func, **kwargs):
    """
    Return derivative df(x)/dx or gradient vectors
    for f: R^n -> R
    """
    x = Variable(inputs.data, requires_grad=True) # We start to track gradients from here
    z = func(x, **kwargs)

    if len(z.size()) > 1: # Higher dimensional output
        return get_jacobian(inputs=inputs, func=func, **kwargs)

    z.backward([torch.ones_like(z)], create_graph=True)
    return x.grad


def get_jacobian(inputs, func, **kwargs):
    """
    Return Jacobian matrix of size m x n
    of a function f: R^n -> R^m
    """
    size = inputs.size()
    n    = size[0]

    # Input dim
    if (len(size)) > 1:
        dim_input = size[1]
    else:
        dim_input = 1

    test = func(inputs, **kwargs)

    # Output dim
    if len(test.size()) > 1:
        dim_output = test.size()[1]
    else:
        dim_output = 1

    J = torch.zeros((n, dim_output, dim_input))

    # Loop over all batch samples
    for i in range(n):
        x = inputs[i,...].squeeze()

        x = x.repeat(dim_output, 1)
        x.requires_grad_(True)
        z = func(x, **kwargs)

        if dim_output == 1:
            z.backward([torch.ones_like(z)])
        else:
            z.backward(torch.eye(dim_output))

        J[i,:,:] = x.grad.data

    return J


def get_full_hessian(loss, param):
    """
    Compute the full Hessian matrix of a neural net (can be very slow)
    
    https://www.cs.toronto.edu/~rgrosse/courses/csc2541_2021/readings/L02_Taylor_approximations.pdf
    
    Args:
        loss: neural net loss function
        param: neural net parameters
    """

    N = param.numel()
    H = torch.zeros(N,N)
    
    loss_grad = torch.autograd.grad(loss, param, create_graph=True, retain_graph=True, only_inputs=True)[0].view(-1)
    for i in tqmd(range(N)):
        grad2 = torch.autograd.grad(loss_grad[i], param, create_graph=False, retain_graph=True, only_inputs=True)
        H[i] = grad2[0].view(-1)
    return H


"""
def get_hessian(inputs, func, **kwargs):
    
    The Hessian matrix of a function of is the Jacobian matrix of the gradient of the function f.
    
    H(f(x)) = J(grad(f(x))
    
    Note that sometimes Hessian is approximated using a product of Jacobians as JJ^T

    TBD: Do we assume derivatives commute here (c.f. Grasmann variables) ?
""" 


def observed_fisher_info(logL, inputs, **kwargs):
    """
    Observed Fisher information -- the negative of the second derivative
    (Hessian matrix) of the log-likelihood function.
    
    The observed information is typically evaluated at the maximum-likelihood estimate of theta.
    
    'Fisher information represents the curvature of the relative entropy'

    Args:
        logL:   The log-likelihood function l(theta|X1,...X_n) = sum_{i=1}^n log f(X_i|theta),
                with X_i being the independent random observations
        inputs: log-likelihood function input

    References:
        https://en.wikipedia.org/wiki/Observed_information
        https://en.wikipedia.org/wiki/Optimal_design

        Efron, B.; Hinkley, D.V. (1978). Assessing the accuracy of the maximum likelihood estimator:
        Observed versus expected Fisher Information". Biometrika. 65 (3): 457–487.
    """
    I = - torch.autograd.functional.hessian(func=logL, inputs=inputs).squeeze()

    return I


def test_hessians(EPS=1e-4):
    """
    Test Hessian matrix computations
    """

    def func_f1(x):
        # R^3 -> R
        #
        # Input dim: n_batches x 3
        #
        return x[:,0]**2 + x[:,1]**2 + x[:,2]**2

    def H_1(x):
        # Hessian for func_f1
        return torch.tensor([[2,  0,  0],
                             [0,  2,  0],
                             [0,  0,  2]])

    def func_f2(x):
        # R^3 -> R
        #
        # Input dim: n_batches x 3
        return torch.sin(x[:,0] * x[:,1] * x[:,2])

    def H_2(X):
        # Hessian for func_f2
        x = X[:,0]
        y = X[:,1]
        z = X[:,2]

        sin = torch.sin
        cos = torch.cos

        return torch.tensor([[-y**2*z**2*sin(x*y*z), -x*y*z**2*sin(x*y*z) + z*cos(x*y*z), -x*y**2*z*sin(x*y*z) + y*cos(x*y*z)],
                          [-x*y*z**2*sin(x*y*z) + z*cos(x*y*z), -x**2*z**2*sin(x*y*z), -x**2*y*z*sin(x*y*z) + x*cos(x*y*z)],
                          [-x*y**2*z*sin(x*y*z) + y*cos(x*y*z), -x**2*y*z*sin(x*y*z) + x*cos(x*y*z), -x**2*y**2*sin(x*y*z)]])

    for i in range(1000):

        ### TEST CASE 1
        x          = Variable(torch.randn(size=(1,3)))
        H_auto     = torch.autograd.functional.hessian(func=func_f1, inputs=x).squeeze()
        H_analytic = H_1(x)

        assert torch.norm(H_auto - H_analytic) < EPS


        ### TEST CASE 2
        x          = Variable(torch.randn(size=(1,3)))
        H_auto     = torch.autograd.functional.hessian(func=func_f2, inputs=x).squeeze()
        H_analytic = H_2(x)

        assert torch.norm(H_auto - H_analytic) < EPS


def test_dimension_interfaces():
    # Test different dimensional input-output.
    
    import pytest

    def fx_R_R(x, param=None):
        # Function f: R -> R
        
        return x**2;

    def fx_R3_R(x, param=None):
        # Function f: R^3 -> R
        
        return (x[:,0] + x[:,1]**2 + x[:,2]**4)

    def fx_R_R3(x, param=None):
        # Function f: R -> R^3
        
        y = torch.hstack((x, x**4, x**2))
        return y

    def fx_R3_R3(x, param=None):
        # Function f: R^3 -> R^3
        
        y = torch.vstack((x[:,0], x[:,1]**4, x[:,2])).transpose(0,1);
        return y

    def fx_R3_R4(x, param=None):
        # Function f: R^3 -> R^4
        
        y = torch.vstack((x[:,0], x[:,1]**4, x[:,2]-x[:,0], x[:,2]*x[:,0])).transpose(0,1);
        return y

    def fx_R4_R3(x, param=None):
        # Function f: R^4 -> R^3
        
        y = torch.vstack((x[:,1]**4, x[:,2]-x[:,0], x[:,2]*x[:,0])).transpose(0,1);
        return y

    # Number of samples
    n = 2

    # Dimensions
    dim_domain = [1,3,4]
    dim_range  = [1,3,4]

    for d_in in dim_domain:
        for d_out in dim_range:

            # Test input vector with arbitrary 1,2,3,4... elements
            x = Variable(torch.ones(n, d_in).cumsum(0))

            if   d_in == 1 and d_out == 1:
                func = fx_R_R;
            elif d_in == 1 and d_out == 3:
                func = fx_R_R3;
            elif d_in == 3 and d_out == 1:
                func = fx_R3_R;
            elif d_in == 3 and d_out == 3:
                func = fx_R3_R3;
            elif d_in == 3 and d_out == 4:
                func = fx_R3_R4;
            elif d_in == 4 and d_out == 3:
                func = fx_R4_R3;
            else:
                continue # Did not found test case

            print(f'----------------------------------------------------------')
            print(f'f: R^{d_in} -> R^{d_out}')
            print(f'x        = {x}')

            print(f'f(x)     = {func(x)}')
            print(f'df(x)/dx = {get_gradient(func=func, inputs=x)} (get_gradient)')
            print(f'df(x)/dx = {get_jacobian(func=func, inputs=x)} (get_jacobian)')
            print(f'df(x)/dx = {jacobian(func=func, inputs=x)} (jacobian)')
            print('\n')


def test_jacobians(EPS=1e-4):
    
    #Test Jacobians from:
    #https://en.wikipedia.org/wiki/Jacobian_matrix_and_determinant
    
    import pytest

    def func_f1(x):
        # R^2 -> R^2
        #
        # n_batches x 2
        #
        return torch.vstack((x[:,0]**2 * x[:,1], 5*x[:,0] + torch.sin(x[:,1]))).transpose(0,1);

    def J_f1(x):
        # This will accept only one sample (batch size = 1)
        J = torch.tensor([[2*x[0]*x[1], x[0]**2], [5, torch.cos(x[1])]])
        return J

    def func_f2(x):
        # Polar-Cartesian
        # R_+ x [0, 2\pi) -> R^2
        # 
        # n_batches x 2
        #
        return torch.vstack((x[:,0] * torch.cos(x[:,1]), x[:,0]*torch.sin(x[:,1]))).transpose(0,1);

    def J_f2(x):
        # This will accept only one sample (batch size = 1)
        J = torch.tensor([[torch.cos(x[1]), -x[0]*torch.sin(x[1])], [torch.sin(x[1]), x[0]*torch.cos(x[1])]])
        return J

    def func_f3(x):
        # Polar-Cartesian
        # R_+ x [0, 2\pi) x [0, 2\pi) -> R^3
        # 
        # n_batches x 2
        #
        return torch.vstack((x[:,0]*torch.sin(x[:,1])*torch.cos(x[:,2]), 
                             x[:,0]*torch.sin(x[:,1])*torch.sin(x[:,2]),
                             x[:,0]*torch.cos(x[:,1]))).transpose(0,1);

    def J_f3(x):
        # This will accept only one sample (batch size = 1)
        J = torch.tensor([[torch.sin(x[1])*torch.cos(x[2]), x[0]*torch.cos(x[1])*torch.cos(x[2]), -x[0]*torch.sin(x[1])*torch.sin(x[2])], 
                          [torch.sin(x[1])*torch.sin(x[2]), x[0]*torch.cos(x[1])*torch.sin(x[2]),  x[0]*torch.sin(x[1])*torch.cos(x[2])],
                          [torch.cos(x[1]), -x[0]*torch.sin(x[1]),  0]])
        return J
    
    def func_f4(x):
        # R^3 -> R^4
        # 
        # n_batches x 2
        #
        return torch.vstack((x[:,0], 5*x[:,2], 4*x[:,1]**2 - 2*x[:,2], x[:,2]*torch.sin(x[:,0]))).transpose(0,1);

    def J_f4(x):
        # This will accept only one sample (batch size = 1)
        J = torch.tensor([[ 1, 0, 0 ], 
                          [ 0, 0, 5 ],
                          [ 0, 8*x[1], -2],
                          [x[2]*torch.cos(x[0]), 0, torch.sin(x[0])]])
        return J


    # Test with random input
    for k in range(100):
        
        ### TEST CASE 1
        x = Variable(torch.randn(size=(1,2)))

        J_auto     = get_jacobian(func=func_f1, inputs=x).squeeze()
        J_analytic = J_f1(x[0,...])

        assert torch.norm(J_auto - J_analytic) < EPS


        ### TEST CASE 2
        x = Variable(torch.rand(size=(1,2)))
        x[0,0]    *= 1000
        x[0,1]    *= 2*np.pi

        J_auto     = get_jacobian(func=func_f2, inputs=x).squeeze()
        J_analytic = J_f2(x[0,...])

        assert torch.norm(J_auto - J_analytic) < EPS


        ### TEST CASE 3
        x = Variable(torch.rand(size=(1,3)))
        x[0,0]    *= 1000
        x[0,1]    *= 2*np.pi
        x[0,2]    *= 2*np.pi

        J_auto     = get_jacobian(func=func_f3, inputs=x).squeeze()
        J_analytic = J_f3(x[0,...])
        
        assert torch.norm(J_auto - J_analytic) < EPS


        ### TEST CASE 4
        x = Variable(torch.rand(size=(1,3)))

        J_auto     = get_jacobian(func=func_f4, inputs=x).squeeze()
        J_analytic = J_f4(x[0,...])
        
        print(J_auto)
        print(J_analytic)

        assert torch.norm(J_auto - J_analytic) < EPS
        