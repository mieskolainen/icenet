# Deep domain adaptation functions
#
# m.mieskolainen@imperial.ac.uk, 2024

import torch
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Unsupervised Domain Adaptation by Backpropagation
    https://arxiv.org/abs/1409.7495
    
    Notes: The forward pass is an identity map. In the backprogation,
    the gradients are reversed by grad -> -alpha * grad.
    
    Example:
        net = nn.Sequential(nn.Linear(10, 10), GradientReversal(alpha=1.0))
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        alpha = ctx.alpha
        alpha = grads.new_tensor(alpha)
        dx = -alpha * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, alpha = 1.0):
        super(GradientReversal, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)
