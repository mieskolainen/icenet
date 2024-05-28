# Based on <github.com/ZiyaoLi/fast-kan/blob/master/fastkan>
# 
# m.mieskolainen@imperial.ac.uk, 2024

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *

from icenet.deep.deeptools import Multiply

class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw) -> None:
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)

class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.,
        grid_max: float =  2.,
        num_grids: int  =   8,
        denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class FastKANLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, grid_min: float = -2., grid_max: float = 2.,
        num_grids: int = 8, use_base_update: bool = False, base_activation = F.silu, spline_weight_init_scale: float = 0.1):
        
        super().__init__()
        
        self.layernorm = nn.LayerNorm(input_dim)
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(input_dim * num_grids, output_dim, spline_weight_init_scale)
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear     = nn.Linear(input_dim, output_dim)

    def forward(self, x, time_benchmark=False):
        if not time_benchmark:
            spline_basis = self.rbf(self.layernorm(x))
        else:
            spline_basis = self.rbf(x)
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            base = self.base_linear(self.base_activation(x))
            ret = ret + base
        return ret


class FastKAN(nn.Module):
    def __init__(self, D, C, mlp_dim: List[int], grid_min: float = -2.0, grid_max: float =  2.0, num_grids: int = 8,
                 use_base_update: bool = False, base_activation = F.silu, spline_weight_init_scale: float = 0.1,
                 out_dim=None, last_tanh=False, last_tanh_scale=10.0, **kwargs):
        
        super().__init__()
        
        self.D = D
        self.C = C
        
        if out_dim is None:
            self.out_dim = C
        else:
            self.out_dim = out_dim
        
        layers_hidden = [D] + mlp_dim + [self.out_dim]
        
        self.layers = nn.ModuleList([
            FastKANLayer(
                input_dim       = in_dim,
                output_dim      = out_dim,
                grid_min        = grid_min,
                grid_max        = grid_max,
                num_grids       = num_grids,
                use_base_update = use_base_update,
                base_activation = base_activation,
                spline_weight_init_scale = spline_weight_init_scale,
            ) for in_dim, out_dim in zip(layers_hidden[:-1], layers_hidden[1:])
        ])

        # Add extra final squeezing activation and post-scale aka "soft clipping"
        if last_tanh:
            self.layers.add_module("tanh",  nn.Tanh())
            self.layers.add_module("scale", Multiply(last_tanh_scale))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def softpredict(self,x) :
        """ Softmax probability
        """
        
        if self.out_dim > 1:
            return F.softmax(self.forward(x), dim=-1)
        else:
            return torch.sigmoid(self.forward(x))


class AttentionWithFastKANTransform(nn.Module):
    
    def __init__(
        self,
        q_dim: int,
        k_dim: int,
        v_dim: int,
        head_dim: int,
        num_heads: int,
        gating: bool = True,
    ):
        super(AttentionWithFastKANTransform, self).__init__()

        self.num_heads = num_heads
        total_dim = head_dim * self.num_heads
        self.gating = gating
        self.linear_q = FastKANLayer(q_dim, total_dim)
        self.linear_k = FastKANLayer(k_dim, total_dim)
        self.linear_v = FastKANLayer(v_dim, total_dim)
        self.linear_o = FastKANLayer(total_dim, q_dim)
        self.linear_g = None
        if self.gating:
            self.linear_g = FastKANLayer(q_dim, total_dim)
        # precompute the 1/sqrt(head_dim)
        self.norm = head_dim**-0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor = None,      # additive attention bias
    ) -> torch.Tensor:         

        wq = self.linear_q(q).view(*q.shape[:-1], 1, self.num_heads, -1) * self.norm     # *q1hc
        wk = self.linear_k(k).view(*k.shape[:-2], 1, k.shape[-2], self.num_heads, -1)    # *1khc
        att = (wq * wk).sum(-1).softmax(-2)     # *qkh
        del wq, wk
        if bias is not None:
            att = att + bias[..., None]

        wv = self.linear_v(v).view(*v.shape[:-2],1, v.shape[-2], self.num_heads, -1)     # *1khc
        o = (att[..., None] * wv).sum(-3)        # *qhc
        del att, wv

        o = o.view(*o.shape[:-2], -1)           # *q(hc)

        if self.linear_g is not None:
            # gating, use raw query input
            g = self.linear_g(q)
            o = torch.sigmoid(g) * o

        # merge heads
        o = self.linear_o(o)
        return o
