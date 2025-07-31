# Coupling layer flow Normalizing Flows with affine RealNVP and spline (RQS) designs
#
# Dinh et al.,   https://arxiv.org/abs/1605.08803
# Durkan et al., https://arxiv.org/abs/1906.04032
#
# m.mieskolainen@imperial.ac.uk, 2025

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from . import splines
from icenet.deep.daux import ResidualMLP

class CouplingFlow(nn.Module):
    def __init__(self,
                 x_dim: int,
                 cond_dim: int    = None,
                 flow_layers: int = 4,
                 flow_type: str   = "realnvp",
                 permute: bool    = True,
                 nn_param: list[dict] = [{"hidden_dim": [64, 64], "act": "relu", "layer_norm": True}]):
        
        super().__init__()
        self.x_dim = x_dim
        self.flow_type = flow_type
        
        print(f'CouplingFlow: using flow_type = {flow_type} with flow_layers = {flow_layers} and permute = {permute}')
        
        # Register mask buffers and create layers
        self.coupling_layers = nn.ModuleList()
        
        # Create alternating "1D checkerboard" binary masks
        for i in range(flow_layers):
            
            # Start with 0 or 1 depending on layer index
            pattern = [(j + i) % 2 for j in range(x_dim)]
            mask = torch.tensor(pattern, dtype=torch.bool)
            
            if flow_type == "realnvp":
                self.coupling_layers.append(
                    RealNVPCoupling(mask=mask, cond_dim=cond_dim, nn_param=nn_param)
                )
            elif flow_type == "spline":
                self.coupling_layers.append(
                    SplineCoupling(mask=mask, cond_dim=cond_dim, nn_param=nn_param)
                )
            else:
                raise Exception("Unknown type (choose 'realnvp' or 'spline')")
            
            # Add random permutation (except after last layer)
            if permute and i < flow_layers - 1:
                self.coupling_layers.append(RandomPermutation(x_dim))
    
    def log_base(self, z):
        return -0.5 * torch.sum(z**2, dim=1) - 0.5 * self.x_dim * math.log(2*math.pi)
        
    def loss(self, x, cond):
        """ negative log-likelihood loss """
        return -self.log_prob(x, cond)
    
    def forward(self, x, cond):
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in self.coupling_layers:
            z, log_det = layer(z, cond)
            log_det_total += log_det
        return z, log_det_total
    
    def inverse(self, z, cond):
        log_det_total = torch.zeros(z.shape[0], device=z.device)
        x = z
        for layer in reversed(self.coupling_layers):
            x, log_det = layer.inverse(x, cond)
            log_det_total += log_det
        return x, log_det_total
    
    def log_prob(self, x, cond):
        z, log_det = self.forward(x, cond)
        return self.log_base(z) + log_det
    
    def sample(self, num_samples, cond, return_ldj=False):
        device = next(self.coupling_layers[0].parameters()).device
        z      = torch.randn(num_samples, self.x_dim, device=device)
        x, ldj = self.inverse(z, cond)
        
        if return_ldj:
            return x, ldj
        else:
            return x

class RandomPermutation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        perm = torch.randperm(dim)
        inv_perm = torch.argsort(perm)
        self.register_buffer("perm", perm)
        self.register_buffer("inv_perm", inv_perm)

    def forward(self, x, cond=None):
        return x[:, self.perm], torch.zeros(x.shape[0], device=x.device)

    def inverse(self, z, cond=None):
        return z[:, self.inv_perm], torch.zeros(z.shape[0], device=z.device)

class RealNVPCoupling(nn.Module):
    """
    RealNVP type affine flow coupling type layers, with optional conditioning
    """
    def __init__(self,
                 mask: torch.Tensor,
                 cond_dim: int = None,
                 nn_param: list[dict] = [{}, {}]):

        super().__init__()
        mask = mask.to(torch.bool)
        assert mask.dim() == 1, "mask must be a 1D tensor"

        self.cond_dim = cond_dim or 0
        
        # register mask + its inverse so they move with .to(device)
        self.register_buffer('mask',     mask)
        self.register_buffer('inv_mask', ~mask)
        
        # how many coords in each partition?
        self.pass_dim  = int(mask.sum().item())
        self.trans_dim = mask.numel() - self.pass_dim
        
        # sub‐nets see (passive + cond) as input, and output one param per trans‐dim
        net_in_dim = self.pass_dim + self.cond_dim
        
        self.scale_net     = ResidualMLP(in_dim = net_in_dim, out_dim = self.trans_dim, **nn_param[0])
        self.translate_net = ResidualMLP(in_dim = net_in_dim, out_dim = self.trans_dim, **nn_param[1])
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None):
        
        # split x into the "passive" and "to-transform" parts
        x_pass  = x[:, self.mask]
        x_trans = x[:, self.inv_mask]
        
        # build net input = [x_pass, cond?]
        net_in = torch.cat([x_pass, cond], dim=1) if cond is not None else x_pass
        
        # compute scale & translation
        s = self.scale_net(net_in)
        t = self.translate_net(net_in)
        
        # affine transform
        y_trans = x_trans * torch.exp(s) + t
        
        y = x.clone()
        y[:, self.inv_mask] = y_trans
        log_det = s.sum(dim=1)
        
        return y, log_det
        
    def inverse(self, y: torch.Tensor, cond: torch.Tensor | None = None):
        
        # same partitioning
        y_pass  = y[:, self.mask]
        y_trans = y[:, self.inv_mask]
        
        net_in = torch.cat([y_pass, cond], dim=1) if cond is not None else y_pass
        
        s = self.scale_net(net_in)
        t = self.translate_net(net_in)
        
        # invert the affine
        x_trans = (y_trans - t) * torch.exp(-s)
        
        x = y.clone()
        x[:, self.inv_mask] = x_trans
        log_det = -s.sum(dim=1)
        
        return x, log_det

class SplineCoupling(nn.Module):
    """
    Neural spline flow with coupling type layers, with optional conditioning
    """
    def __init__(self,
                 mask: torch.Tensor,
                 cond_dim: int = None,
                 K: int = 5,
                 B: int = 3,
                 nn_param=[{}, {}]):
        super().__init__()
        
        # mask: 1d Boolean or 0/1 Tensor of length D
        mask = mask.to(torch.bool)
        assert mask.dim() == 1, "mask must be a vector"
        D = mask.numel()
        assert D >= 2, "Dimension must be at least 2 for coupling layers."
        
        self.register_buffer('mask', mask)
        self.register_buffer('inv_mask', ~mask)
        
        # how many dims in each part
        self.upper_dim = int(mask.sum().item())
        self.lower_dim = D - self.upper_dim
        self.K = K
        self.B = B
        self.cond_dim = cond_dim or 0
        
        # sub‑network inputs include conditioning if provided
        x_dim_lower = self.lower_dim + self.cond_dim
        x_dim_upper = self.upper_dim + self.cond_dim
        
        # networks
        self.f1 = ResidualMLP(in_dim=x_dim_lower, out_dim=(3 * K - 1) * self.upper_dim, **nn_param[0])
        self.f2 = ResidualMLP(in_dim=x_dim_upper, out_dim=(3 * K - 1) * self.lower_dim, **nn_param[1])
        
    def forward(self, x: torch.Tensor, cond: torch.Tensor = None):
        # x: (batch, D)
        log_det = x.new_zeros(x.shape[0])
        
        # pick out lower / upper by mask
        x_lower = x[:, self.inv_mask]  # passive half
        x_upper = x[:, self.mask]      # to-be-transformed
        
        # 1: lower -> transform upper
        inp1 = torch.cat([x_lower, cond], dim=1) if cond is not None else x_lower
        params1 = self.f1(inp1).view(-1, self.upper_dim, 3 * self.K - 1)
        W1, H1, D1 = torch.split(params1, self.K, dim=2)
        W1 = 2 * self.B * torch.softmax(W1, dim=2)
        H1 = 2 * self.B * torch.softmax(H1, dim=2)
        D1 = F.softplus(D1)
        x_upper, ld1 = splines.unconstrained_RQS(x_upper, W1, H1, D1, inverse=False, tail_bound=self.B)
        log_det = log_det + ld1.sum(1)
        
        # 2: transformed upper -> transform lower
        inp2 = torch.cat([x_upper, cond], dim=1) if cond is not None else x_upper
        params2 = self.f2(inp2).view(-1, self.lower_dim, 3 * self.K - 1)
        W2, H2, D2 = torch.split(params2, self.K, dim=2)
        W2 = 2 * self.B * torch.softmax(W2, dim=2)
        H2 = 2 * self.B * torch.softmax(H2, dim=2)
        D2 = F.softplus(D2)
        x_lower, ld2 = splines.unconstrained_RQS(x_lower, W2, H2, D2, inverse=False, tail_bound=self.B)
        log_det = log_det + ld2.sum(1)
        
        # scatter back into full D‑dim vector
        z = x.clone()
        z[:, self.mask]     = x_upper
        z[:, self.inv_mask] = x_lower

        return z, log_det
        
    def inverse(self, z: torch.Tensor, cond: torch.Tensor = None):
        log_det = z.new_zeros(z.shape[0])
        
        # same indexing but reverse the two steps
        z_lower = z[:, self.inv_mask]
        z_upper = z[:, self.mask]
        
        # 1. inverse: first undo f2 on lower
        inp2 = torch.cat([z_upper, cond], dim=1) if cond is not None else z_upper
        params2 = self.f2(inp2).view(-1, self.lower_dim, 3 * self.K - 1)
        W2, H2, D2 = torch.split(params2, self.K, dim=2)
        W2 = 2 * self.B * torch.softmax(W2, dim=2)
        H2 = 2 * self.B * torch.softmax(H2, dim=2)
        D2 = F.softplus(D2)
        z_lower, ld2 = splines.unconstrained_RQS(z_lower, W2, H2, D2, inverse=True, tail_bound=self.B)
        log_det = log_det + ld2.sum(1)
        
        # 2. then reverse f1 on upper
        inp1 = torch.cat([z_lower, cond], dim=1) if cond is not None else z_lower
        params1 = self.f1(inp1).view(-1, self.upper_dim, 3 * self.K - 1)
        W1, H1, D1 = torch.split(params1, self.K, dim=2)
        W1 = 2 * self.B * torch.softmax(W1, dim=2)
        H1 = 2 * self.B * torch.softmax(H1, dim=2)
        D1 = F.softplus(D1)
        z_upper, ld1 = splines.unconstrained_RQS(z_upper, W1, H1, D1, inverse=True, tail_bound=self.B)
        log_det = log_det + ld1.sum(1)
        
        # scatter back into full D‑dim vector
        x = z.clone()
        x[:, self.mask]     = z_upper
        x[:, self.inv_mask] = z_lower
        
        return x, log_det
