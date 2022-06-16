# Variational autoencoder
# 
# https://arxiv.org/abs/1312.6114
#
# m.mieskolainen@imperial.ac.uk, 2022

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from icenet.deep.dmlp import MLP

"""
        |                                                    |
        |                                                    |
input x | decoder | latent space (z) ~ N(mu,sigma) | encoder | output xhat
        |                                                    |
        |                                                    |

Training loop ELBO minimization:
    xhat = model.forward(x)
    loss = ((x - x_hat)**2).sum() + model.encoder.kl.sum()

"""

class Encoder(nn.Module):
    def __init__(self, D, hidden_dim=128, latent_dim=32):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(D, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)

        # modules  = [self.linear2, torch.sigmoid()]
        #self.linear2 = nn.Sequential(*modules)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)

class VariationalEncoder(nn.Module):
    def __init__(self, D, hidden_dim, latent_dim):
        super(VariationalEncoder, self).__init__()
        
        self.mlp        = MLP([D, hidden_dim, hidden_dim], activation='relu')

        self.mlp_mu     = MLP([hidden_dim, hidden_dim, latent_dim], activation='tanh')
        self.mlp_logvar = MLP([hidden_dim, hidden_dim, latent_dim], activation='tanh')

        self.N          = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):

        q       = self.mlp(x)

        # "Re-parametrization" trick
        mu      = self.mlp_mu(q)
        logvar  = self.mlp_logvar(q)
        std     = torch.exp(0.5 * logvar)

        z       = mu + std*self.N.sample(mu.shape)

        # KL-divergence between unit Gaussian (Wikipedia)
        self.kl = 0.5 * torch.sum(std.pow(2) + mu.pow(2) - 1 - logvar, dim=-1)

        return z

    def to_device(self, device):
        """ Needed for cuda
        """
        self.N.loc   = self.N.loc.to(device) # Get sampling on GPU "hack"
        self.N.scale = self.N.scale.to(device)
        return self


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, D):
        super(Decoder, self).__init__()

        self.mlp = MLP([latent_dim, hidden_dim, hidden_dim, D], activation='relu')

    def forward(self, z):
        return self.mlp(z)


class VAE(nn.Module):
    def __init__(self, D, latent_dim, hidden_dim, C=None):
        super(VAE, self).__init__()

        self.D       = D
        self.C       = C

        self.encoder = VariationalEncoder(D=D, hidden_dim=hidden_dim, latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, D=D)

    def to_device(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder.to_device(device)
        return self

    def forward(self, x):
        z    = self.encoder(x)
        xhat = self.decoder(z)
        return xhat

    def softpredict(self, x):
        xhat  = self.forward(x)
        mse   = torch.sum((xhat - x)**2 / x.shape[-1], dim=-1)
        
        # "test statistic"
        score = mse

        if self.training:
            return torch.log(torch.tanh(score)) # Numerically more stable
        else:
            return torch.tanh(score)
