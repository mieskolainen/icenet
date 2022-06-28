# Variational autoencoder
# 
# https://arxiv.org/abs/1312.6114
# https://lilianweng.github.io/posts/2018-08-12-vae/
#
# m.mieskolainen@imperial.ac.uk, 2022

import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from icenet.deep.dmlp import MLP, MLP_ALL_ACT

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
    def __init__(self, D, hidden_dim=128, latent_dim=32, activation='tanh', batch_norm=True):
        super(Encoder, self).__init__()

        self.mlp = MLP_ALL_ACT([D, hidden_dim, latent_dim], activation=activation, batch_norm=batch_norm)
        
        # modules  = [self.linear2, torch.sigmoid()]
        #self.linear2 = nn.Sequential(*modules)

    def forward(self, x):
        return self.mlp(x)

class VariationalEncoder(nn.Module):
    def __init__(self, D, hidden_dim, latent_dim, activation='relu', batch_norm=True):
        super(VariationalEncoder, self).__init__()
        
        self.mlp        = MLP_ALL_ACT([D, hidden_dim, hidden_dim], activation=activation, batch_norm=batch_norm)
        self.mlp_mu     = MLP_ALL_ACT([hidden_dim, latent_dim],    activation='tanh', batch_norm=batch_norm)
        
        # This should output positive definite value (use relu)
        self.mlp_logvar = MLP_ALL_ACT([hidden_dim, latent_dim],    activation='relu', batch_norm=batch_norm)
        
        self.N          = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        q        = self.mlp(x)

        # "Re-parametrization" trick to allow gradients backpropagate
        mu       = self.mlp_mu(q)
        logvar   = self.mlp_logvar(q)
        std      = torch.exp(0.5 * logvar)

        z        = mu + std*self.N.sample(mu.shape)

        # Analytic KL-divergence against unit Gaussian (Wikipedia)
        # Take torch.sum(self.kl_i, dim=-1) to obtain the MC estimate
        self.kl_i = 0.5 * (std.pow(2) + mu.pow(2) - 1 - logvar)

        return z, mu, std

    def to_device(self, device):
        """ Needed for cuda
        """
        self.N.loc   = self.N.loc.to(device) # Get sampling on GPU "hack"
        self.N.scale = self.N.scale.to(device)
        return self


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, D, activation='tanh', batch_norm=True):
        super(Decoder, self).__init__()

        self.mlp = MLP([latent_dim, hidden_dim, hidden_dim, D], activation=activation, batch_norm=batch_norm)

    def forward(self, z):
        return self.mlp(z)


class VAE(nn.Module):
    def __init__(self, D, latent_dim, hidden_dim,
            encoder_bn=True, encoder_act='relu', decoder_bn=False, decoder_act='relu', anomaly_score='MSE', C=None):
        super(VAE, self).__init__()

        self.D       = D
        self.C       = C
        self.anomaly_score = anomaly_score

        self.encoder = VariationalEncoder(D=D, hidden_dim=hidden_dim, latent_dim=latent_dim, activation=encoder_act, batch_norm=encoder_bn)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, activation=decoder_act, batch_norm=decoder_bn, D=D)

    def to_device(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder.to_device(device)
        return self

    def forward(self, x):
        z,mu,std = self.encoder(x)
        xhat     = self.decoder(z)
        return xhat

    def softpredict(self, x):
        z,mu,std = self.encoder(x)
        xhat     = self.decoder(z)
        
        # "Test statistic"
        if   self.anomaly_score == 'MSE':
            score = torch.sum((xhat - x)**2, dim=-1)/x.shape[-1]
            return torch.tanh(score)

        elif self.anomaly_score == 'MSE_KL':
            score = torch.sum((xhat - x)**2, dim=-1)/x.shape[-1] + torch.sum(self.encoder.kl_i, dim=-1)/z.shape[-1]
            return 1 - torch.tanh(1/score)
        
        else:
            raise Exception(__name__ + f'.softpredict: Unknown <anomaly_score> = {self.anomaly_score} selected.')