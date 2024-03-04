# Variational autoencoder
# 
# https://arxiv.org/abs/1312.6114
# https://lilianweng.github.io/posts/2018-08-12-vae/
#
# m.mieskolainen@imperial.ac.uk, 2024

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

"""

class Encoder(nn.Module):
    def __init__(self, D, hidden_dim=128, latent_dim=32, activation='tanh', batch_norm=False, dropout=0.0):
        super(Encoder, self).__init__()

        self.mlp = MLP_ALL_ACT([D, hidden_dim, latent_dim], activation=activation, batch_norm=batch_norm, dropout=dropout)
        
        # modules  = [self.linear2, torch.sigmoid()]
        #self.linear2 = nn.Sequential(*modules)

    def forward(self, x):
        return self.mlp(x)

class VariationalEncoder(nn.Module):
    def __init__(self, D, hidden_dim, latent_dim, activation='relu', batch_norm=False, dropout=0.0):
        super(VariationalEncoder, self).__init__()
        
        self.mlp        = MLP_ALL_ACT([D, hidden_dim, hidden_dim], activation=activation, batch_norm=batch_norm, dropout=dropout)
        self.mlp_mu     = MLP_ALL_ACT([hidden_dim, latent_dim],    activation='tanh', batch_norm=batch_norm, dropout=dropout)
        self.mlp_logvar = MLP_ALL_ACT([hidden_dim, latent_dim],    activation='tanh', batch_norm=batch_norm, dropout=dropout)
        
        self.N          = torch.distributions.Normal(0,1)

    def forward(self, x):
        q        = self.mlp(x)

        # "Re-parametrization" trick to allow gradients backpropagate
        # For more info, see: https://arxiv.org/abs/1805.08498
        mu       = self.mlp_mu(q)
        logvar   = self.mlp_logvar(q)
        std      = torch.exp(0.5 * logvar)

        z        = mu + std*self.N.sample(mu.shape)

        return z, mu, std

    def to_device(self, device):
        """ Needed for cuda
        """
        self.N.loc   = self.N.loc.to(device) # Get sampling on GPU "hack"
        self.N.scale = self.N.scale.to(device)
        return self


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, D, activation='tanh', batch_norm=False, dropout=0.0):
        super(Decoder, self).__init__()

        self.mlp = MLP([latent_dim, hidden_dim, hidden_dim, D], activation=activation, batch_norm=batch_norm, dropout=dropout)

    def forward(self, z):
        return self.mlp(z)


class VAE(nn.Module):
    def __init__(self, D, latent_dim, hidden_dim,
            encoder_bn=True, encoder_act='relu', encoder_dropout=0.0, decoder_bn=False, decoder_act='relu', decoder_dropout=0.0,
            reco_prob='Gaussian', kl_prob='Gaussian', anomaly_score='KL_RECO', C=None):

        super(VAE, self).__init__()
        
        self.D             = D
        self.C             = C
        self.reco_prob     = reco_prob
        self.kl_prob       = kl_prob
        self.anomaly_score = anomaly_score

        self.encoder = VariationalEncoder(D=D, hidden_dim=hidden_dim, latent_dim=latent_dim,
            activation=encoder_act, batch_norm=encoder_bn, dropout=encoder_dropout)
        self.decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim,
            activation=decoder_act, batch_norm=decoder_bn, dropout=encoder_dropout, D=D)

    def to_device(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)
        self.encoder.to_device(device)
        return self

    def forward(self, x):
        z,mu,std = self.encoder(x)
        xhat     = self.decoder(z)
        return xhat,z,mu,std

    def softpredict(self, x):
        xhat,z,mu,std = self.forward(x)
        
        # "Test statistic"
        if   self.anomaly_score == 'RECO':

            score = self.log_pxz(x=x, xhat=xhat)
            return 1/(1 + torch.exp(-1/score))
            
        elif self.anomaly_score == 'KL_RECO':

            score = self.loss_kl_reco(x=x, xhat=xhat, z=z, mu=mu, std=std)
            return 1/(1 + torch.exp(1/score))
        
        else:
            raise Exception(__name__ + f'.softpredict: Unknown <anomaly_score> = {self.anomaly_score} selected.')

    def loss_kl_reco(self, x, xhat, z, mu, std, beta=1.0):
        """
        min ( E_q[log q(z|x) - log p(z)] - E_q log p(x|z) )
        """
        return beta * self.kl_div(z=z, mu=mu, std=std) - self.log_pxz(x=x, xhat=xhat)

    def log_pxz(self, x, xhat):
        """
        Reconstruction loss
        
        log p(x|z)
        """

        if   self.reco_prob == 'Bernoulli':
            # For continous Bernoulli, see: https://arxiv.org/abs/1907.06845

            reco = F.binary_cross_entropy(xhat, x, reduction='none')
            reco = reco.sum(dim=-1) # Sum over all dimensions

            return reco

        elif self.reco_prob == 'Gaussian':
            log_scale = nn.Parameter(torch.Tensor([0.0])).to(x.device)
            scale     = torch.exp(log_scale)
            dist      = torch.distributions.Normal(xhat, scale)

            reco = dist.log_prob(x)
            reco = reco.sum(dim=-1) # Sum over all dimensions

            return reco

        else:
            raise Exception(__name__ + f'.log_pxz: Unknown reco_prob = <{self.reco_prob}> chosen.')

    def kl_div(self, z, mu, std):
        """
        KL divergence (always positive), taken against a diagonal multivariate normal here
        
        log q(z|x) - log p(z)
        """
        # We use here normal densities
        if self.kl_prob == 'Gaussian':
            p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
            q = torch.distributions.Normal(mu, std)
        else:
            raise Exception(__name__ + f'.kl_div: Unknown kl_prob = <{self.kl_prob}> chosen')

        # Log-density values
        log_qzx = q.log_prob(z)
        log_pz  = p.log_prob(z)

        # KL
        kl = log_qzx - log_pz
        kl = kl.sum(dim=-1) # Sum over all dimensions
        
        # Analytic KL-divergence against (diagonal) unit multivariate Gaussian (Wikipedia)
        # Take torch.sum(self.kl_i, dim=-1) to obtain the MC estimate
        #kl = (0.5 * (std.pow(2) + mu.pow(2) - 1 - logvar).sum(dim=-1)

        return kl
