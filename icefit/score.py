# Case study of implementing "score test" with pytorch
# for the population mean
#
# m.mieskolainen@imperial.ac.uk, 2021

import torch
import numpy as np
from torch.autograd import Variable, grad
import scipy.stats

from torch.autograd.functional import jacobian as jacobian
from torch.autograd.functional import hessian as hessian


mu_true     = 3.0


# Observed data (sampled under H0)
N           = 150
x_data      = torch.randn(N) + mu_true


# Parameters of interest
mu_0        = mu_true * torch.ones(1)
sigma_0_hat = Variable(torch.ones(1), requires_grad=True)


def logL(mu, sigma):
    """
    Log likelihood function logL(mu,sigma | X)
    """
    return torch.distributions.Normal(loc=mu, scale=sigma).log_prob(x_data).sum()

# Optimize log-likelihood function
opt = torch.optim.Adam([sigma_0_hat], lr=0.01)

for epoch in range(1500):
    opt.zero_grad()
    loss = -logL(mu_0, sigma_0_hat)
    loss.backward()
    opt.step()

print(f'Parameters found under H0:')
print(f'sigma: {sigma_0_hat} (expected: {torch.std(x_data)})')

theta_0_hat = (mu_0, sigma_0_hat)


### Jacobian -- partial derivatives of log likelihood w.r.t. the parameters == "score"
U = torch.tensor(jacobian(logL, theta_0_hat))

### Observed Fisher information matrix
I = -torch.tensor(hessian(logL, theta_0_hat)) / N


### Score test based test statistic, this is zero at the maximum likelihood estimate
S = torch.t(U) @ torch.inverse(I) @ U / N


### S follows chi^2 distribution asymptotically (use MC for non-asymptotic behavior)
pval_score_test = 1 - scipy.stats.chi2(df = 1).cdf(S)
print(f'p-value Chi^2-based score test: {pval_score_test:0.3f}')

### Student's t-test:
pval_t_test = scipy.stats.ttest_1samp(x_data, popmean = mu_true).pvalue
print(f'p-value Student\'s t-test: {pval_t_test:0.3f}')

