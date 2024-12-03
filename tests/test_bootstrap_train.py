# Boostrapped AI models at training stage example
#
# Run with: python filename.py
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import matplotlib.pyplot as plt

N = 1000    # Number of events
B = 50      # Number of bootstrap trained AI models

# Generate some dummy data
eta         = np.random.normal(loc=0, scale=1.5, size=N)   # Observable of interest
weights     = np.random.uniform(0.0, 1.0, size=N)          # Default AI model weights
weights_BS  = np.random.uniform(0.0, 1.0, size=(N, B))     # Bootstrap trained AI model weights

# Define histogram bins
bins = np.linspace(-4, 4, 21)  # Example bin edges
bin_centers = 0.5 * (bins[:-1] + bins[1:])

# Compute the original histogram
orig_h, _ = np.histogram(eta, bins=bins, weights=weights)

# Compute histograms for each bootstrap trained AI model
histograms = np.array([
    np.histogram(eta, bins=bins, weights=weights_BS[:, c])[0]
    for c in range(B)
])

# Compute standard deviation across different AI models as a simple proxy measure
std_BS = np.std(histograms, axis=0)

# Compute percentile bootstrap for 1 sigma, protect non-nested intervals with abs
# and symmetrize (average) over upper/lower interval
lo_BS  = np.percentile(histograms, q=16, axis=0)
hi_BS  = np.percentile(histograms, q=84, axis=0)
prc_BS = np.array([np.abs(lo_BS - orig_h), np.abs(hi_BS - orig_h)])
prc_BS = np.mean(prc_BS, axis=0)

# Plot the original histogram with bootstrap error bars
plt.errorbar(x=bin_centers, y=orig_h, yerr=std_BS, fmt='s', color='black', label='$\\pm 1 \\sigma$ (std)', lw=8.0)
plt.errorbar(x=bin_centers, y=orig_h, yerr=prc_BS, fmt='s', color='red',   label='$\\pm 1 \\sigma$ (symmetrized prc)', lw=2.5)

plt.xlabel("Observable $x$")
plt.ylabel("Weighted Count")
plt.ylim([0,None])
plt.title(f'Number of events = {N}, Number of bootstrap trained models = {B}')
plt.legend()
plt.show()
