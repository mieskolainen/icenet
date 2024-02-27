# Test for reading processed DQCD data
#
# m.mieskolainen@imperial.ac.uk, 2024

import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import os

sys.path.append(".")
import icenet
from iceplot import iceplot

# ------------------------------------------------------------------------
# Find the latest file

path = './output/dqcd/processed_data_train_'

def find_latest(path, filetype):

    list_of_files = glob.glob(f'{path}*{filetype}')
    return max(list_of_files, key=os.path.getctime)

fname = find_latest(path=path, filetype='pkl')

# Load data
with open(fname, 'rb') as file:
    obj = pickle.load(file)
data = obj['trn']['data']

# ------------------------------------------------------------------------
# Selections
cutstring = 'muonSV_mass_0 > 0.2 AND muonSV_chi2_0 < 10 AND muonSV_deltaR_0 < 0.7'
new  = data[cutstring]

# Pick out the variable of interest
new  = new['muonSV_mass_0']
x    = new.x.squeeze()
y    = new.y.astype(int)
w    = new.w.squeeze()

# ------------------------------------------------------------------------
# ** Histogramming **

obs_M = {

    # Axis limits
    'xlim'    : (0, 20.0),
    'ylim'    : (1e-4, 3),
    'xlabel'  : r'$M_{\mu^+\mu^-}$',
    'ylabel'  : r'Counts',
    'units'   : {'x': r'GeV', 'y' : r'counts'},
    'label'   : r'Invariant mass',
    'figsize' : (4, 3.75),

    # Ratio
    'ylim_ratio': (0.0, 2.0),

    # Histogramming
    #'bins'    : np.logspace(np.log10(0.2), np.log10(20.0), 80),
    #'bins'    : np.linspace(0.2, 20, 80),
    'bins'    : np.array([0.0, 0.25, 0.75, 1.0, 1.25, 1.5, 1.75,
                          2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.25,
                          5.0, 6.25, 8.5, 11.5, 15.5, 20.0]),
    'density' : True,
}

# Define class labels
DATA, QCD  = -2, 0

fig, ax = iceplot.create_axes(**obs_M, ratio_plot=True)

# MC
counts0, errs0, bins, cbins = iceplot.hist(x[y==QCD], weights=w[y==QCD], bins=obs_M['bins'], density=obs_M['density'])
iceplot.hist_filled_error(ax[0], bins, cbins, counts0, errs0, color=(1,0,0))
ax[0].hist(x=cbins, bins=bins, weights=counts0, color=(1,0,0), label='QCD MC', **iceplot.hist_style_step)

# Data
counts2, errs2, bins, cbins = iceplot.hist(x[y==DATA], weights=w[y==DATA], bins=obs_M['bins'], density=obs_M['density'])

iceplot.hist_filled_error(ax[0], bins, cbins, counts2, errs2, color=(0,0,0))
#ax[0].errorbar(x=cbins, y=counts2, yerr=errs2, color=(0,0,0), label='Data (2018-D)', **iceplot.errorbar_style)
ax[0].plot(cbins, counts2, color=(0,0,0), label='Data (2018-D)', **iceplot.errorbar_style)
ax[0].hist(x=cbins, bins=bins, weights=counts2, color=(0,0,0), alpha=0.5, **iceplot.hist_style_step)

# Ratio
iceplot.plot_horizontal_line(ax[1])
ratio  = counts2 / np.clip(counts0, a_min=1E-30, a_max=None)
rerr   = iceplot.ratioerr(counts2, counts0, errs0, errs2, sigma_AB = 0, EPS = 1E-15)
iceplot.hist_filled_error(ax[1], bins, cbins, ratio, rerr, color=(0,0,0))
ax[1].hist(x=cbins, bins=bins, weights=ratio,
           color=(0,0,0), alpha=0.5, label='Data / MC', **iceplot.hist_style_step)

# Visuals
ax[0].set_title(cutstring, fontsize=6)
ax[0].set_yscale('log')
iceplot.ordered_legend(ax = ax[0], order=['Data (2018-D)', 'QCD MC'])
ax[1].set_ylabel('Data / MC')
ax[1].yaxis.label.set_fontsize(7)
for item in ax[1].get_yticklabels():
    item.set_fontsize(6)

# Save
filename = 'muonSV_mass_0.pdf'
print(f'Saving fig to: {filename}')
fig.savefig(filename, bbox_inches='tight')

