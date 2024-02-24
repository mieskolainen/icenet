# m.mieskolainen@imperial.ac.uk, 2023

import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle

sys.path.append(".")
import icenet
from iceplot import iceplot

hash  = 'RpK0tm4jMTIvAThS_HRL+yeV2kGHMY+yjD9AqDiM0H0=__aJw1b950j1f+MA6MsrDTY6_E9xq7u+Knq27coZ9TfOg='
fname = f'/vols/cms/mmieskol/icenet/output/dqcd/processed_data_train_{hash}.pkl'

with open(fname, 'rb') as file:
    obj = pickle.load(file)

data = obj['trn']['data']

# Selections
new  = data['muonSV_mass_0 > 0 AND muonSV_chi2_0 < 5 AND muonSV_deltaR_0 < 0.7']

# Pick out the variable of interest
new  = new['muonSV_mass_0']
x    = new.x.squeeze()
y    = new.y.astype(int)
w    = new.w.squeeze()

obs_M = {

    # Axis limits
    'xlim'    : (0, 7.0),
    'ylim'    : None,
    'xlabel'  : r'$M$',
    'ylabel'  : r'Counts',
    'units'   : {'x': r'GeV', 'y' : r'counts'},
    'label'   : r'Invariant mass',
    'figsize' : (4, 3.75),

    # Ratio
    'ylim_ratio' : (0.25, 1.75),

    # Histogramming
    'bins'    : np.linspace(0, 7.0, 60),
    'density' : True,

    # Function to calculate
    #'func'    : pt2
}

# ------------------------------------------------------------------------
# ** Histogramming **

fig3, ax3 = iceplot.create_axes(**obs_M, ratio_plot=True)

counts0, errs, bins, cbins = iceplot.hist(x[y==0], weights=w[y==0], bins=obs_M['bins'], density=obs_M['density'])
ax3[0].hist(x=cbins, bins=bins, weights=counts0, color=(0,0,0), label='QCD MC', **iceplot.hist_style_step)

counts2, errs, bins, cbins = iceplot.hist(x[y==2], weights=w[y==2], bins=obs_M['bins'], density=obs_M['density'])
ax3[0].hist(x=cbins, bins=bins, weights=counts2, color=(1,0,0), alpha=0.5, label='Data (2018-D)', **iceplot.hist_style_step)

iceplot.ordered_legend(ax = ax3[0], order=['QCD MC', 'Data (2018-D)'])

# Ratio
iceplot.plot_horizontal_line(ax3[1])
ax3[1].hist(x=cbins, bins=bins, weights=counts2 / (counts0 + 1E-30), color=(1,0,0), alpha=0.5, label='Data / MC', **iceplot.hist_style_step)

fig3.savefig('muonSV_mass_0.pdf', bbox_inches='tight')
