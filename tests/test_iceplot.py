# Iceplot examples
# 
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk


import numpy as np
import matplotlib.pyplot as plt
import math
import os

import sys
sys.path.append(".")
import iceplot

import pathlib
pathlib.Path("./testfigs").mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------
iceplot.set_global_style()


# Synthetic input data
r1 = np.random.randn(25000) * 0.8
r2 = np.random.randn(25000) * 1
r3 = np.random.randn(25000) * 1.2
r4 = np.random.randn(25000) * 1.5


# ------------------------------------------------------------------------
# Mathematical definitions

# Momentum squared
def pt2(x):
    return np.power(x,2);


# ------------------------------------------------------------------------
# Observables containers

obs_pt2 = {

# Axis limits
'xlim'    : (0, 1.5),
'ylim'    : None,
'xlabel'  : r'$p_t^2$',
'ylabel'  : r'Counts',
'units'   : r'GeV$^2$',
'label'   : r'Transverse momentum squared',
'figsize' : (4,4),

# Histogramming
'bins'    : np.linspace(0, 1.5, 60),
'density' : False,
 
# Function to calculate
'func'    : pt2
}


# ------------------------------------------------------------------------
# ** Example **

fig1, ax1 = iceplot.create_axes(**obs_pt2, ratio_plot=False)
counts, errs, bins, cbins = iceplot.hist(obs_pt2['func'](r1), bins=obs_pt2['bins'], density=obs_pt2['density'])
ax1[0].errorbar(x=cbins, y=counts, yerr=errs, color=(0,0,0), label='Data $\\alpha$', **iceplot.errorbar_style)
ax1[0].legend(frameon=False)
fig1.savefig('./testfigs/testplot_1.pdf', bbox_inches='tight')


# ------------------------------------------------------------------------
# ** Example **

fig2, ax2 = iceplot.create_axes(**obs_pt2, ratio_plot=False)
counts, errs, bins, cbins = iceplot.hist(obs_pt2['func'](r1), bins=obs_pt2['bins'], density=obs_pt2['density'])
ax2[0].hist(x=cbins, bins=bins, weights=counts, color=(0.5, 0.2, 0.1), label='Data $\\alpha$', **iceplot.hist_style_step)
ax2[0].legend(frameon=False)
fig2.savefig('./testfigs/testplot_2.pdf', bbox_inches='tight')


# ------------------------------------------------------------------------
# ** Example **

fig3, ax3 = iceplot.create_axes(**obs_pt2, ratio_plot=True)

counts1, errs, bins, cbins = iceplot.hist(obs_pt2['func'](r1), bins=obs_pt2['bins'], density=obs_pt2['density'])
ax3[0].hist(x=cbins, bins=bins, weights=counts1, color=(0,0,0), label='Data 1', **iceplot.hist_style_step)

counts2, errs, bins, cbins = iceplot.hist(obs_pt2['func'](r2), bins=obs_pt2['bins'], density=obs_pt2['density'])
ax3[0].hist(x=cbins, bins=bins, weights=counts2, color=(1,0,0), alpha=0.5, label='Data 2', **iceplot.hist_style_step)

iceplot.ordered_legend(ax = ax3[0], order=['Data 1', 'Data 2'])

# Ratio
iceplot.plot_horizontal_line(ax3[1])
ax3[1].hist(x=cbins, bins=bins, weights=counts2 / (counts1 + 1E-30), color=(1,0,0), alpha=0.5, label='Data $\\beta$', **iceplot.hist_style_step)

fig3.savefig('./testfigs/testplot_3.pdf', bbox_inches='tight')


# ------------------------------------------------------------------------
# ** Example **

data_template = {
    'data'   : None,
    'weights': None,
    'legend' : 'Data',
    'hfunc'  : 'errorbar',
    'style'  : iceplot.errorbar_style,
    'obs'    : obs_pt2,
    'hdata'  : None,
    'color'  : None
}

# Data source <-> Observable collections
data1 = data_template.copy() # Deep copies
data2 = data_template.copy()
data3 = data_template.copy()
data4 = data_template.copy()

data1.update({
    'data'   : r1,
    'legend' : 'Data $\\alpha$',
    'hfunc'  : 'errorbar',
    'style'  : iceplot.errorbar_style,
})
data2.update({
    'data'   : r2,
    'legend' : 'Data $\\beta$',
    'hfunc'  : 'hist',
    'style'  : iceplot.hist_style_step,
})
data3.update({
    'data'   : r3,
    'legend' : 'Data $\\gamma$',
    'hfunc'  : 'hist',
    'style'  : iceplot.hist_style_step,
})
data4.update({
    'data'   : r4,
    'legend' : 'Data $\\delta$',
    'hfunc'  : 'plot',
    'style'  : iceplot.plot_style,
})

data = [data1, data2, data3, data4]


# Calculate histograms
for i in range(len(data)):
    data[i]['hdata'] = iceplot.hist_obj(data[i]['obs']['func'](data[i]['data']), bins=data[i]['obs']['bins'], density=data[i]['obs']['density'])

# Plot it
fig4, ax4 = iceplot.superplot(data, ratio_plot=True, yscale='log')
fig5, ax5 = iceplot.superplot(data, ratio_plot=True, yscale='linear', ratio_error_plot=False)

fig4.savefig('./testfigs/testplot_4.pdf', bbox_inches='tight')
fig5.savefig('./testfigs/testplot_5.pdf', bbox_inches='tight')

plt.show()




