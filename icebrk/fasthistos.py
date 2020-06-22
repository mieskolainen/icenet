# Raw "fast" observable containers for B/RK analyzer
# 
# 
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk


import bz2
import copy
import numpy as np
import iceplot
import icebrk.tools as tools


obs_template = {

# Axis limits
'xlim'    : None,
'ylim'    : None,
'xlabel'  : r'',
'ylabel'  : r'Counts',
'units'   : r'',
'label'   : r'',
'figsize' : (4,4),

# Histogramming
'bins'    : iceplot.stepspace(0.0, 10.0, 0.1),
'density' : False,

# Function to calculate
'func'    : None,

# Disk save
'pickle'  : False
}


# Fast triplet histograms
fasthist = {
      'BToKEE_l1_isPF': {'xmin': 0, 'xmax': 2, 'nbins': 2},
      'BToKEE_l2_isPF': {'xmin': 0, 'xmax': 2, 'nbins': 2}
}


def initialize():
    """Initialize histogram dictionaries.

    Args:

    Returns:
        obj
    """

    # For signal and background
    hobj = {'S': dict(), 'B': dict()}

    # Over different sources
    for mode in hobj.keys():

        # Over histograms
        for key in fasthist.keys():
            obs             = copy.deepcopy(obs_template)
            obs['xlabel']   = key
            obs['bins']     = np.linspace(fasthist[key]['xmin'], fasthist[key]['xmax'], fasthist[key]['nbins'])
            hobj[mode][key] = copy.deepcopy(obs)

    return hobj
