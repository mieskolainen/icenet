# Electron ID [TRAINING] steering code
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

# icenet system paths
import _icepaths_

import math
import numpy as np
import argparse
import pprint
import os
import datetime
import json
import pickle
import sys
import yaml
import copy
from tqdm import tqdm
#import graphviz

from termcolor import cprint

# matplotlib
from matplotlib import pyplot as plt

from icenet.tools import plots

# iceid
from iceid import common



# Main function
#
def main() :

    ### Get input
    data, args, features = common.init(MAXEVENTS=3000)

    
    targetdir = f'./figs/eid/{args["config"]}/image/'; os.makedirs(targetdir, exist_ok = True)

    ### Split and factor data
    data, data_tensor, data_kin = common.splitfactor(data=data, args=args)

    ### Mean images
    VMAX    = 0.5 # GeV, maximum visualization scale
    channel = 0
    
    for class_ind in [0,1]:

        # <E>
        XY = np.mean(data_tensor['trn'][(data.trn.y == class_ind), channel, :,:], axis=0)

        fig,ax,c = plots.plot_matrix(XY = XY,
            x_bins=args['image_param']['eta_bins'],
            y_bins=args['image_param']['phi_bins'],
            vmin=0, vmax=VMAX, figsize=(5,3), cmap='hot')

        ax.set_xlabel('$\\eta$')
        ax.set_ylabel('$\\phi$ [rad]')

        fig.colorbar(c, ax=ax)
        ax.set_title(f'$\\langle E \\rangle$ GeV | class = {class_ind}')
        plt.savefig(f'{targetdir}/mean_E_channel_{channel}_class_{class_ind}.pdf', bbox_inches='tight')
        plt.close()

    ### Loop over individual events
    for i in tqdm(range(np.min([100, data_tensor['trn'].shape[0]]))):

        fig,ax,c = plots.plot_matrix(XY=data_tensor['trn'][i,channel,:,:],
            x_bins=args['image_param']['eta_bins'],
            y_bins=args['image_param']['phi_bins'],
            vmin=0, vmax=VMAX, figsize=(5,3), cmap='hot')

        ax.set_xlabel('$\\eta$')
        ax.set_ylabel('$\\phi$ [rad]')
        
        pt  = data_kin.trn.x[i, data_kin.VARS.index("trk_pt")]
        eta = data_kin.trn.x[i, data_kin.VARS.index("trk_eta")]
        phi = data_kin.trn.x[i, data_kin.VARS.index("trk_phi")]
        ax.set_title(f'Track $(p_t = {pt:0.1f}, \\eta = {eta:0.1f}, \\phi = {phi:0.1f})$ | class = {data.trn.y[i]:0.0f}')

        fig.colorbar(c, ax=ax)
        os.makedirs(f'{targetdir}/channel_{channel}_class_{data.trn.y[i]:0.0f}/', exist_ok = True)
        plt.savefig(f'{targetdir}/channel_{channel}_class_{data.trn.y[i]:0.0f}/{i}.pdf', bbox_inches='tight')
        plt.close()

    print(__name__ + ' [done]')


if __name__ == '__main__' :

   main()

