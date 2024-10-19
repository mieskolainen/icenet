# Electron ID [TENSOR VISUALIZATION] steering code
#
# N.B. One needs to have a classifier with
# predict: 'torch_image' turned on in the steering .json file.
# (otherwise no image tensor input is processed)
# 
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

# icenet system paths
import sys
sys.path.append(".")

# Configure plotting backend
import matplotlib
matplotlib.use('Agg')

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

from icenet.tools import aux
from icenet.tools import plots
from icenet.tools import process
from icenet.algo import nmf

import numpy as np

# iceid
from iceid import common

# Main function
#
def main() :
    
    runmode     = 'genesis'
    args, cli   = process.read_config(config_path=f'configs/eid', runmode=runmode)
    process.process_raw_data(args=args, func_loader=common.load_root_file) 
    
    runmode     = 'train'
    args, cli   = process.read_config(config_path=f'configs/eid', runmode=runmode)
    
    # Force tensor processing on
    args['inputvar_image'] = 'CMSSW_MVA_IMAGE_VARS'
    
    rawdata     = process.combine_pickle_data(args=args) 
    data        = process.process_data(args=args, data=rawdata, func_factor=common.splitfactor, mvavars='configs.eid.mvavars', runmode='train')
    
    # Pick them out
    data_tensor = data['trn']['data_tensor']
    data_kin    = data['trn']['data_kin']
    data        = data['trn']['data']

    targetdir = aux.makedir(f'./figs/{args["rootname"]}/{args["config"]}/image/')
    
    
    # --------------------------------------------------------------------
    # NMF factorization

    k = 3        # Number of basis components

    for channel in [0,1]:  # Which calorimeter

        for class_ind in [0,1]:
            
            V = data_tensor[(data.y == class_ind), channel, :,:]
            print(V.shape)

            V_new = np.zeros((V.shape[1]*V.shape[2], V.shape[0]))
            for i in range(V.shape[0]):
                V_new[:,i] = V[i,:,:].flatten() + 1e-12
            print(V_new.shape)

            # Non-Negative matrix factorization
            W,H = nmf.ML_nmf(V=V_new, k=k, threshold=1e-5, maxiter=300)
            
            # Loop over "basis" components
            for i in range(k):
                B = W[:,i].reshape(V.shape[1],V.shape[2])
                print(B.shape)
                
                fig,ax,c = plots.plot_matrix(XY = B,
                    x_bins=args['image_param']['eta_bins'],
                    y_bins=args['image_param']['phi_bins'],
                    vmin=0, vmax=None, figsize=(5,3), cmap='hot')
                
                ax.set_xlabel('$\\eta$')
                ax.set_ylabel('$\\phi$ [rad]')
                fig.colorbar(c, ax=ax)
                ax.set_title(f'basis matrix $b_{{{i}}}$ | $E$ GeV | class = {class_ind} | channel = {channel}')

                os.makedirs(f'{targetdir}/NMF/', exist_ok = True)            
                plt.savefig(f'{targetdir}/NMF/channel_{channel}_class_{class_ind}_basis_{i}.pdf', bbox_inches='tight')
                #plt.show()
                plt.close()

    # --------------------------------------------------------------------

    ### Moment images
    VMAX    = 0.5 # GeV, maximum visualization scale
    
    for channel in [0,1]:
        
        for class_ind in [0,1]:

            for moment in ['mean', 'std']:

                XY = data_tensor[(data.y == class_ind), channel, :,:]

                if   moment == 'mean':
                    XY = np.mean(XY, axis=0)
                elif moment == 'std':
                    XY = np.std(XY, axis=0)

                fig,ax,c = plots.plot_matrix(XY = XY,
                    x_bins=args['image_param']['eta_bins'],
                    y_bins=args['image_param']['phi_bins'],
                    vmin=0, vmax=None, figsize=(5,3), cmap='hot')

                ax.set_xlabel('$\\eta$')
                ax.set_ylabel('$\\phi$ [rad]')

                fig.colorbar(c, ax=ax)
                ax.set_title(f'{moment}$(E)$ GeV | class = {class_ind}')
                plt.savefig(f'{targetdir}/{moment}_E_channel_{channel}_class_{class_ind}.pdf', bbox_inches='tight')
                plt.close()


        ### Loop over individual events
        MAXN = 30 # max number

        for i in tqdm(range(np.min([MAXN, data_tensor.shape[0]]))):

            fig,ax,c = plots.plot_matrix(XY=data_tensor[i,channel,:,:],
                x_bins=args['image_param']['eta_bins'],
                y_bins=args['image_param']['phi_bins'],
                vmin=0, vmax=VMAX, figsize=(5,3), cmap='hot')

            ax.set_xlabel('$\\eta$')
            ax.set_ylabel('$\\phi$ [rad]')
            
            pt  = data_kin.x[i, data_kin.ids.index("trk_pt")]
            eta = data_kin.x[i, data_kin.ids.index("trk_eta")]
            phi = data_kin.x[i, data_kin.ids.index("trk_phi")]
            ax.set_title(f'Track $(p_t = {pt:0.1f}, \\eta = {eta:0.1f}, \\phi = {phi:0.1f})$ | class = {data.y[i]:0.0f}')

            fig.colorbar(c, ax=ax)
            os.makedirs(f'{targetdir}/channel_{channel}_class_{data.y[i]:0.0f}/', exist_ok = True)
            plt.savefig(f'{targetdir}/channel_{channel}_class_{data.y[i]:0.0f}/event_{i}.pdf', bbox_inches='tight')
            plt.close()


    print(__name__ + ' [done]')


if __name__ == '__main__' :

   main()

