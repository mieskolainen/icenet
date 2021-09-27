# Electron HLT trigger [TRAINING] steering code
#
# Mikael Mieskolainen, 2021
# m.mieskolainen@imperial.ac.uk

# icenet system paths
import sys
sys.path.append(".")

import math
import numpy as np
import torch
import os
import pickle
import sys
from termcolor import cprint

# matplotlib
from matplotlib import pyplot as plt

# icenet
from icenet.tools import io
from icenet.tools import aux
from icenet.tools import reweight

from icenet.tools import plots
from icenet.tools import prints
from icenet.tools import process


# icetrg
from icetrg import common


# Main function
#
def main() :

    ### Get input
    data, args, features = common.init()

    ### Print ranges
    #prints.print_variables(X=data.trn.x, ids=data.ids)
    
    ### Compute reweighting weights
    trn_weights = reweight.compute_ND_reweights(x=data.trn.x, y=data.trn.y, ids=data.ids, args=args['reweight_param'])


    ### Plot some kinematic variables
    targetdir = f'./figs/trg/{args["config"]}/reweight/1D_kinematic/'
    os.makedirs(targetdir, exist_ok = True)
    for k in ['x_hlt_pt', 'x_hlt_eta']:
        plots.plotvar(x = data.trn.x[:, data.ids.index(k)], y = data.trn.y, weights = trn_weights, var = k, NBINS = 70,
            targetdir = targetdir, title = f"training re-weight reference class: {args['reweight_param']['reference_class']}")

    ### Plot variables
    if args['plot_param']['basic_on'] == True:
        print(__name__ + f': plotting basic histograms ...')
        targetdir = f'./figs/trg/{args["config"]}/train/1D_all/'; os.makedirs(targetdir, exist_ok = True)
        plots.plotvars(X = data.trn.x, y = data.trn.y, NBINS = 70, ids = data.ids,
            weights = trn_weights, targetdir = targetdir, title = f'training reweight reference: {args["reweight_param"]["mode"]}')

    ### Split and factor data
    data, data_kin = common.splitfactor(data=data, args=args)

    ### Print scalar variables
    fig,ax = plots.plot_correlations(data.trn.x, data.ids)
    targetdir = f'./figs/trg/{args["config"]}/train/'; os.makedirs(targetdir, exist_ok = True)
    plt.savefig(fname = targetdir + 'correlations.pdf', pad_inches = 0.2, bbox_inches='tight')
    
    print(__name__ + ': Active variables:')
    prints.print_variables(X=data.trn.x, ids=data.ids)
    
    # Add args['modeldir']
    args["modeldir"] = f'./checkpoint/trg/{args["config"]}/'; os.makedirs(args["modeldir"], exist_ok = True)
    
    ### Execute training
    process.train_models(data = data, data_kin = data_kin, trn_weights = trn_weights, args = args)
        
    print(__name__ + ' [done]')


if __name__ == '__main__' :
   main()
