# Electron ID [TRAINING] steering code
#
# Mikael Mieskolainen, 2022
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


# iceid
from iceid import common
from iceid import graphio
from configs.eid.mvavars import *


# Main function
#
def main() :
    
    args, cli     = process.read_config(config_path='./configs/eid')

    ### Load data full in memory
    data          = io.IceTriplet(func_loader=common.load_root_file, files=args['root_files'],
                    load_args={'entry_start': 0, 'entry_stop': args['MAXEVENTS'], 'args': args},
                    class_id=[0,1], frac=args['frac'], rngseed=args['rngseed'])

    ### Imputation
    features      = globals()[args['imputation_param']['var']]
    data, imputer = process.impute_datasets(data=data, features=features, args=args['imputation_param'], imputer=None)
    pickle.dump(imputer, open(args["modeldir"] + '/imputer.pkl', 'wb'))    

    ### Print ranges
    #prints.print_variables(X=data.trn.x, ids=data.ids)

    ### Compute reweighting weights
    trn_weights,_ = reweight.compute_ND_reweights(x=data.trn.x, y=data.trn.y, ids=data.ids, args=args['reweight_param'])
    val_weights,_ = reweight.compute_ND_reweights(x=data.val.x, y=data.val.y, ids=data.ids, args=args['reweight_param'])
    
    ### Plot some kinematic variables
    targetdir = aux.makedir(f'./figs/{args["rootname"]}/{args["config"]}/reweight/1D_kinematic/')
    for k in ['trk_pt', 'trk_eta', 'trk_phi', 'trk_p']:
        plots.plotvar(x = data.trn.x[:, data.ids.index(k)], y = data.trn.y, weights = trn_weights, var = k, NBINS = 70,
            targetdir = targetdir, title = f"training re-weight reference class: {args['reweight_param']['reference_class']}")
    
    # --------------------------------------------------------------------
    ### Parse data into graphs
    
    graph = None
    if args['graph_on']:
        graph = {}
        graph['trn'] = graphio.parse_graph_data(X=data.trn.x, Y=data.trn.y, weights=trn_weights, ids=data.ids, 
            features=features, global_on=args['graph_param']['global_on'], coord=args['graph_param']['coord'])
        
        graph['val'] = graphio.parse_graph_data(X=data.val.x, Y=data.val.y, weights=val_weights, ids=data.ids,
            features=features, global_on=args['graph_param']['global_on'], coord=args['graph_param']['coord'])
    
    ### Split and factor data
    data, data_tensor, data_kin = common.splitfactor(data=data, args=args)
    
    ### Plot variables
    if args['plot_param']['basic']['active']:
        targetdir = aux.makedir(f'./figs/{args["rootname"]}/{args["config"]}/train/1D_all/')
        plots.plotvars(X = data.trn.x, y = data.trn.y, NBINS = args['plot_param']['basic']['nbins'], ids = data.ids,
            weights = trn_weights, targetdir = targetdir, title = f"training re-weight reference class: {args['reweight_param']['reference_class']}")
    
    ### Plot correlations
    targetdir = aux.makedir(f'./figs/{args["rootname"]}/{args["config"]}/train/')
    fig,ax    = plots.plot_correlations(X=data.trn.x, netvars=data.ids, classes=data.trn.y, targetdir=targetdir)

    print(__name__ + ': Active variables:')
    prints.print_variables(X=data.trn.x, ids=data.ids)

    ### Execute training
    args["modeldir"] = aux.makedir(f'./checkpoint/{args["rootname"]}/{args["config"]}/')
    process.train_models(data = data, data_tensor = data_tensor, data_kin = data_kin, data_graph = graph,
        trn_weights = trn_weights, val_weights=val_weights, args = args)
    
    print(__name__ + ' [done]')


if __name__ == '__main__' :

   main()

