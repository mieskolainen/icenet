# Electron ID [DEEP BATCHED TRAINING] steering code
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

# icenet system paths
import sys
sys.path.append(".")

# Configure plotting backend
import matplotlib
matplotlib.use('Agg')

import uproot
import math
import numpy as np
import torch
import torch_geometric
from matplotlib import pyplot as plt

import argparse
import pprint
import os
import datetime
import json
import yaml

import pickle
import sys
import copy
from termcolor import cprint

# icenet
from icenet.tools import io
from icenet.tools import aux
from icenet.tools import aux_torch
from icenet.tools import reweight
from icenet.tools import plots
from icenet.tools import prints
from icenet.tools import iceroot


# deep learning
from icenet.deep  import train
from icenet.tools import process
import icenet.deep as deep

# iceid
from iceid import common
from iceid import graphio
from configs.eid.mvavars import *


def get_model(gdata, args, param):

    # =========================================================================
    # INITIALIZE GRAPH MODEL

    # Get model
    netparam, conv_type = train.getgraphparam(data_trn=gdata['trn'], num_classes=args['num_classes'], param=param)
    model               = train.getgraphmodel(conv_type=conv_type, netparam=netparam)

    # CPU or GPU
    model, device       = deep.optimize.model_to_cuda(model=model, device_type=param['device'])

    # Count the number of parameters
    cprint(__name__ + f'.graph_train: Number of free parameters = {aux_torch.count_parameters_torch(model)}', 'yellow')
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=param['opt_param']['lr'], weight_decay=param['opt_param']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=param['scheduler_param']['step_size'], gamma=param['scheduler_param']['gamma'])

    return model, device, optimizer, scheduler


def compute_reweight(root_files, num_events, args):

    index        = 0 # Use the first file by default
    cprint(__name__ + f': Loading from {root_files[index]} for differential re-weight PDFs', 'yellow')

    entry_stop = np.min([args['reweight_param']['maxevents'], num_events[index]])
    predata    = common.load_root_file(root_path=[root_files[index]], ids=None, entry_stop=entry_stop, args=args, library='np')
    

    X,Y,W,ids  = predata['X'],predata['Y'],predata['W'],predata['ids']

    # Compute re-weights
    _, pdf = reweight.compute_ND_reweights(x=X, y=Y, w=W, ids=ids, args=args['reweight_param'])

    return pdf, X, Y, W, ids


# Main function
#
def main():
    
    ### Get input
    cli, cli_dict  = process.read_cli()
    runmode   = cli_dict['runmode']

    args, cli  = process.read_config(config_path='./configs/eid', runmode=runmode)
    root_files = args['root_files']
    scalar_var = globals()[args['inputvar_scalar']]
    
    # Create save path
    args["modeldir"] = aux.makedir(f'./checkpoint/eid/{args["config"]}/')
    
    # Load stats
    num_events = iceroot.load_tree_stats(rootfile=root_files, tree=args['tree_name'])
    print(f'Number of events per file: {num_events}')


    # =========================================================================
    # Load data for each re-weight PDFs

    pdf,X,Y,W,ids = compute_reweight(root_files=root_files, num_events=num_events, args=args)

    gdata = {}
    gdata['trn'] = graphio.parse_graph_data(X=X, Y=Y, ids=ids, weights=W, entry_start=0, entry_stop=1,
        features=scalar_var, graph_param=args['graph_param'])
    
    # =========================================================================
    ### Initialize all models
    model     = {}
    device    = {}
    optimizer = {}
    scheduler = {}
    param     = {}

    for i in range(len(args['active_models'])):

        ID        = args['active_models'][i]
        param[ID] = args['models'][ID]
        
        if param[ID]['train'] == 'torch_graph':
            model[ID], device[ID], optimizer[ID], scheduler[ID] = get_model(gdata, args=args, param=param[ID])
    # -------------------------------------------------------------------------

    visited    = False
    N_epochs   = args['batch_train_param']['epochs']
    block_size = args['batch_train_param']['blocksize']

    ### Over each global epoch
    for epoch in range(N_epochs):

        prints.printbar('=')
        cprint(__name__ + f".epoch {epoch+1} / {N_epochs} \n", 'yellow')

        ### Over each file
        for f in range(len(root_files)):

            prints.printbar('=')
            cprint(__name__ + f'.file: {f+1} / {len(root_files)} (events = {num_events[f]}) \n', 'yellow')
            
            # ------------------------------------------------------------
            N_blocks   = int(np.ceil(num_events[f] / args['batch_train_param']['blocksize']))
            block_ind  = aux.split_start_end(range(num_events[f]), N_blocks)
            # ------------------------------------------------------------

            ### Over blocks of data from this file
            for block in range(N_blocks):
                
                entry_start = block_ind[block][0]
                entry_stop  = block_ind[block][-1]
                
                prints.printbar('=')
                cprint(__name__ + f'.block: {block+1} / {N_blocks} (events = {entry_stop - entry_start} | total = {num_events[f]}) \n', 'yellow')

                # =========================================================================
                # LOAD DATA
                if (N_blocks > 1) or (len(root_files) > 1) or (len(root_files) == 1 and N_blocks == 1 and visited == False):

                    visited = True # For the special case

                    predata        = common.load_root_file(root_path=[root_files[f]], entry_start=entry_start, entry_stop=entry_stop, args=args, library='np')
                    
                    X,Y,W,ids      = predata['X'],predata['Y'],predata['W'],predata['ids']
                    trn, val, tst  = io.split_data(X=X, Y=Y, W=W, ids=ids, frac=args['frac'])
                    
                    # =========================================================================
                    # COMPUTE RE-WEIGHTS
                    
                    trn_weights,_ = reweight.compute_ND_reweights(pdf=pdf, x=trn.x, y=trn.y, w=trn.w, ids=ids, args=args['reweight_param'])
                    val_weights,_ = reweight.compute_ND_reweights(pdf=pdf, x=val.x, y=val.y, w=val.w, ids=ids, args=args['reweight_param'])

                    # =========================================================================
                    ### Parse data into graphs
                    
                    gdata = {}
                    gdata['trn'] = graphio.parse_graph_data(X=trn.x, Y=trn.y, ids=ids, weights=trn_weights,
                        features=scalar_var, graph_param=args['graph_param'])

                    gdata['val'] = graphio.parse_graph_data(X=val.x, Y=val.y, ids=ids, weights=val_weights,
                        features=scalar_var, graph_param=args['graph_param'])
                    
                    io.showmem()

                # =========================================================================
                ### Train all model over this block of data
                for ID in model.keys():
                    cprint(__name__ + f' Training model <{ID}>', 'green')

                    train_loader     = torch_geometric.loader.DataLoader(gdata['trn'], batch_size=param[ID]['opt_param']['batch_size'], shuffle=True)
                    test_loader      = torch_geometric.loader.DataLoader(gdata['val'], batch_size=512, shuffle=False)
                    
                    # Train
                    loss             = deep.optimize.train(model=model[ID], loader=train_loader, optimizer=optimizer[ID], device=device[ID], opt_param=param[ID]['opt_param'])
                    
                    # Evaluate
                    trn_acc, trn_AUC = deep.optimize.test( model=model[ID], loader=train_loader, optimizer=optimizer[ID], device=device[ID])
                    val_acc, val_AUC = deep.optimize.test( model=model[ID], loader=test_loader,  optimizer=optimizer[ID], device=device[ID])
                    
                    scheduler[ID].step()
                    
                    print(f"[epoch: {epoch+1:03d}/{N_epochs:03d} | file: {f+1}/{len(root_files)} | block: {block+1}/{N_blocks} | "
                        f"train loss: {deep.optimize.printloss(loss)} | train: {trn_acc:.4f} (acc), {trn_AUC:.4f} (AUC) | validate: {val_acc:.4f} (acc), {val_AUC:.4f} (AUC) | lr = {scheduler[ID].get_last_lr()}")
                    
        ## Save each model per global epoch
        for ID in model.keys():
            checkpoint = {'model': model[ID], 'state_dict': model[ID].state_dict()}
            torch.save(checkpoint, args['modeldir'] + f'/{param[ID]["label"]}_{epoch}' + '.pth')

    print(__name__ + f' [Done!]')

if __name__ == '__main__' :
    main()
