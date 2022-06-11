# Electron ID [DEEP BATCHED TRAINING] steering code
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk


# icenet system paths
import sys
sys.path.append(".")

import uproot
import math
import numpy as np
import torch
import argparse
import pprint
import os
import datetime
import json
import pickle
import sys
import yaml
import copy
#import graphviz
import torch_geometric
from termcolor import cprint

# matplotlib
from matplotlib import pyplot as plt

# scikit
from sklearn         import metrics
from sklearn.metrics import accuracy_score

# icenet
from icenet.tools import io
from icenet.tools import aux
from icenet.tools import aux_torch
from icenet.tools import reweight
from icenet.tools import plots
from icenet.tools import prints

# deep learning
from icenet.deep  import train
import icenet.deep as deep

# iceid
from iceid import common
from iceid import graphio


def get_model(X, Y, ids, weights, features, args, param):
    
    # ---------------------------------------------------------------
    # Read test graph data to get dimensions

    gdata = {}
    gdata['trn'] = graphio.parse_graph_data(X=X, Y=Y, ids=ids, weights=weights, 
        features=features, global_on=args['graph_param']['global_on'], coord=args['graph_param']['coord'])
    
    # =========================================================================
    # INITIALIZE GRAPH MODEL

    # Get model
    netparam, conv_type = train.getgraphparam(data_trn=gdata['trn'], num_classes=args['num_classes'], param=param)
    model    = train.getgraphmodel(conv_type=conv_type, netparam=netparam)

    # CPU or GPU
    model, device = deep.dopt.model_to_cuda(model=model, device_type=param['device'])

    # Count the number of parameters
    cprint(__name__ + f'.graph_train: Number of free parameters = {aux_torch.count_parameters_torch(model)}', 'yellow')
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=param['opt_param']['learning_rate'], weight_decay=param['opt_param']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=param['scheduler_param']['step_size'], gamma=param['scheduler_param']['gamma'])

    return model, device, optimizer, scheduler


def compute_reweight(root_files, N_events, args):

    index        = 0 # Use the first file by default
    cprint(__name__ + f': Loading from {root_files[index]} for differential re-weight PDFs', 'yellow')

    entrystop    = np.min([args['reweight_param']['maxevents'], N_events[index]])
    X,Y,ids      = common.load_root_file(root_files[index], ids=None, class_id = [0,1], max_num_elements=entrystop, args=args, library='np')

    ### Compute differential re-weighting 2D-PDFs
    X_A          = X[:, ids.index(args['reweight_param']['var_A'])]
    X_B          = X[:, ids.index(args['reweight_param']['var_B'])]

    bins_A       = args['reweight_param']['bins_A']
    bins_B       = args['reweight_param']['bins_B']
    binedges_A   = np.linspace(bins_A[0], bins_A[1], bins_A[2])
    binedges_B   = np.linspace(bins_B[0], bins_B[1], bins_B[2])
    
    print(__name__ + f".compute_reweights: reference_class: <{args['reweight_param']['reference_class']}>")

    ### Compute 2D-pdfs for each class
    pdf = {}
    for c in range(args['num_classes']):
        pdf[c] = reweight.pdf_2D_hist(X_A=X_A[Y==c], X_B=X_B[Y==c], binedges_A=binedges_A, binedges_B=binedges_B)
    
    pdf['binedges_A'] = binedges_A
    pdf['binedges_B'] = binedges_B

    return pdf, X, Y, ids


# Main function
#
def main():

    ### Get input
    data, args, features = common.init()
    root_files = args['root_files']
    
    # Find number of events in each file
    N_events = np.zeros(len(root_files), dtype=int)

    for i in range(len(root_files)):
        file   = uproot.open(root_files[i])
        events = file["ntuplizer"]["tree"]

        X      = events.arrays('is_mc')
        N_events[i] = len(X)
        file.close()
        
        # ** Apply MAXEVENTS cutoff for each file **
        N_events[i] = np.min([N_events[i], args['MAXEVENTS']])

    print(f'Number of events per file: {N_events} (MAXEVENTS = {args["MAXEVENTS"]})')


    # =========================================================================
    # Load data for each re-weight PDFs

    pdf,X,Y,ids = compute_reweight(root_files=root_files, N_events=N_events, args=args)

    # =========================================================================
    ### Initialize all models
    model     = {}
    device    = {}
    optimizer = {}
    scheduler = {}
    param     = {}
    for i in range(len(args['active_models'])):

        ID        = args['active_models'][i]
        param[ID] = args[f'{ID}_param']

        if param[ID]['train'] == 'torch_graph':

            print(f'Training <{ID}> | {param[ID]} \n')

            # If not zero, then force the same value for every model
            if args['batch_train_param']['local_epochs'] != 0:
                param[ID]['epochs'] = int(args['batch_train_param']['local_epochs'])

            model[ID], device[ID], optimizer[ID], scheduler[ID] = \
                get_model(X=X, Y=Y, ids=ids, weights=None, features=features, args=args, param=param[ID])
    # ----------------------------------------------------------

    visited    = False
    N_epochs   = args['batch_train_param']['epochs']
    block_size = args['batch_train_param']['blocksize']

    ### Over each epoch
    for epoch in range(N_epochs):

        prints.printbar('=')
        cprint(__name__ + f".epoch {epoch+1} / {N_epochs} \n", 'yellow')

        ### Over each file
        for f in range(len(root_files)):

            prints.printbar('=')
            cprint(__name__ + f'.file {f+1} / {len(root_files)} \n', 'yellow')

            # ------------------------------------------------------------
            N_blocks   = int(np.ceil(N_events[f] / args['batch_train_param']['blocksize']))
            block_ind  = aux.split_start_end(range(N_events[f]), N_blocks)
            # ------------------------------------------------------------

            ### Over blocks of data from this file
            for block in range(N_blocks):
                
                entry_start = block_ind[block][0]
                entry_stop  = block_ind[block][-1]

                prints.printbar('=')
                cprint(__name__ + f'.block {block+1} / {N_blocks} \n', 'yellow')

                # =========================================================================
                # LOAD DATA
                if (N_blocks > 1) or (len(root_files) > 1) or (len(root_files) == 1 and N_blocks == 1 and visited == False):

                    visited = True # For the special case

                    X,Y,ids       = common.load_root_file(root_files[f], entry_start=entry_start, max_num_elements=entry_stop, args=args, library='np')
                    trn, val, tst = io.split_data(X=X, Y=Y, frac=args['frac'], rngseed=args['rngseed'])

                    # =========================================================================
                    # COMPUTE RE-WEIGHTS
                    # Re-weighting variables

                    PT  = trn.x[:, ids.index('trk_pt')]
                    ETA = trn.x[:, ids.index('trk_eta')]

                    # Compute event-by-event weights
                    if args['reweight_param']['reference_class'] != -1:
                            
                        trn_weights = reweight.reweightcoeff2D(
                            X_A = PT, X_B = ETA, pdf = pdf, y = trn.y, N_class=N_class,
                            equal_frac       = args['reweight_param']['equal_frac'],
                            reference_class  = args['reweight_param']['reference_class'],
                            max_reg          = args['reweight_param']['max_reg'])
                    else:
                        # No re-weighting
                        weights_doublet = np.zeros((trn.x.shape[0], N_class))
                        for c in range(N_class):    
                            weights_doublet[trn.y == c, c] = 1
                        trn_weights = np.sum(weights_doublet, axis=1)
                    
                    # Compute the sum of weights per class for the output print
                    frac = np.zeros(N_class)
                    sums = np.zeros(N_class)
                    for c in range(N_class):
                        frac[c] = np.sum(trn.y == c)
                        sums[c] = np.sum(trn_weights[trn.y == c])
                    
                    print(__name__ + f'.compute_reweights: sum(Y==c): {frac}')
                    print(__name__ + f'.compute_reweights: sum(trn_weights[Y==c]): {sums}')
                    print(__name__ + f'.compute_reweights: [done]\n')

                    # =========================================================================
                    ### Parse data into graphs

                    gdata = {}
                    gdata['trn'] = graphio.parse_graph_data(X=trn.x, Y=trn.y, ids=ids, weights=trn_weights,
                        features=features, global_on=args['graph_param']['global_on'], coord=args['graph_param']['coord'])

                    gdata['val'] = graphio.parse_graph_data(X=val.x, Y=val.y, ids=ids, weights=None,
                        features=features, global_on=args['graph_param']['global_on'], coord=args['graph_param']['coord'])
                
                # =========================================================================
                ### Train all model over this block of data
                for ID in model.keys():
                    cprint(__name__ + f' Training model <{ID}>', 'green')

                    train_loader = torch_geometric.loader.DataLoader(gdata['trn'], batch_size=param[ID]['opt_param']['batch_size'], shuffle=True)
                    test_loader  = torch_geometric.loader.DataLoader(gdata['val'], batch_size=512, shuffle=False)

                    # Local epoch loop
                    for local_epoch in range(param[ID]['epochs']):

                        loss                       = deep.graph.train(model=model[ID], loader=train_loader, optimizer=optimizer[ID], device=device[ID], param=param[ID]['opt_param'])
                        validate_acc, validate_AUC = deep.graph.test( model=model[ID], loader=test_loader,  optimizer=optimizer[ID], device=device[ID])
                        scheduler[ID].step()
                        
                        print(f"[epoch: {epoch+1:03d}/{N_epochs:03d}, block {block+1:03d}/{N_blocks:03d}, local epoch: {local_epoch+1:03d}/{param[ID]['epochs']:03d}] "
                            f"train loss: {loss:.4f} | validate: {validate_acc:.4f} (acc), {validate_AUC:.4f} (AUC) | learning_rate = {scheduler[ID].get_last_lr()}")
                        
                    ## Save
                    args["modeldir"] = aux.makedir(f'./checkpoint/eid/{args["config"]}/')
                    checkpoint = {'model': model[ID], 'state_dict': model[ID].state_dict()}
                    torch.save(checkpoint, args['modeldir'] + f'/{param[ID]["label"]}_checkpoint' + '.pth')

    print(__name__ + f' [Done!]')

        
if __name__ == '__main__' :
    main()
