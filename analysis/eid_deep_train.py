# Electron ID [DEEP TRAINING] steering code
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk


# icenet system paths
import _icepaths_

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
from icenet.tools import plots
from icenet.tools import prints

# deep learning
from icenet.deep  import train
import icenet.deep as deep


# iceid
from iceid import common
from iceid import graphio


def get_model(X, Y, VARS, features, args, param):

    # ---------------------------------------------------------------
    # Read test graph data to get dimensions

    gdata = {}
    gdata['trn'] = graphio.parse_graph_data(X=X[0:100], Y=Y[0:100], VARS=VARS, 
        features=features, global_on=args['graph_param']['global_on'], coord=args['graph_param']['coord'])

    # =========================================================================
    # INITIALIZE GRAPH MODEL

    num_classes         = 2
    num_node_features   = gdata['trn'][0].x.size(-1)
    num_edge_features   = gdata['trn'][0].edge_attr.size(-1)
    num_global_features = len(gdata['trn'][0].u)
    
    conv_type = param['conv_type']
    netparam = {
        'C' :           num_classes,
        'D' :           num_node_features,
        'E' :           num_edge_features,
        'G' :           num_global_features,
        'conv_aggr'  :  param['conv_aggr'],
        'global_pool':  param['global_pool'],
        'task'       :  'graph'
    }

    if   conv_type == 'GAT':
        model = deep.graph.GATNet(**netparam)
    elif conv_type == 'DEC':
        model = deep.graph.DECNet(**netparam)
    elif conv_type == 'EC':
        model = deep.graph.ECNet(**netparam)
    elif conv_type == 'SUP':
        model = deep.graph.SUPNet(**netparam)
    elif conv_type == 'SG':
        model = deep.graph.SGNet(**netparam)
    elif conv_type == 'SAGE':
        model = deep.graph.SAGENet(**netparam)
    elif conv_type == 'NN':
        model = deep.graph.NNNet(**netparam)
    elif conv_type == 'GINE':
        model = deep.graph.GINENet(**netparam)
    elif conv_type == 'spline':
        model = deep.graph.SplineNet(**netparam)
    else:
        raise Except(name__ + f'.graph_train: Unknown network convolution model "conv_type" = {conv_type}')

    # CPU or GPU
    model, device = deep.dopt.model_to_cuda(model=model, device_type=param['device'])

    # Count the number of parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    cprint(__name__ + f'.graph_train: Number of free parameters = {params}', 'yellow')
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=param['learning_rate'], weight_decay=param['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=param['step_size'], gamma=param['gamma'])

    return model, device, optimizer, scheduler


# Main function
#
def main():

    ### Get input
    args, cli, features = common.read_config()

    ### Load data (ONLY ONE FILE SUPPORTED FOR NOW)
    root_path = cli.datapath + '/output_' + str(cli.datasets[0]) + '.root'

    file   = uproot.open(root_path)
    EVENTS = file["ntuplizer"]["tree"].numentries
    file.close()
    
    
    # =========================================================================
    # LOAD DATA
    X,Y,VARS = common.load_root_file_new(root_path, entrystart=0, entrystop=args['reweight_param']['maxevents'], args=args)


    ### Compute differential re-weighting 2D-PDFs
    PT           = X[:, VARS.index('trk_pt')]
    ETA          = X[:, VARS.index('trk_eta')]

    bins_pt      = args['reweight_param']['bins_pt']
    bins_eta     = args['reweight_param']['bins_eta']
    pt_binedges  = np.linspace(bins_pt[0], bins_pt[1], bins_pt[2])
    eta_binedges = np.linspace(bins_eta[0], bins_eta[1], bins_eta[2])
    
    print(__name__ + f".compute_reweights: reference_class: <{args['reweight_param']['reference_class']}>")

    ### Compute 2D-pdfs for each class
    N_class = 2
    pdf     = {}
    for c in range(N_class):
        pdf[c] = aux.pdf_2D_hist(X_A=PT[Y==c], X_B=ETA[Y==c], binedges_A=pt_binedges, binedges_B=eta_binedges)

    pdf['binedges_A'] = pt_binedges
    pdf['binedges_B'] = eta_binedges

    # ----------------------------------------------------------
    ### Initialize all models
    model     = {}
    device    = {}
    optimizer = {}
    scheduler = {}
    param     = {}
    for i in range(len(args['active_models'])):

        ID        = args['active_models'][i]
        param[ID] = args[f'{ID}_param']
        print(f'Training <{ID}> | {param[ID]} \n')

        model[ID], device[ID], optimizer[ID], scheduler[ID] = \
            get_model(X=X,Y=Y,VARS=VARS,features=features,args=args,param=param[ID])
    # ----------------------------------------------------------

    ### EPOCH LOOP HERE
    block_size = args['blocksize']
    N_blocks   = int(np.ceil(EVENTS / args['blocksize']))
    block_ind  = aux.split_start_end(range(EVENTS), N_blocks)


    # Over blocks of data
    for block in range(N_blocks):
                    
        entrystart = block_ind[block][0]
        entrystop  = block_ind[block][-1]

        print(__name__ + f'.block {block+1} / {N_blocks} \n\n\n\n')

        # =========================================================================
        # LOAD DATA
        X,Y,VARS      = common.load_root_file_new(root_path, entrystart=entrystart, entrystop=entrystop, args=args)
        trn, val, tst = io.split_data(X=X, Y=Y, frac=args['frac'], rngseed=args['rngseed'])

        # =========================================================================
        # COMPUTE RE-WEIGHTS
        # Re-weighting variables

        PT  = trn.x[:, VARS.index('trk_pt')]
        ETA = trn.x[:, VARS.index('trk_eta')]

        # Compute event-by-event weights
        if args['reweight_param']['reference_class'] != -1:
                
            trn_weights = aux.reweightcoeff2D(
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
        gdata['trn'] = graphio.parse_graph_data(X=trn.x, Y=trn.y, VARS=VARS, 
            features=features, global_on=args['graph_param']['global_on'], coord=args['graph_param']['coord'])
        gdata['val'] = graphio.parse_graph_data(X=val.x, Y=val.y, VARS=VARS,
            features=features, global_on=args['graph_param']['global_on'], coord=args['graph_param']['coord'])

        # =========================================================================
        ### Train all model over this block of data
        for ID in model.keys():
            print(__name__ + f' Training model <{ID}>')

            train_loader = torch_geometric.data.DataLoader(gdata['trn'], batch_size=param[ID]['batch_size'], shuffle=True)
            test_loader  = torch_geometric.data.DataLoader(gdata['val'], batch_size=512, shuffle=False)

            for epoch in range(param[ID]['epochs']):

                loss                       = deep.graph.train(model=model[ID], loader=train_loader, optimizer=optimizer[ID], device=device[ID])
                validate_acc, validate_AUC = deep.graph.test( model=model[ID], loader=test_loader,  optimizer=optimizer[ID], device=device[ID])
                scheduler[ID].step()

                print(f'block {block+1:03d} [epoch: {epoch:03d}] train loss: {loss:.4f} | validate: {validate_acc:.4f} (acc), {validate_AUC:.4f} (AUC)')
                    
            ## Save
            args["modeldir"] = f'./checkpoint/eid/{args["config"]}/'; os.makedirs(args["modeldir"], exist_ok = True)
            checkpoint = {'model': model[ID], 'state_dict': model[ID].state_dict()}
            torch.save(checkpoint, args['modeldir'] + f'/{param[ID]["label"]}_checkpoint' + '.pth')


if __name__ == '__main__' :

   main()

