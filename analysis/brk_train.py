# B/RK analysis [TRAINING] code
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

# icenet system paths
import _icepaths_

import sys
import os
import torch
import uproot
import uproot_methods
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

import xgboost
from sklearn.model_selection import train_test_split


# icenet
from icenet.tools import io
from icenet.tools import aux
from icenet.tools import plots
from icenet.tools import prints

from icenet.algo  import flr
from icenet.deep  import dopt
from icenet.deep  import bnaf
from icenet.deep  import dbnf
from icenet.deep  import deps
from icenet.deep  import maxo

from icenet.optim import adam
from icenet.optim import adamax
from icenet.optim import scheduler


# icebrk
from icebrk import common
from icebrk import loop
from icebrk import features



# Main function
#
def main() :

    ### Get input
    paths, args, cli, iodir = common.init()
    VARS = features.generate_feature_names(args['MAXT3'])
    modeldir = f'./checkpoint/brk/{args["config"]}/'; os.makedirs(modeldir, exist_ok = True)
    plotdir  = f'./figs/brk/{args["config"]}/train/'; os.makedirs(plotdir,  exist_ok = True)

    # ========================================================================
    ### Event loop
    
    ## Binary matrix
    BMAT = aux.generatebinary(args['MAXT3'], args['MAXN'])
    print(f'POWERSET [NCLASS = {BMAT.shape[0]}]:')
    prints.print_colored_matrix(BMAT)

    output = loop.process(paths=paths, 
        isMC=True, MAXT3=args['MAXT3'], MAXN=args['MAXN'], WNORM=args['WNORM'],
        MAXEVENTS=args['MAXEVENTS'], VERBOSE=args['VERBOSE'], BMAT=BMAT, SUPERSETS=args['SUPERSETS'], hd5dir=None, outputXY=True)

    X = output['X']
    Y = output['Y']

    print(__name__ + ': Variables before normalization:')
    prints.print_variables(X, VARS)    
    # =======================================================================
    
    
    # ------------------------------------------------------------------------
    # One-hot to a powerset class index

    print('Output Y as a binary vector:')
    print(Y)

    print('Output Y as a powerset index:')
    Y_i = aux.binvec2powersetindex(Y, BMAT)
    print(Y_i)


    ### Truncate outliers (component by component) from the training set
    if args['outlier_param']['algo'] == 'truncate' :
        for j in range(X.shape[1]):

            minval = np.percentile(X[:,j], args['outlier_param']['qmin'])
            maxval = np.percentile(X[:,j], args['outlier_param']['qmax'])

            X[X[:,j] < minval,j] = minval
            X[X[:,j] > maxval,j] = maxval

    """
    # Training data imputation
    # NOT IMPLEMENTED
    """

    ### Variable normalization
    if args['varnorm'] == 'zscore' :

        print('\nz-score normalizing variables ...')

        X_mu, X_std = io.calc_zscore(X)
        X  = io.apply_zscore(X, X_mu, X_std)

        VARS = features.generate_feature_names(args['MAXT3'])

        print(__name__ + ': Variables after normalization:')
        prints.print_variables(X, VARS)

        # Save it for the evaluation
        pickle.dump([X_mu, X_std], open(modeldir + '/zscore.dat', 'wb'))


    ### Weights: CURRENTLY UNIT WEIGHTS
    # (UPDATE THIS TO IMPLEMENT E.G. PILE-UP DEPENDENT WEIGHTS)!
    trn_weights = np.ones(X.shape[0])

    # Number of classes (powerset dimension)
    C = BMAT.shape[0]


    # ====================================================================
    # Training
    print('<< TRAINING XGBOOST >>')
    
    if args['xgb_param']['active']:

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y_i, test_size=0.1, random_state=42)

        if args['xgb_param']['tree_method'] == 'auto':
            args['xgb_param'].update({'tree_method' : 'gpu_hist' if torch.cuda.is_available() else 'hist'})
        
        # Update parameters
        args['xgb_param'].update({'num_class': C})
        
        dtrain = xgboost.DMatrix(data = X_train, label = Y_train) #, weight = trn_weights)
        dtest  = xgboost.DMatrix(data = X_test , label = Y_test)

        evallist  = [(dtrain, 'train'), (dtest, 'eval')]
        results   = dict()
        xgb_model = xgboost.train(params=args['xgb_param'], num_boost_round = args['xgb_param']['num_boost_round'],
            dtrain=dtrain, evals=evallist, evals_result=results, verbose_eval=True)

        ## Save
        pickle.dump(xgb_model, open(modeldir + '/XGB_model.dat', 'wb'))


    # ------------------------------------------------------------------------
    # PyTorch based training
    
    # Target outputs [** NOW SAME -- UPDATE THIS **]
    Y_trn = torch.from_numpy(Y_i).type(torch.LongTensor)
    Y_val = torch.from_numpy(Y_i).type(torch.LongTensor)
    
    
    # ====================================================================
    print('<< TRAINING DEEP MAXOUT >>')
    
    if args['dmax_param']['active']:
        
        label = args['dmax_param']['label']
        
        # Input [** NOW SAME -- UPDATE THIS **]
        X_trn = torch.from_numpy(X).type(torch.FloatTensor)
        X_val = torch.from_numpy(X).type(torch.FloatTensor)

        print(f'\nTraining {label} classifier ...')
        dmax_model = maxo.MAXOUT(D = X_trn.shape[1], C = C, num_units=args['dmax_param']['num_units'], neurons=args['dmax_param']['neurons'], dropout=args['dmax_param']['dropout'])
        dmax_model, losses, trn_aucs, val_aucs = dopt.train(model = dmax_model, X_trn = X_trn, Y_trn = Y_trn, X_val = X_val, Y_val = Y_val,
            trn_weights = trn_weights, param = args['dmax_param'])
        
        # Plot evolution
        fig,ax = plots.plot_train_evolution(losses, trn_aucs, val_aucs, label)
        plt.savefig(f'{plotdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()
        
        ## Save
        checkpoint = {'model': dmax_model, 'state_dict': dmax_model.state_dict()}
        torch.save(checkpoint, modeldir + '/DMAX_checkpoint.pth')
    
    # ====================================================================
    print('<< TRAINING DEEP SETS >>')

    if args['deps_param']['active']:

        M = args['MAXT3']           # Number of triplets per event
        D = features.getdimension() # Triplet feature vector dimension
        C = BMAT.shape[0]           # Number of classes

        X_DS = aux.longvec2matrix(X, M, D)
        
        # [NOW SAME -- UPDATE THIS!]
        X_trn = torch.from_numpy(X_DS).type(torch.FloatTensor)
        X_val = torch.from_numpy(X_DS).type(torch.FloatTensor)
        
        # ----------------------------------------------------------------
        
        label = args['deps_param']['label']
        
        print(f'\nTraining {label} classifier ...')
        deps_model = deps.DEPS(D = D, z_dim = args['deps_param']['z_dim'], C = C)
        deps_model, losses, trn_aucs, val_aucs = dopt.train(model = deps_model, X_trn = X_trn, Y_trn = Y_trn, X_val = X_val, Y_val = Y_val, 
            trn_weights = trn_weights, param = args['deps_param'])
        
        # Plot evolution
        fig,ax = plots.plot_train_evolution(losses, trn_aucs, val_aucs, label)
        plt.savefig(f'{plotdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()
        
        ## Save
        checkpoint = {'model': deps_model, 'state_dict': deps_model.state_dict()}
        torch.save(checkpoint, modeldir + '/DEPS_checkpoint.pth')
        
        
    print('\n' + __name__+ ' DONE')

    
if __name__ == '__main__' :
    
    main()
