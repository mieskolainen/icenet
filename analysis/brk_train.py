# B/RK analysis [TRAINING] code
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

# icenet system paths
import sys
sys.path.append(".")

# Configure plotting backend
import matplotlib
matplotlib.use('Agg')

import sys
import os
import torch
import uproot
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

import xgboost
import sklearn


# icenet

from icenet.tools import process
from icenet.tools import io
from icenet.tools import aux
from icenet.tools import plots
from icenet.tools import prints

from icenet.algo  import flr
from icenet.deep  import optimize
from icenet.deep  import bnaf
from icenet.deep  import dbnf
from icenet.deep  import deps
from icenet.deep  import maxo
from icenet.deep  import train

from icenet.optim import adam
from icenet.optim import adamax
from icenet.optim import scheduler


# icebrk
from icebrk import common
from icebrk import loop
from icebrk import features
from iceplot import iceplot


# Main function
#
def main() :
        
    cli, cli_dict = process.read_cli()
    runmode  = cli_dict['runmode']
    
    args,cli = process.read_config(config_path='configs/brk', runmode=runmode)
    iodir    = aux.makedir(f'output/{args["rootname"]}/{cli.tag}/')
    paths    = io.glob_expand_files(datasets=cli.datasets, datapath=cli.datapath)


    VARS = features.generate_feature_names(args['MAXT3'])

    args['modeldir'] = aux.makedir(f'checkpoint/{args["rootname"]}/{args["config"]}')
    plotdir = aux.makedir(f'figs/{args["rootname"]}/{args["config"]}/train/')
    

    # ========================================================================
    ### Event loop
    
    ## Binary matrix
    BMAT = aux.generatebinary(args['MAXT3'], args['MAXN'])
    print(f'POWERSET [NCLASS = {BMAT.shape[0]}]:')
    prints.print_colored_matrix(BMAT)

    output = loop.process(paths=paths, 
        isMC=True, MAXT3=args['MAXT3'], MAXN=args['MAXN'], WNORM=args['WNORM'],
        maxevents=args['maxevents'], VERBOSE=args['VERBOSE'], BMAT=BMAT, SUPERSETS=args['SUPERSETS'], hd5dir=None, outputXY=True)

    X = output['X']
    Y = output['Y']

    print(__name__ + ': Variables before normalization:')
    prints.print_variables(X, VARS)    
    # =======================================================================
    
    targetdir = aux.makedir(f'figs/{args["rootname"]}/{args["config"]}/train/')
    fig,ax    = plots.plot_correlations(X=X, ids=VARS, targetdir=targetdir)    
    
    
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
        pickle.dump([X_mu, X_std], open(args['modeldir'] + '/zscore.dat', 'wb'))


    ### Weights: CURRENTLY UNIT WEIGHTS
    # (UPDATE THIS TO IMPLEMENT E.G. PILE-UP DEPENDENT WEIGHTS)!
    trn_weights = None
    val_weights = None

    # Number of classes (powerset dimension)
    C = BMAT.shape[0]
    
    
    # ====================================================================
    # Training
    print('<< TRAINING XGBOOST >>')
    label = 'xgb'
        
    if label in args['active_models']:

        X_train, X_eval, Y_train, Y_eval = \
            sklearn.model_selection.train_test_split(X, Y_i, test_size=1-args['frac'], random_state=args['rngseed'])

        if args['models'][label]['model_param']['tree_method'] == 'auto':
            args['models'][label]['model_param'].update({'tree_method' : 'gpu_hist' if torch.cuda.is_available() else 'hist'})
        
        # Update parameters
        args['models'][label]['model_param'].update({'num_class': C})
        
        dtrain = xgboost.DMatrix(data = X_train, label = Y_train) #, weight = trn_weights)
        deval  = xgboost.DMatrix(data = X_eval , label = Y_eval)

        evallist  = [(dtrain, 'train'), (deval, 'eval')]
        results   = dict()

        num_boost_round = args['models'][label]['model_param']['num_boost_round']
        del args['models'][label]['model_param']['num_boost_round']

        xgb_model = xgboost.train(params=args['models'][label]['model_param'], num_boost_round = num_boost_round,
            dtrain=dtrain, evals=evallist, evals_result=results, verbose_eval=True)
        
        ## Save
        pickle.dump(xgb_model, open(args['modeldir'] + f"/{args['models'][label]['label']}_model.dat", 'wb'))
        
        # ================================================================
        losses   = results['train']['mlogloss']
        trn_aucs = results['train']['auc']
        val_aucs = results['eval']['auc']
        
        # Plot evolution
        plotdir  = aux.makedir(f'figs/{args["rootname"]}/{args["config"]}/train/')
        fig,ax   = plots.plot_train_evolution_multi({'train': losses}, trn_aucs, val_aucs, label)
        plt.savefig(f'{plotdir}/{label}_evolution.pdf', bbox_inches='tight'); plt.close()

        # Plot feature importance
        fig,ax = plots.plot_xgb_importance(model=xgb_model, tick_label=VARS, label='xgb')
        targetdir = aux.makedir(f'figs/{args["rootname"]}/{args["config"]}/train')
        plt.savefig(f'{targetdir}/{label}_importance.pdf', bbox_inches='tight'); plt.close()
        # ================================================================
    

    # ------------------------------------------------------------------------
    # PyTorch based training

    def torch_split(x, y, test_size, random_state):

        X_trn, X_val, Y_trn, Y_val = \
            sklearn.model_selection.train_test_split(x,y, test_size=test_size, random_state=random_state)
        
        X_trn = torch.from_numpy(X_trn).type(torch.FloatTensor)
        X_val = torch.from_numpy(X_val).type(torch.FloatTensor)
        
        Y_trn = torch.from_numpy(Y_trn).type(torch.LongTensor) # class targets as ints
        Y_val = torch.from_numpy(Y_val).type(torch.LongTensor)

        return X_trn, X_val, Y_trn, Y_val
    

    # ====================================================================
    print('<< TRAINING DEEP MAXOUT >>')
    
    label = 'maxo'
    if label in args['active_models']:
        
        X_trn, X_val, Y_trn, Y_val = torch_split(x=X, y=Y_i, test_size=1 - args['frac'], random_state=args['rngseed'])
        args['num_classes'] = C

        model, train_loader, test_loader = \
            train.torch_construct(X_trn=X_trn, Y_trn=Y_trn, X_val=X_val, Y_val=Y_val, X_trn_2D=None, X_val_2D=None, \
             trn_weights=trn_weights, val_weights=val_weights, param=args['models'][label], args=args)

        model = train.torch_loop(model=model, train_loader=train_loader, test_loader=test_loader, \
                    args=args, param=args['models'][label])


    # ====================================================================
    print('<< TRAINING DEEP SETS >>')

    label = 'deps'
    if label in args['active_models']:

        M    = args['MAXT3']                      # Number of triplets per event
        D    = features.getdimension()            # Triplet feature vector dimension
        X_DS = aux.longvec2matrix(X=X, M=M, D=D)
        
        X_trn, X_val, Y_trn, Y_val = torch_split(x=X_DS, y=Y_i, test_size=1 - args['frac'], random_state=args['rngseed'])
        args['num_classes'] = C

        model, train_loader, test_loader = \
            train.torch_construct(X_trn=X_trn, Y_trn=Y_trn, X_val=X_val, Y_val=Y_val, X_trn_2D=None, X_val_2D=None, \
             trn_weights=trn_weights, val_weights=val_weights, param=args['models'][label], args=args)

        model = train.torch_loop(model=model, train_loader=train_loader, test_loader=test_loader, \
                    args=args, param=args['models'][label])

    print('\n' + __name__+ ' DONE')

    
if __name__ == '__main__' :
    main()
