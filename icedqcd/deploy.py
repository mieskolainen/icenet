# Deploy MVA-model to data (or MC) files (work in progress ...)
#
# m.mieskolainen@imperial.ac.uk, 2022


import torch
import torch_geometric
import numpy as np
import awkward as ak
import pickle


# GLOBALS
from configs.dqcd.mvavars import *

from icedqcd import common
from icenet.tools import iceroot, io
from icenet.deep import predict


def process_data(args, path, filename):

    # ------------------
    # Phase 1: Read ROOT-file to awkward-format

    VARS  = []
    VARS += TRIGGER_VARS
    VARS += MVA_SCALAR_VARS
    VARS += MVA_JAGGED_VARS
    VARS += MVA_PF_VARS

    rootfile  = path + filename

    param = {
        'rootfile'    : rootfile,
        'tree'        : 'Events',
        'entry_start' : 0,
        'entry_stop'  : None,
        'maxevents'   : int(1E9),
        'ids'         : LOAD_VARS,
        'library'     : 'ak'
    }
    
    X_uncut, ids = iceroot.load_tree(**param)
    
    # ------------------
    # Phase 2: Apply selections (no selections applied --> event numbers kept intact)

    X = X_uncut

    #N_before     = len(X_uncut)
    #X,ids,stats  = common.process_root(X=X_uncut, ids=ids, isMC=False, args=args)
    #N_after      = len(X)
    #eff_acc      = N_after / N_before
    #print(__name__ + f' efficiency x acceptance = {eff_acc:0.6f}')
    
    # ------------------
    # Phase 3: Convert to icenet dataformat

    Y        = ak.Array(np.zeros(len(X))) # Dummy
    W        = ak.Array(np.zeros(len(X))) # Dummy
    data     = common.splitfactor(x=X, y=Y, w=W, ids=ids, args=args)
    
    # ------------------
    # Phase 4: Apply MVA-models (so far only to events which pass the trigger & pre-cuts)

    for i in range(len(args['active_models'])):
        
        ID    = args['active_models'][i]
        param = args['models'][ID]
        
        if param['predict'] == 'xgb':

            print(f'Evaluating <{ID}> \n')    
            
            func_predict = get_predictor(args=args, param=param)
            X            = data['data'].x

            ### Variable normalization
            if   args['varnorm'] == 'zscore':

                print('\nZ-score normalizing variables ...')
                X_mu, X_std = pickle.load(open(args["modeldir"] + '/zscore.pkl', 'rb'))
                X = io.apply_zscore(X, X_mu, X_std)
            
            scores = func_predict(X)

            # Simple x-check
            if len(X) != len(scores):
                raise Exception(__name__ + f'.process_data: Error: length of input != length of output')

            # ------------------
            # Phase 5: Write MVA-scores out
            outputfile = './output/' + filename.replace('/', '--').replace('.root', '.xgbscore')
            print(__name__ + f'.process_data: Saving output to "{outputfile}"')
            np.savetxt(outputfile, scores)

        else:
            continue

        #if param['predict'] == 'torch_graph':
        #    scores = func_predict(data['graph'])


def get_predictor(args, param):    

    if   param['predict'] == 'xgb':
        func_predict = predict.pred_xgb(args=args, param=param)

    elif param['predict'] == 'xgb_logistic':
        func_predict = predict.pred_xgb_logistic(args=args, param=param)

    elif param['predict'] == 'torch_vector':
        func_predict = predict.pred_torch_generic(args=args, param=param)

    elif param['predict'] == 'torch_scalar':
        func_predict = predict.pred_torch_scalar(args=args, param=param)

    elif param['predict'] == 'torch_flow':
        func_predict = predict.pred_flow(args=args, param=param, n_dims=X_ptr.shape[1])

    elif   param['predict'] == 'torch_graph':
        func_predict = predict.pred_torch_graph(args=args, param=param)

    elif param['predict'] == 'graph_xgb':
        func_predict = predict.pred_graph_xgb(args=args, param=param)
    
    elif param['predict'] == 'torch_deps':
        func_predict = predict.pred_torch_generic(args=args, param=param)
    
    elif param['predict'] == 'torch_image':
        func_predict = predict.pred_torch_generic(args=args, param=param)
        
    elif param['predict'] == 'torch_image_vector':
        func_predict = predict.pred_torch_generic(args=args, param=param)

    elif param['predict'] == 'flr':
        func_predict = predict.pred_flr(args=args, param=param)
        
    #elif param['predict'] == 'xtx':
    # ...   
    #
    
    elif param['predict'] == 'exp':
        func_predict = predict.pred_exp(args=args, param=param)
    
    elif param['predict'] == 'cut':
        func_predict = predict.pred_cut(args=args, param=param)
    
    elif param['predict'] == 'cutset':
        func_predict = predict.pred_cutset(args=args, param=param)
    
    else:
        raise Exception(__name__ + f'.Unknown param["predict"] = {param["predict"]} for ID = {ID}')

    return func_predict



def apply_models(data=None, args=None):
    #
    #Evaluate ML/AI models.
    #
    #Args:
    #    Different datatype objects (see the code)
    #

    try:
        ### Tensor variable normalization
        if data['data_tensor'] is not None and (args['varnorm_tensor'] == 'zscore'):

            print('\nZ-score normalizing tensor variables ...')
            X_mu_tensor, X_std_tensor = pickle.load(open(args["modeldir"] + '/zscore_tensor.pkl', 'rb'))
            X_2D = io.apply_zscore_tensor(X_2D, X_mu_tensor, X_std_tensor)
        
        ### Variable normalization
        if   args['varnorm'] == 'zscore':

            print('\nZ-score normalizing variables ...')
            X_mu, X_std = pickle.load(open(args["modeldir"] + '/zscore.pkl', 'rb'))
            X = io.apply_zscore(X, X_mu, X_std)

        elif args['varnorm'] == 'madscore':

            print('\nMAD-score normalizing variables ...')
            X_m, X_mad = pickle.load(open(args["modeldir"] + '/madscore.pkl', 'rb'))
            X = io.apply_madscore(X, X_m, X_mad)

    except:
        cprint('\n' + __name__ + f' WARNING: {sys.exc_info()[0]} in normalization. Continue without! \n', 'red')
    
    # --------------------------------------------------------------------
    # For pytorch based
    if X is not None:
        X_ptr      = torch.from_numpy(X).type(torch.FloatTensor)

    if X_2D is not None:
        X_2D_ptr   = torch.from_numpy(X_2D).type(torch.FloatTensor)
        
    if X_deps is not None:
        X_deps_ptr = torch.from_numpy(X_deps).type(torch.FloatTensor)

