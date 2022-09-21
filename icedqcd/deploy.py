# Deploy MVA-model

import torch
import numpy as np

from icedqcd.common import *
from icenet.tools import iceroot


def deploy_model():

    # ------------------
    # Phase 1: Read ROOT-file to awkward-format

    param = {
        rootfile    = ''
        tree        = 'Events'
        entry_start = 0
        entry_stop  = None
        maxevents   = None
        ids         = LOAD_IDS
        library     = 'ak'
    }
    
    X_uncut, ids = iceroot.load_tree(**param)
    N_before = len(X_uncut)
    
    # ------------------
    # Phase 2: Apply selections
    X,ids,stats = process_func(X=X_uncut, ids=ids, **param)
    N_after     = len(X)
    eff_acc     = N_after / N_before
    print(__name__ + f' efficiency x acceptance = {eff_acc:0.6f}')
    
    # ------------------
    # Phase 3: Convert to icenet-format
    data     = splitfactor(x, y, w, ids, args):
    
    # ------------------
    # Phase 4: Apply MVA-models (only to events which pass the trigger & cuts)
    data     = apply_models(data=data, args=args)
    
    # ------------------
    # Phase 5: Save MVA-scores


    # ------------------
    # Phase 5: Write MVA-scores out
    
    


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
        

    # ====================================================================
    # **  MAIN LOOP OVER MODELS **
    #

    for i in range(len(args['active_models'])):
        
        ID    = args['active_models'][i]
        param = args['models'][ID]
        print(f'Evaluating <{ID}> | {param} \n')
        
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

            # Geometric type -> need to use batch loader, get each graph, node or edge prediction
            import torch_geometric
            loader  = torch_geometric.loader.DataLoader(X_graph, batch_size=len(X_graph), shuffle=False)
            for batch in loader: # Only one big batch
                plot_XYZ_wrap(func_predict = func_predict, x_input=X_graph, y=batch.to('cpu').y.detach().cpu().numpy(), **inputs)
        
        elif param['predict'] == 'graph_xgb':
            func_predict = predict.pred_graph_xgb(args=args, param=param)
        
        elif param['predict'] == 'torch_deps':
            func_predict = predict.pred_torch_generic(args=args, param=param)
        
        elif param['predict'] == 'torch_image':
            func_predict = predict.pred_torch_generic(args=args, param=param)
            
        elif param['predict'] == 'torch_image_vector':
            func_predict = predict.pred_torch_generic(args=args, param=param)

            X_dual      = {}
            X_dual['x'] = X_2D_ptr # image tensors
            X_dual['u'] = X_ptr    # global features
            plot_XYZ_wrap(func_predict = func_predict, x_input = X_dual, y=y, **inputs)
            
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

    return
"""
