# Deploy MVA-model to data (or MC) files (work in progress ...)
#
# m.mieskolainen@imperial.ac.uk, 2022


import os
import torch
import torch_geometric
import numpy as np
import awkward as ak
import pickle
import uproot


# GLOBALS
from configs.dqcd.mvavars import *

from icedqcd import common
from icenet.tools import iceroot, io, aux
from icenet.deep import predict


def process_data(args):

    ## ** Special YAML loader here **
    from libs.ccorp.ruamel.yaml.include import YAML
    yaml = YAML()
    yaml.allow_duplicate_keys = True
    cwd = os.getcwd()

    with open(f'{cwd}/configs/dqcd/{args["inputmap"]}', 'r') as f:
        try:
            inputmap = yaml.load(f)

            if 'includes' in inputmap:
                del inputmap['includes'] # ! crucial

        except yaml.YAMLError as exc:
            print(f'yaml.YAMLError: {exc}')
            exit()

    root_path = args['root_files']
    if type(root_path) is list:
        root_path = root_path[0] # Remove [] list

    # ------------------
    # Phase 1: Read ROOT-file to awkward-format

    VARS  = []
    VARS += TRIGGER_VARS
    VARS += MVA_SCALAR_VARS
    VARS += MVA_JAGGED_VARS
    VARS += MVA_PF_VARS

    for key in inputmap.keys():

        print(__name__ + f'.process_data: Processing "{key}"')

        # Get all files
        datasets  = inputmap[key]['path'] + '/' + inputmap[key]['files']
        rootfiles = io.glob_expand_files(datasets=datasets, datapath=root_path)
        
        # Loop over the files
        for k in range(len(rootfiles)):

            filename = rootfiles[k]
            param = {
                'rootfile'    : filename,
                'tree'        : 'Events',
                'entry_start' : None,
                'entry_stop'  : None,
                'maxevents'   : None,
                'ids'         : LOAD_VARS,
                'library'     : 'ak'
            }
            
            X_uncut, ids_uncut = iceroot.load_tree(**param)

            # -------------------------------------------------
            # Add conditional (theory param) variables
            model_param = {'ctau': 0.0, 'm': 0.0, 'xiO': 0.0, 'xiL': 0.0}

            if args['use_conditional']:

                print(__name__ + f'.process_data: Initializing conditional theory (model) parameters')
                for var in model_param.keys():
                    # Create new 'record' (column) to ak-array
                    col_name    = f'MODEL_{var}'
                    X_uncut[col_name] = model_param[var]

                ids_uncut = ak.fields(X_uncut)

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
            data     = common.splitfactor(x=X, y=Y, w=W, ids=ids_uncut, args=args, skip_graph=True)
            
            # ------------------
            # Phase 4: Apply MVA-models (so far only to events which pass the trigger & pre-cuts)

            for i in range(len(args['active_models'])):
                
                ID    = args['active_models'][i]
                param = args['models'][ID]
                
                if param['predict'] == 'xgb':

                    print(f'Evaluating MVA-model "{ID}" \n')

                    X,ids = aux.red(X=data['data'].x, ids=data['data'].ids, param=param)
                    func_predict = get_predictor(args=args, param=param, feature_names=ids)

                    ### Set the conditional variables
                    model_param = {'ctau': 0.0, 'm': 0.0, 'xiO': 0.0, 'xiL': 0.0}

                    """
                    ### Variable normalization
                    if   args['varnorm'] == 'zscore':

                        print(__name__ + f'.process_data: Z-score normalizing variables ...')
                        X_mu, X_std = pickle.load(open(args["modeldir"] + '/zscore.pkl', 'rb'))

                        print(__name__ + f'.process_data: X.shape = {X.shape} | X_mu.shape = {X_mu.shape} | X_std.shape = {X_std.shape}')
                        X = io.apply_zscore(X, X_mu, X_std)
                    """
                    scores = func_predict(X)
                    
                    # Simple x-check
                    if len(X_uncut) != len(scores):
                        raise Exception(__name__ + f'.process_data: Error: len(X_uncut) != len(scores)')

                    if len(X) != len(scores):
                        raise Exception(__name__ + f'.process_data: Error: len(X) != len(scores)')

                    # ------------------
                    # Phase 5: Write MVA-scores out

                    basepath   = f'{cwd}/output/deploy' + '/' + f"modeltag-[{args['modeltag']}]"
                    outpath    = aux.makedir(basepath + '/' + filename.rsplit('/', 1)[0])
                    outputfile = basepath + '/' + filename.replace('.root', '-icenet.root')

                    with uproot.recreate(outputfile, compression=None) as file:
                        print(__name__ + f'.process_data: Saving root output to "{outputfile}"')
                        file[f"tree1"] = {f"{ID}": scores}

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

