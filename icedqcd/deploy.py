# Deploy MVA-model to data (or MC) files (work in progress, not all models supported ...)
#
# m.mieskolainen@imperial.ac.uk, 2022

import matplotlib.pyplot as plt
import os
import torch
import torch_geometric
import numpy as np
import awkward as ak
import pickle
import uproot
import logging
import shap
import xgboost
import copy
import socket

from datetime import datetime
from termcolor import colored, cprint


# GLOBALS
from configs.dqcd.mvavars import *

from icedqcd import common
from icenet.tools import iceroot, io, aux, process
from icenet.deep import predict
from tqdm import tqdm


def generate_cartesian_param(ids):
    """
    Generate cartesian array for the theory model parameters

    Note. Keep the order m, ctau, xiO, xiL
    """

    values    = {'m':    np.round(np.array([2.0, 3.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]), 1),
                 'ctau': np.round(np.array([10, 25, 50, 75, 100, 250, 500]), 1),
                 'xiO':  np.round(np.array([1.0, 2.5]), 1),
                 'xiL':  np.round(np.array([1.0, 2.5]), 1)}

    CAX       = aux.cartesian_product(*[values['m'], values['ctau'], values['xiO'], values['xiL']])

    pindex    = np.zeros(4, dtype=int)
    pindex[0] = ids.index('MODEL_m')
    pindex[1] = ids.index('MODEL_ctau')
    pindex[2] = ids.index('MODEL_xiO')
    pindex[3] = ids.index('MODEL_xiL')

    return CAX, pindex

def f2s(value):
    """
    Convert floating point "1.5" to "1p5"
    """
    return str(np.round(value,1)).replace('.', 'p')

def zscore_normalization(X, args):
    """
    Z-score normalization
    """
    if   args['varnorm'] == 'zscore':                            
        print(__name__ + f'.process_data: Z-score normalizing variables ...')
        X_mu, X_std = pickle.load(open(args["modeldir"] + '/zscore.pkl', 'rb'))

        print(__name__ + f'.process_data: X.shape = {X.shape} | X_mu.shape = {X_mu.shape} | X_std.shape = {X_std.shape}')
        X = io.apply_zscore(X, X_mu, X_std)

    return X

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

    basepath = aux.makedir(f"{cwd}/output/dqcd/deploy/modeltag__{args['modeltag']}")

    nodestr  = (f"inputmap__{io.safetxt(args['inputmap'])}--hostname__{socket.gethostname()}--time__{datetime.now()}").replace(' ', '')
    logging.basicConfig(filename=f'{basepath}/deploy--{nodestr}.log', encoding='utf-8',
        level=logging.DEBUG, format='%(asctime)s | %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

    # Save to log-file
    logging.debug(__name__ + f'.process_data: {nodestr}')
    logging.debug('')
    logging.debug(f'{inputmap}')
    
    for key in inputmap.keys():

        print(__name__ + f'.process_data: Processing "{key}"')

        # Write to log-file
        logging.debug('-------------------------------------------------------')
        logging.debug(f'Process: {key}')

        # Get all files
        datasets  = inputmap[key]['path'] + '/' + inputmap[key]['files']
        rootfiles = io.glob_expand_files(datasets=datasets, datapath=root_path)

        # Loop over the files
        total_num_events = 0

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
            
            try:
                X_nocut, ids_nocut = iceroot.load_tree(**param)

                # Write to log-file
                logging.debug(f'{filename} | Number of events: {len(X_nocut)}')
                total_num_events += len(X_nocut)

            except:
                cprint(__name__ + f'.process_data: A fatal error in iceroot.load_tree with a file "{filename}"', 'red')
                
                # Write to log-file
                logging.debug(f'{filename} | A fatal error in iceroot.load_tree !')
                continue

            # -------------------------------------------------
            # Add conditional (theory param) variables
            model_param = {'m': 0.0, 'ctau': 0.0, 'xiO': 0.0, 'xiL': 0.0}

            if args['use_conditional']:

                print(__name__ + f'.process_data: Initializing conditional theory (model) parameters')
                for var in model_param.keys():
                    # Create new 'record' (column) to ak-array
                    col_name    = f'MODEL_{var}'
                    X_nocut[col_name] = model_param[var]

                ids_nocut = ak.fields(X_nocut)

            # ------------------
            # Phase 2: Apply pre-selections to get an event mask

            mask = common.process_root(X=X_nocut, args=args, return_mask=True)
            
            if len(X_nocut[mask]) == 0:
                X        = X_nocut
                logging.debug("No events left after pre-cuts -- scores will be -1 for all events")
                OK_event = False
            else:
                X        = X_nocut[mask]
                OK_event = True
            
            # ------------------
            # Phase 3: Convert to icenet dataformat

            Y        = ak.Array(np.zeros(len(X))) # Dummy [does not exist here]
            W        = ak.Array(np.ones(len(X)))  # Dummy [does not exist here]
            data     = common.splitfactor(x=X, y=Y, w=W, ids=ids_nocut, args=args, skip_graph=True)
            
            # ------------------
            # Phase 4: Apply MVA-models
            
            ALL_scores = {}

            for i in range(len(args['active_models'])):
                
                ID    = args['active_models'][i]
                param = args['models'][ID]
                
                if param['predict'] == 'xgb':

                    print(f'Evaluating MVA-model "{ID}" \n')

                    # Impute data
                    if args['imputation_param']['active']:
                        imputer = pickle.load(open(args["modeldir"] + f'/imputer.pkl', 'rb'))
                        data['data'], _  = process.impute_datasets(data=data['data'], features=None, args=args['imputation_param'], imputer=imputer)

                    ## Apply the input variable set reductor
                    X,ids = aux.red(X=data['data'].x, ids=data['data'].ids, param=param)

                    ## Get the MVA-model
                    func_predict, model = get_predictor(args=args, param=param, feature_names=ids)

                    ## Conditional model
                    if args['use_conditional']:

                        ## Get conditional parameters
                        CAX,pindex = generate_cartesian_param(ids=ids)

                        ## Run the MVA-model on all the theory model points in CAX array
                        scores = {}
                        for z in tqdm(range(len(CAX))):

                            # Set the new conditional model parameters to X
                            nval          = CAX[z,:]
                            ID_label = f'{ID}__m_{f2s(nval[0])}_ctau_{f2s(nval[1])}_xiO_{f2s(nval[2])}_xiL_{f2s(nval[3])}'

                            if OK_event:
                                XX            = copy.deepcopy(X)
                                XX[:, pindex] = nval  # Set new values

                                # Variable normalization
                                XX = zscore_normalization(X=XX, args=args)

                                # Predict
                                pred = func_predict(XX)
                                pred = aux.unmask(x=pred, mask=mask, default_value=-1)
                            else:
                                pred = (-1) * np.ones(len(mask)) # all -1

                            # Save
                            ALL_scores[io.rootsafe(ID_label)] = pred

                    else:

                        if OK_event:

                            # Variable normalization
                            XX = copy.deepcopy(X)
                            XX = zscore_normalization(X=XX, args=args)

                            # Predict
                            pred = func_predict(XX)
                            pred = aux.unmask(x=pred, mask=mask, default_value=-1)

                            # ----------------------------
                            # Import SHAP
                            
                            """
                            explainer   = shap.Explainer(model, feature_names=ids)
                            maxEvent    = 3
                            shap_values = explainer(XX[0:maxEvent,:])
                            
                            # Visualize the SHAP value explanation
                            for n in range(len(shap_values)):
                                shap.plots.waterfall(shap_values[i], max_display=30)
                                plt.savefig(f'{basepath}/waterfall_test_{key}_{n}.pdf', bbox_inches='tight')
                                plt.close()
                            """                            
                        else:
                            pred = (-1) * np.ones(len(mask)) # all -1

                        # Save
                        ALL_scores[io.rootsafe(ID)] = pred

                else:
                    # if param['predict'] == 'torch_graph': # Turned off for now
                    #   scores = func_predict(data['graph'])

                    # Write to log-file
                    logging.debug(f'Did not evaluate model (unsupported deployment): {ID}')
                    continue
            
            # ------------------
            # Phase 5: Write MVA-scores out

            outpath    = aux.makedir(basepath + '/' + filename.rsplit('/', 1)[0])
            outputfile = basepath + '/' + filename.replace('.root', '-icenet.root')
            
            print(__name__ + f'.process_data: Saving root output to "{outputfile}"')

            with uproot.recreate(outputfile, compression=uproot.ZLIB(4)) as rfile:
                rfile[f"Events"] = ALL_scores

        # Write to log-file
        logging.debug(f'Total number of events: {total_num_events}')

def get_predictor(args, param, feature_names=None):    

    model = None

    if   param['predict'] == 'xgb':
        func_predict, model = predict.pred_xgb(args=args, param=param, feature_names=feature_names, return_model=True)

    elif param['predict'] == 'xgb_logistic':
        func_predict, model = predict.pred_xgb_logistic(args=args, param=param, feature_names=feature_names, return_model=True)

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

    return func_predict, model

"""
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
"""
