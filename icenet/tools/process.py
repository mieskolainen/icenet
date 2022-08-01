# Common input & data reading routines
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

import argparse
import numpy as np
import awkward as ak
import torch

from importlib import import_module
from termcolor import colored, cprint
import os
import copy
import sys
import pickle
import xgboost
from pprint import pprint
from yamlinclude import YamlIncludeConstructor


import icenet.deep.train as train
import icenet.deep.predict as predict


from icenet.tools import stx
from icenet.tools import io
from icenet.tools import prints
from icenet.tools import aux
from icenet.tools import reweight
from icenet.tools import plots


import matplotlib.pyplot as plt


# ******** GLOBALS *********
roc_mstats        = []
roc_labels        = []
roc_paths         = []
corr_mstats       = []
ROC_binned_mstats = []
ROC_binned_mlabel = []

# **************************

def read_cli():

    parser = argparse.ArgumentParser()
    
    ## argparse.SUPPRESS removes argument from the namespace if not passed
    parser.add_argument("--runmode",         type=str,  default='null')
    parser.add_argument("--config",          type=str,  default='tune0.yml')
    parser.add_argument("--datapath",        type=str,  default='.')
    parser.add_argument("--datasets",        type=str,  default='*.root')
    parser.add_argument("--tag",             type=str,  default='tag0')

    parser.add_argument("--maxevents",       type=int,  default=argparse.SUPPRESS)
    parser.add_argument("--use_conditional", type=int,  default=argparse.SUPPRESS)
    parser.add_argument("--use_cache",       type=int,  default=1)
    parser.add_argument("--inputmap",        type=str,  default=None)
    parser.add_argument("--modeltag",        type=str,  default=None)


    cli      = parser.parse_args()
    cli_dict = vars(cli)

    return cli, cli_dict


def read_config(config_path='configs/xyz/', runmode='all'):
    """
    Commandline and YAML configuration reader
    """

    # -------------------------------------------------------------------
    ## Parse command line arguments
    cli, cli_dict = read_cli()

    # -------------------------------------------------------------------
    ## Read yaml configuration

    # This allows to use "!include foo.yaml" syntax
    import yaml
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir="")

    args = {}
    config_yaml_file = cli.config
    with open(f'{config_path}/{config_yaml_file}', 'r') as f:
        try:
            args = yaml.load(f, Loader=yaml.FullLoader)

            if 'includes' in args:
                del args['args'] # ! crucial

        except yaml.YAMLError as exc:
            print(exc)
            exit()

    # -------------------------------------------------------------------
    ## Inputmap .yml setup


    if cli_dict['inputmap'] is not None:
        args["genesis_runmode"]["inputmap"] = cli_dict['inputmap']        

    if args["genesis_runmode"]["inputmap"] is not None:
        file = args["genesis_runmode"]["inputmap"]

        ## ** Special YAML loader here **
        from libs.ccorp.ruamel.yaml.include import YAML
        yaml = YAML()
        yaml.allow_duplicate_keys = True

        with open(f'{config_path}/{file}', 'r') as f:
            try:
                inputmap = yaml.load(f)

                if 'includes' in inputmap:
                    del inputmap['includes'] # ! crucial

            except yaml.YAMLError as exc:
                print(f'yaml.YAMLError: {exc}')
                exit()
    else:
        inputmap = {}
    
    # -------------------------------------------------------------------
    # Mode setup

    new_args = {}
    new_args['rootname'] = args['rootname']
    new_args['rngseed']  = args['rngseed']
    
    # -----------------------------------------------------
    if   runmode == 'genesis':
        new_args.update(args['genesis_runmode'])
        new_args.update(inputmap)
    elif runmode == 'train':
        new_args.update(args['train_runmode'])
        new_args['plot_param'] = args['plot_param']
    elif runmode == 'eval':
        new_args.update(args['eval_runmode'])
        new_args['plot_param'] = args['plot_param']
    elif runmode == 'all':
        for key in args.keys():
            if "_runmode" in key:
                new_args.update(args[key])
        new_args['plot_param'] = args['plot_param']
    else:
        raise Exception(__name__ + f'.read_config: Unknown runmode = {runmode}')
    # -----------------------------------------------------

    old_args = copy.deepcopy(args)
    args     = copy.deepcopy(new_args)

    # -------------------------------------------------------------------
    ## Commandline override of yaml variables
    for key in cli_dict.keys():
        if key in args:
            cprint(__name__ + f'.read_config: {config_yaml_file} <{key}> default value cli-override with <{cli_dict[key]}>', 'red')
            args[key] = cli_dict[key]
    print()

    # -------------------------------------------------------------------
    ## Create a hash based on "rngseed", "maxevents", "genesis" and "inputmap" fields of yaml
    
    hash_args = {}

    mvavars_path = f'{config_path}/mvavars.py'
    if os.path.exists(mvavars_path):
        hash_args['__hash__mvavars.py'] = io.make_hash_sha256_file(mvavars_path)
    
    hash_args.update(old_args['genesis_runmode']) # This first !
    hash_args['rngseed']   = args['rngseed']
    hash_args['maxevents'] = args['maxevents']
    hash_args.update(inputmap)

    args['__hash__'] = io.make_hash_sha256_object(hash_args)


    # -------------------------------------------------------------------
    ## Create new variables

    args["config"]     = cli_dict['config']
    args["modeltag"]   = cli_dict['modeltag']

    args['datadir']    = aux.makedir(f'output/{args["rootname"]}')
    args['modeldir']   = aux.makedir(f'checkpoint/{args["rootname"]}/config-[{cli_dict["config"]}]/modeltag-[{cli_dict["modeltag"]}]')
    args['plotdir']    = aux.makedir(f'figs/{args["rootname"]}/config-[{cli_dict["config"]}]/inputmap-[{cli_dict["inputmap"]}]--modeltag-[{cli_dict["modeltag"]}]')
    
    args['root_files'] = io.glob_expand_files(datasets=cli.datasets, datapath=cli.datapath)

    # Technical
    args['__use_cache__']       = bool(cli_dict['use_cache'])
    args['__raytune_running__'] = False

    # -------------------------------------------------------------------
    ## Create directories
    aux.makedir('tmp')
    aux.makedir(args['datadir'])
    aux.makedir(args['modeldir'])
    aux.makedir(args['plotdir'])

    # -------------------------------------------------------------------
    # Set random seeds for reproducability and train-validate-test splits

    print('')
    print(" torch.__version__: " + torch.__version__)

    cprint(__name__ + f'.read_config: Setting random seed: {args["rngseed"]}', 'yellow')
    np.random.seed(args['rngseed'])
    torch.manual_seed(args['rngseed'])

    # ------------------------------------------------
    print(__name__ + f'.read_config: Created arguments dictionary with runmode = <{runmode}> :')    
    # ------------------------------------------------
    
    return args, cli


def read_data(args, func_loader, runmode):
    """
    Load input data
    
    Args:
        args:  main argument dictionary
        func_loader:  application specific root file loader function
        
    Returns:
        data
    """
    cache_filename = f'{args["datadir"]}/data_{args["__hash__"]}.pkl'

    if args['__use_cache__'] == False or (not os.path.exists(cache_filename)):

        load_args = {'entry_start': 0, 'entry_stop': args['maxevents'], 'args': args}
        
        if runmode != "genesis":
            raise Exception(__name__ + f'.read_data: Data not in cache (or __use_cache__ == False) but --runmode is not "genesis"')

        # N.B. This loop is needed, because certain applications have each root file loaded here,
        # whereas some apps do all the multi-file processing under 'func_loader'
        for k in range(len(args['root_files'])):
            X_,Y_,W_,ids = func_loader(root_path=args['root_files'][k], **load_args)

            if k == 0:
                X,Y,W = copy.deepcopy(X_), copy.deepcopy(Y_), copy.deepcopy(W_)
            else:
                if   isinstance(X, np.ndarray):
                    concat = np.concatenate
                elif isinstance(X, ak.Array):
                    concat = ak.concatenate
                
                X = concat((X, X_), axis=0)
                Y = concat((Y, Y_), axis=0)
                if W is not None:
                    W = concat((W, W_), axis=0)

        with open(cache_filename, 'wb') as handle:
            cprint(__name__ + f'.read_data: Saving to cache: "{cache_filename}"', 'yellow')
            pickle.dump([X, Y, W, ids, args], handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        with open(cache_filename, 'rb') as handle:
            cprint(__name__ + f'.read_data: Loading from cache: "{cache_filename}"', 'yellow')
            X, Y, W, ids, genesis_args = pickle.load(handle)

            cprint(__name__ + f'.read_data: Cached data was generated with arguments:', 'yellow')
            pprint(genesis_args)

    return X, Y, W, ids


def process_data(args, X, Y, W, ids, func_factor, mvavars, runmode):

    # ----------------------------------------------------------
    # Pop out conditional variables if they exist
    if args['use_conditional'] == False:

        index  = []
        idxvar = []
        for i in range(len(ids)):
            if '__model' not in ids[i]:
                index.append(i)
                idxvar.append(ids[i])
            else:
                print(__name__ + f'.process_data: Removing conditional variable "{ids[i]}"')

        if   isinstance(X, np.ndarray):
            X = X[:,np.array(index, dtype=int)]
        elif isinstance(X, ak.Array):
            X = X[idxvar]

        ids = [ids[j] for j in index]
    # ----------------------------------------------------------
    
    # Split into training, validation, test
    trn, val, tst = io.split_data(X=X, Y=Y, W=W, ids=ids, frac=args['frac'])
    
    # ----------------------------------------
    if args['imputation_param']['active']:
        module = import_module(mvavars, 'configs.subpkg')

        var    = args['imputation_param']['var']
        if var is not None:
            impute_vars = getattr(module, var)
        else:
            impute_vars = None
    # ----------------------------------------
    
    ### Split and factor data
    output = {}
    if   runmode == 'train':

        ### Compute reweighting weights (before funcfactor because we need all the variables !)
        if args['reweight']:
            trn.w, pdf = reweight.compute_ND_reweights(pdf=None, x=trn.x, y=trn.y, w=trn.w, ids=trn.ids, args=args['reweight_param'])
            val.w,_    = reweight.compute_ND_reweights(pdf=pdf,  x=val.x, y=val.y, w=val.w, ids=val.ids, args=args['reweight_param'])
            pickle.dump(pdf, open(args["modeldir"] + '/reweight_pdf.pkl', 'wb'))
        
        # Compute different data representations
        output['trn'] = func_factor(x=trn.x, y=trn.y, w=trn.w, ids=trn.ids, args=args)
        output['val'] = func_factor(x=val.x, y=val.y, w=val.w, ids=val.ids, args=args)
        
        ## Imputate
        if args['imputation_param']['active']:
            output['trn']['data'], imputer = impute_datasets(data=output['trn']['data'], features=impute_vars, args=args['imputation_param'], imputer=None)
            output['val']['data'], imputer = impute_datasets(data=output['val']['data'], features=impute_vars, args=args['imputation_param'], imputer=imputer)
            
            pickle.dump(imputer, open(args["modeldir"] + f'/imputer.pkl', 'wb'))

    elif runmode == 'eval':
        
        ### Compute reweighting weights (before funcfactor because we need all the variables !)
        if args['reweight']:
            pdf      = pickle.load(open(args["modeldir"] + '/reweight_pdf.pkl', 'rb'))
            tst.w, _ = reweight.compute_ND_reweights(pdf=pdf, x=tst.x, y=tst.y, w=tst.w, ids=tst.ids, args=args['reweight_param'])
        
        # Compute different data representations
        output['tst'] = func_factor(x=tst.x, y=tst.y, w=tst.w, ids=tst.ids, args=args)
        
        ## Imputate
        if args['imputation_param']['active']:
            
            imputer = pickle.load(open(args["modeldir"] + f'/imputer.pkl', 'rb'))
            output['tst']['data'], _  = impute_datasets(data=output['tst']['data'], features=impute_vars, args=args['imputation_param'], imputer=imputer)
    
    return output


def impute_datasets(data, args, features=None, imputer=None):
    """
    Dataset imputation

    Args:
        data:        .x, .y, .w, .ids type object
        args:        imputer parameters
        features:    variables to impute (list), if None, then all are considered
        imputer:     imputer object (scikit-type)

    Return:
        imputed data
    """

    if features is None:
        features = data.ids

    # Choose active dimensions
    dim = np.array([i for i in range(len(data.ids)) if data.ids[i] in features], dtype=int)

    if args['values'] is not None:
        
        special_values = args['values'] # possible special values
        cprint(__name__ + f'.impute_datasets: Imputing data for special values {special_values} in variables {features}', 'yellow')

        # Parameters
        param = {
            "dim":        dim,
            "values":     special_values,
            "labels":     data.ids,
            "algorithm":  args['algorithm'],
            "fill_value": args['fill_value'],
            "knn_k":      args['knn_k']
        }
        
        data.x, imputer = io.impute_data(X=data.x, imputer=imputer, **param)

    else:
        cprint(__name__ + f'.impute_datasets: Imputing data for Inf/Nan in variables {features}', 'yellow')

        # No other imputation, but fix spurious NaN / Inf
        data.x[np.logical_not(np.isfinite(data.x[:, dim]))] = args['fill_value']

    return data, imputer


def train_models(data_trn, data_val, args=None) :
    """
    Train ML/AI models wrapper with pre-processing.
    
    Args:
        Different datatype objects (see the code)
    
    Returns:
        Saves trained models to disk
    """

    # @@ Tensor normalization @@
    if data_trn['data_tensor'] is not None and (args['varnorm_tensor'] == 'zscore'):
            
        print('\nZ-score normalizing tensor variables ...')
        X_mu_tensor, X_std_tensor = io.calc_zscore_tensor(data_trn['data_tensor'])
        
        data_trn['data_tensor'] = io.apply_zscore_tensor(data_trn['data_tensor'], X_mu_tensor, X_std_tensor)
        data_val['data_tensor'] = io.apply_zscore_tensor(data_val['data_tensor'], X_mu_tensor, X_std_tensor)
        
        # Save it for the evaluation
        pickle.dump([X_mu_tensor, X_std_tensor], open(args["modeldir"] + '/zscore_tensor.pkl', 'wb'))    
    
    # --------------------------------------------------------------------

    # @@ Truncate outliers (component by component) from the training set @@
    if args['outlier_param']['algo'] == 'truncate' :
        for j in range(data_trn['data'].x.shape[1]):

            minval = np.percentile(data_trn['data'].x[:,j], args['outlier_param']['qmin'])
            maxval = np.percentile(data_trn['data'].x[:,j], args['outlier_param']['qmax'])

            data_trn['data'].x[data_trn['data'].x[:,j] < minval, j] = minval
            data_trn['data'].x[data_trn['data'].x[:,j] > maxval, j] = maxval

    # @@ Variable normalization @@
    if   args['varnorm'] == 'zscore' :

        print('\nZ-score normalizing variables ...')
        X_mu, X_std = io.calc_zscore(data_trn['data'].x)
        data_trn['data'].x  = io.apply_zscore(data_trn['data'].x, X_mu, X_std)
        data_val['data'].x  = io.apply_zscore(data_val['data'].x, X_mu, X_std)

        # Save it for the evaluation
        pickle.dump([X_mu, X_std], open(args['modeldir'] + '/zscore.pkl', 'wb'))
        
        prints.print_variables(data_trn['data'].x, data_trn['data'].ids)

    elif args['varnorm'] == 'madscore' :

        print('\nMAD-score normalizing variables ...')
        X_m, X_mad  = io.calc_madscore(data_trn['data'].x)
        data_trn['data'].x  = io.apply_zscore(data_trn['data'].x, X_m, X_mad)
        data_val['data'].x  = io.apply_zscore(data_val['data'].x, X_m, X_mad)

        # Save it for the evaluation
        pickle.dump([X_m, X_mad], open(args['modeldir'] + '/madscore.pkl', 'wb'))
    
        prints.print_variables(data_trn['data'].x, data_trn['data'].ids)


    def set_distillation_drain(ID, param, inputs, dtype='torch'):
        if args['distillation']['drains'] is not None:
            if ID in args['distillation']['drains']:
                cprint(__name__ + f'.train_models: Creating soft distillation drain for the model <{ID}>', 'yellow')
                
                # By default to torch
                inputs['y_soft'] = torch.tensor(y_soft, dtype=torch.float)
                
                if dtype == 'numpy':
                    inputs['y_soft'] = inputs['y_soft'].detach().cpu().numpy()
    
    # Loop over active models
    for i in range(len(args['active_models'])):

        # Collect garbage
        import gc
        gc.collect()
        
        ID    = args['active_models'][i]
        param = args['models'][ID]
        print(f'Training <{ID}> | {param} \n')

        ## Different model
        if   param['train'] == 'torch_graph':
            
            inputs = {'data_trn': data_trn['data_graph'],
                      'data_val': data_val['data_graph'],
                      'args':     args,
                      'param':    param}
            
            set_distillation_drain(ID=ID, param=param, inputs=inputs)

            if ID in args['raytune']['param']['active']:
                model = train.raytune_main(inputs=inputs, train_func=train.train_torch_graph)
            else:
                model = train.train_torch_graph(**inputs)
        
        elif param['train'] == 'xgb':

            inputs = {'data_trn':    data_trn['data'],
                      'data_val':    data_val['data'],
                      'args':        args,
                      'data_trn_MI': data_trn['data_MI'] if 'data_MI' in data_trn else None,
                      'data_val_MI': data_val['data_MI'] if 'data_MI' in data_val else None,
                      'param':       param}
            
            set_distillation_drain(ID=ID, param=param, inputs=inputs, dtype='numpy')
            
            if ID in args['raytune']['param']['active']:
                model = train.raytune_main(inputs=inputs, train_func=train.train_xgb)
            else:
                model = train.train_xgb(**inputs)

        elif param['train'] == 'torch_deps':
            
            inputs = {'X_trn':       torch.tensor(data_trn['data_deps'].x, dtype=torch.float),
                      'Y_trn':       torch.tensor(data_trn['data'].y,      dtype=torch.long),
                      'X_val':       torch.tensor(data_val['data_deps'].x, dtype=torch.float),
                      'Y_val':       torch.tensor(data_val['data'].y,      dtype=torch.long),
                      'X_trn_2D':    None,
                      'X_val_2D':    None,
                      'trn_weights': torch.tensor(data_trn['data'].w, dtype=torch.float),
                      'val_weights': torch.tensor(data_val['data'].w, dtype=torch.float),
                      'data_trn_MI': data_trn['data_MI'] if 'data_MI' in data_trn else None,
                      'data_val_MI': data_val['data_MI'] if 'data_MI' in data_val else None,
                      'args':        args,
                      'param':       param}
            
            set_distillation_drain(ID=ID, param=param, inputs=inputs)

            if ID in args['raytune']['param']['active']:
                model = train.raytune_main(inputs=inputs, train_func=train.train_torch_generic)
            else:
                model = train.train_torch_generic(**inputs)        

        elif param['train'] == 'torch_generic':
            
            inputs = {'X_trn':       torch.tensor(aux.red(data_trn['data'].x, data_trn['data'].ids, param, 'X'), dtype=torch.float),
                      'Y_trn':       torch.tensor(data_trn['data'].y, dtype=torch.long),
                      'X_val':       torch.tensor(aux.red(data_val['data'].x, data_val['data'].ids, param, 'X'), dtype=torch.float),
                      'Y_val':       torch.tensor(data_val['data'].y, dtype=torch.long),
                      'X_trn_2D':    None if data_trn['data_tensor'] is None else torch.tensor(data_trn['data_tensor'], dtype=torch.float),
                      'X_val_2D':    None if data_val['data_tensor'] is None else torch.tensor(data_val['data_tensor'], dtype=torch.float),
                      'trn_weights': torch.tensor(data_trn['data'].w, dtype=torch.float),
                      'val_weights': torch.tensor(data_val['data'].w, dtype=torch.float),
                      'data_trn_MI': data_trn['data_MI'] if 'data_MI' in data_trn else None,
                      'data_val_MI': data_val['data_MI'] if 'data_MI' in data_val else None,
                      'args':  args,
                      'param': param}

            set_distillation_drain(ID=ID, param=param, inputs=inputs)

            if ID in args['raytune']['param']['active']:
                model = train.raytune_main(inputs=inputs, train_func=train.train_torch_generic)
            else:
                model = train.train_torch_generic(**inputs)

        elif param['train'] == 'graph_xgb':

            inputs = {'y_soft': None}
            set_distillation_drain(ID=ID, param=param, inputs=inputs)

            train.train_graph_xgb(data_trn=data_trn['data_graph'], data_val=data_val['data_graph'], 
                trn_weights=data_trn['data'].w, val_weights=data_val['data'].w, args=args, param=param, y_soft=inputs['y_soft'],
                feature_names=data_trn['data'].ids)  
        
        elif param['train'] == 'flr':
            train.train_flr(data_trn=data_trn['data'], args=args, param=param)

        elif param['train'] == 'flow':
            train.train_flow(data_trn=data_trn['data'], data_val=data_val['data'], args=args, param=param)

        elif param['train'] == 'cut':
            None
        elif param['train'] == 'cutset':
            None
        else:
            raise Exception(__name__ + f'.Unknown param["train"] = {param["train"]} for ID = {ID}')

        # --------------------------------------------------------
        # If distillation
        if ID == args['distillation']['source']:
            
            if args['num_classes'] != 2:
                raise Exception(__name__ + f'.train_models: Distillation supported now only for 2-class classification')
            
            if   param['train'] == 'xgb':
                cprint(__name__ + f'.train.models: Computing distillation soft targets from the source <{ID}> ', 'yellow')
                
                y_soft = model.predict(xgboost.DMatrix(data = data_trn['data'].x))
                if len(y_soft.shape) > 1: y_soft = y_soft[:, args['signalclass']]

            elif 'torch_' in param['train']:
                cprint(__name__ + f'.train.models: Computing distillation soft targets from the source <{ID}> ', 'yellow')
                y_soft = model.softpredict(data_trn['data_graph'])[:, args['signalclass']]
            else:
                raise Exception(__name__ + f".train_models: Unsupported distillation source <{param['train']}>")
        # --------------------------------------------------------

    return


def evaluate_models(data=None, args=None):
    """
    Evaluate ML/AI models.

    Args:
        Different datatype objects (see the code)

    Returns:
        Saves evaluation plots to the disk
    """

    print(__name__ + ".evaluate_models: Evaluating models")
    print('')
    
    
    # -----------------------------
    # ** GLOBALS **

    global roc_mstats
    global roc_labels
    global corr_mstats
    roc_mstats  = {}
    roc_labels  = {}
    roc_paths   = {}
    corr_mstats = {}

    global ROC_binned_mstats
    global ROC_binned_mlabel

    #mva_mstats = []
    #MVA_binned_mstats = []
    #MVA_binned_mlabel = []

    ROC_binned_mstats = [list()] * len(args['active_models'])
    ROC_binned_mlabel = [list()] * len(args['active_models'])

    # -----------------------------
    # Prepare output folders

    targetdir  = f'{args["plotdir"]}/eval'

    subdirs = ['', 'ROC', 'MVA', 'COR']
    for sd in subdirs:
        os.makedirs(targetdir + '/' + sd, exist_ok = True)

    # --------------------------------------------------------------------
    # Collect data

    X       = None
    X_RAW   = None
    ids_RAW = None

    y       = None
    weights = None
    
    if data['data'] is not None:

        ## Set feature indices for simple cut classifiers
        args['features'] = data['data'].ids

        X        = copy.deepcopy(data['data'].x)
        ids      = data['data'].ids

        X_RAW    = data['data'].x
        ids_RAW  = data['data'].ids

        # Add extra variables
        if data['data_kin'] is not None:
            X_RAW    = np.concatenate([X_RAW, data['data_kin'].x], axis=1)
            ids_RAW  = ids_RAW + data['data_kin'].ids

        y        = data['data'].y
        weights  = data['data'].w

    X_2D = None
    if data['data_tensor'] is not None:
        X_2D    = data['data_tensor']

    X_graph = None
    if data['data_graph'] is not None:
        X_graph = data['data_graph']

    X_deps = None
    if data['data_deps'] is not None:
        X_deps  = data['data_deps'].x
    
    X_kin    = None
    VARS_kin = None
    if data['data_kin'] is not None:
        X_kin    = data['data_kin'].x
        VARS_kin = data['data_kin'].ids
    # --------------------------------------------------------------------

    if weights is not None: print(__name__ + ".evaluate_models: -- per event weighted evaluation ON ")
    
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
        
        inputs = {'weights': weights, 'label': param['label'],
                 'targetdir': targetdir, 'args':args, 'X_kin': X_kin, 'VARS_kin': VARS_kin, 'X_RAW': X_RAW, 'ids_RAW': ids_RAW}

        if   param['predict'] == 'xgb':
            func_predict = predict.pred_xgb(args=args, param=param)
            if args['plot_param']['contours']['active']:
                plots.plot_contour_grid(pred_func=func_predict, X=aux.red(X,ids,param,'X'), y=y, ids=aux.red(X,ids,param,'ids'), transform='numpy', 
                    targetdir=aux.makedir(f'{args["plotdir"]}/eval/2D_contours/{param["label"]}/'))
            
            plot_XYZ_wrap(func_predict = func_predict, x_input=aux.red(X,ids,param,'X'), y=y, **inputs)

        elif param['predict'] == 'xgb_logistic':
            func_predict = predict.pred_xgb_logistic(args=args, param=param)
            
            if args['plot_param']['contours']['active']:
                plots.plot_contour_grid(pred_func=func_predict, X=aux.red(X,ids,param,'X'), y=y, ids=aux.red(X,ids,param,'ids'), transform='numpy', 
                    targetdir=aux.makedir(f'{args["plotdir"]}/eval/2D_contours/{param["label"]}/'))
            
            plot_XYZ_wrap(func_predict = func_predict, x_input=aux.red(X,ids,param,'X'), y=y, **inputs)

        elif param['predict'] == 'torch_vector':
            func_predict = predict.pred_torch_generic(args=args, param=param)

            if args['plot_param']['contours']['active']:
                plots.plot_contour_grid(pred_func=func_predict, X=aux.red(X_ptr,ids,param,'X'), y=y, ids=aux.red(X_ptr,ids,param,'ids'),
                    targetdir=aux.makedir(f'{args["plotdir"]}/eval/2D_contours/{param["label"]}/'), transform='torch')

            plot_XYZ_wrap(func_predict = func_predict, x_input=aux.red(X_ptr,ids,param,'X'), y=y, **inputs)

        elif param['predict'] == 'torch_scalar':
            func_predict = predict.pred_torch_scalar(args=args, param=param)

            if args['plot_param']['contours']['active']:
                plots.plot_contour_grid(pred_func=func_predict, X=aux.red(X_ptr,ids,param,'X'), y=y, ids=aux.red(X_ptr,ids,param,'ids'), transform='torch', 
                    targetdir=aux.makedir(f'{args["plotdir"]}/eval/2D_contours/{param["label"]}/'))

            plot_XYZ_wrap(func_predict = func_predict, x_input=aux.red(X_ptr,ids,param,'X'), y=y, **inputs)

        elif param['predict'] == 'torch_flow':
            func_predict = predict.pred_flow(args=args, param=param, n_dims=X_ptr.shape[1])

            if args['plot_param']['contours']['active']:
                plots.plot_contour_grid(pred_func=func_predict, X=aux.red(X_ptr,ids,param,'X'), y=y, ids=aux.red(X_ptr,ids,param,'ids'), transform='torch', 
                    targetdir=aux.makedir(f'{args["plotdir"]}/eval/2D_contours/{param["label"]}/'))

            plot_XYZ_wrap(func_predict = func_predict, x_input=aux.red(X_ptr,ids,param,'X'), y=y, **inputs)

        elif   param['predict'] == 'torch_graph':
            func_predict = predict.pred_torch_graph(args=args, param=param)

            # Geometric type -> need to use batch loader, get each graph, node or edge prediction
            import torch_geometric
            loader  = torch_geometric.loader.DataLoader(X_graph, batch_size=len(X_graph), shuffle=False)
            for batch in loader: # Only one big batch
                plot_XYZ_wrap(func_predict = func_predict, x_input=X_graph, y=batch.to('cpu').y.detach().cpu().numpy(), **inputs)
        
        elif param['predict'] == 'graph_xgb':
            func_predict = predict.pred_graph_xgb(args=args, param=param)
            plot_XYZ_wrap(func_predict = func_predict, x_input = X_graph, y=y, **inputs)
        
        elif param['predict'] == 'torch_deps':
            func_predict = predict.pred_torch_generic(args=args, param=param)
            plot_XYZ_wrap(func_predict = func_predict, x_input = X_deps_ptr, y=y, **inputs)

        elif param['predict'] == 'torch_image':
            func_predict = predict.pred_torch_generic(args=args, param=param)
            plot_XYZ_wrap(func_predict = func_predict, x_input = X_2D_ptr, y=y, **inputs)
            
        elif param['predict'] == 'torch_image_vector':
            func_predict = predict.pred_torch_generic(args=args, param=param)

            X_dual      = {}
            X_dual['x'] = X_2D_ptr # image tensors
            X_dual['u'] = X_ptr    # global features
            plot_XYZ_wrap(func_predict = func_predict, x_input = X_dual, y=y, **inputs)
            
        elif param['predict'] == 'flr':
            func_predict = predict.pred_flr(args=args, param=param)
            plot_XYZ_wrap(func_predict = func_predict, x_input=aux.red(X,ids,param,'X'), y = y, **inputs)
            
        #elif param['predict'] == 'xtx':
        # ...   
        #
        
        elif param['predict'] == 'cut':
            func_predict = predict.pred_cut(args=args, param=param)
            plot_XYZ_wrap(func_predict = func_predict, x_input = X_RAW, y = y, **inputs)
            
        elif param['predict'] == 'cutset':
            func_predict = predict.pred_cutset(args=args, param=param)
            plot_XYZ_wrap(func_predict = func_predict, x_input = X_RAW, y = y, **inputs)

            if args['plot_param']['contours']['active']:
                plots.plot_contour_grid(pred_func=func_predict, X=X_RAW, y=y, ids=ids_RAW, transform='numpy', 
                    targetdir=aux.makedir(f'{args["plotdir"]}/eval/2D_contours/{param["label"]}/'))

        else:
            raise Exception(__name__ + f'.Unknown param["predict"] = {param["predict"]} for ID = {ID}')

    ## Multiple model comparisons
    plot_XYZ_multiple_models(targetdir=targetdir, args=args)

    return


def make_plots(data, args):

    ### Plot variables
    if args['plot_param']['basic']['active']:

        ###
        if data['data_kin'] is not None:
            targetdir = aux.makedir(f'{args["plotdir"]}/reweight/1D_kinematic/')
            for k in data['data_kin'].ids:
                plots.plotvar(x = data['data_kin'].x[:, data['data_kin'].ids.index(k)],
                    y = data['data_kin'].y, weights = data['data_kin'].w, var = k, nbins = args['plot_param']['basic']['nbins'],
                    targetdir = targetdir, title = f"training re-weight reference class: {args['reweight_param']['reference_class']}")
        
        ### Plot correlations
        targetdir = aux.makedir(f'{args["plotdir"]}/train/')
        fig,ax    = plots.plot_correlations(X=data['data'].x, weights=data['data'].w, ids=data['data'].ids, classes=data['data'].y, targetdir=targetdir)
        
        ### Plot basic plots
        targetdir = aux.makedir(f'{args["plotdir"]}/train/1D_distributions/')
        plots.plotvars(X = data['data'].x, y = data['data'].y, weights = data['data'].w, nbins = args['plot_param']['basic']['nbins'], ids = data['data'].ids,
            targetdir = targetdir, title = f"training re-weight reference class: {args['reweight_param']['reference_class']}")


def plot_XYZ_wrap(func_predict, x_input, y, weights, label, targetdir, args,
    X_kin, VARS_kin, X_RAW, ids_RAW):
    """ 
    Arbitrary plot wrapper function.
    """

    global roc_mstats
    global roc_labels
    global roc_paths

    global corr_mstats

    global ROC_binned_mstats
    global ROC_binned_mlabel

    # ** Compute predictions once and for all here **
    y_pred = func_predict(x_input)

    # --------------------------------------
    ### ROC plots

    if args['plot_param']['ROC']['active']:

        def plot_helper(mask, sublabel, pathlabel):
            metric = aux.Metric(y_true=y[mask], y_pred=y_pred[mask], weights=weights[mask])

            if sublabel not in roc_mstats:
                roc_mstats[sublabel] = []
                roc_labels[sublabel] = []
                roc_paths[sublabel]  = []

            roc_mstats[sublabel].append(metric)
            roc_labels[sublabel].append(label)
            roc_paths[sublabel].append(pathlabel)
        
        # ** All inclusive **
        mask      = np.ones(len(y_pred), dtype=bool)
        plot_helper(mask=mask, sublabel='inclusive', pathlabel='inclusive')

        # ** Powerset filtered **
        if 'powerset_filter' in args['plot_param']['ROC']:

            filters = args['plot_param']['ROC']['powerset_filter']
            mask_powerset, text_powerset, path_powerset = filter_constructor(filters=filters, X_RAW=X_RAW, ids_RAW=ids_RAW)

            for m in range(mask_powerset.shape[0]):
                plot_helper(mask=mask_powerset[m,:], sublabel=text_powerset[m], pathlabel=path_powerset[m])

    # --------------------------------------
    ### ROC binned plots (no powerset selection supported here)
    if args['plot_param']['ROC_binned']['active']:

        for i in range(100): # Loop over plot types
            
            pid = f'plot[{i}]'

            if pid in args['plot_param']['ROC_binned']:
                var   = args['plot_param']['ROC_binned'][pid]['var']
                edges = args['plot_param']['ROC_binned'][pid]['edges']
            else:
                break # No more this type of plots 

            ## 1D
            if   len(var) == 1:

                met_1D, label_1D = plots.binned_1D_AUC(y_pred=y_pred, y=y, weights=weights, X_kin=X_kin, \
                    VARS_kin=VARS_kin, edges=edges, label=label, ids=var[0])

                # Save for multiple comparison
                ROC_binned_mstats[i].append(met_1D)
                ROC_binned_mlabel[i].append(label_1D)

                # Plot this one
                plots.ROC_plot(met_1D, label_1D, title = f'{label}', filename=aux.makedir(f'{targetdir}/ROC/{label}') + f'/ROC-binned[{i}]')
                plots.MVA_plot(met_1D, label_1D, title = f'{label}', filename=aux.makedir(f'{targetdir}/MVA/{label}') + f'/MVA-binned[{i}]')

            ## 2D
            elif len(var) == 2:

                fig, ax, met = plots.binned_2D_AUC(y_pred=y_pred, y=y, weights=weights, X_kin=X_kin, \
                    VARS_kin=VARS_kin, edges=edges, label=label, ids=var)

                plt.savefig(aux.makedir(f'{targetdir}/ROC/{label}') + f'/ROC-binned[{i}].pdf', bbox_inches='tight')
                
            else:
                print(var)
                raise Exception(__name__ + f'.plot_AUC_wrap: Unknown dimensionality {len(var)}')

    # ----------------------------------------------------------------
    ### MVA-output 1D-plot
    if args['plot_param']['MVA_output']['active']:

        def plot_helper(mask, sublabel, pathlabel):
            hist_edges = args['plot_param'][f'MVA_output']['edges']
            inputs = {'y_pred': y_pred[mask], 'y': y[mask], 'weights': weights[mask], 'num_classes': args['num_classes'],
                'hist_edges': hist_edges, 'label': f'{label}/{sublabel}', 'path': targetdir + '/MVA/' + {pathlabel}}

            plots.density_MVA_wclass(**inputs)

        # ** All inclusive **
        mask  = np.ones(len(y_pred), dtype=bool)
        plot_helper(mask=mask, sublabel='inclusive', pathlabel='inclusive')

        # ** Powerset filtered **
        if 'powerset_filter' in args['plot_param']['MVA_output']:

            filters = args['plot_param']['MVA_output']['powerset_filter']
            mask_powerset, text_powerset, path_powerset = filter_constructor(filters=filters, X_RAW=X_RAW, ids_RAW=ids_RAW)

            for m in range(mask_powerset.shape[0]):
                plot_helper(mask=mask_powerset[m,:], sublabel=text_powerset[m], pathlabel=path_powerset[m])

    # ----------------------------------------------------------------
    ### MVA-output 2D correlation plots
    if args['plot_param']['MVA_2D']['active']:

        def plot_helper(mask, pick_ind, sublabel='inclusive', pathlabel='inclusive', savestats=False):

            # Two step
            XX = X_RAW[mask, ...]
            XX = XX[:, pick_ind]

            inputs = {'y_pred': y_pred[mask], 'weights': weights[mask], 'X': XX,
                'ids': np.array(ids_RAW, dtype=np.object_)[pick_ind].tolist(),
                'num_classes': args['num_classes'],
                'label': f'{label}/{sublabel}', 'hist_edges': edges, 'path': targetdir + f'/COR/{pathlabel}'}

            output = plots.density_COR_wclass(y=y[mask], **inputs)
            #plots.density_COR(**inputs)

            # Save output
            if savestats:
                if label not in corr_mstats.keys():
                    corr_mstats[label] = {}
                corr_mstats[label][sublabel] = output

        for i in range(100): # Loop over plot types
            
            pid = f'plot[{i}]'
            if pid in args['plot_param']['MVA_2D']:        
                var   = args['plot_param']['MVA_2D'][pid]['var']
                edges = args['plot_param']['MVA_2D'][pid]['edges']
            else:
                break # No more this type of plots 

            # Pick chosen variables based on regular expressions
            var_names = aux.process_regexp_ids(all_ids=ids_RAW, ids=var)
            pick_ind  = np.array(np.where(np.isin(ids_RAW, var_names))[0], dtype=int)
            
            # ** All inclusive **
            mask      = np.ones(len(y_pred), dtype=bool)

            # ** Powerset filtered **
            if 'powerset_filter' in args['plot_param']['MVA_2D'][pid]:

                plot_helper(mask=mask, pick_ind=pick_ind, sublabel='inclusive', pathlabel='inclusive', savestats=True)

                filters = args['plot_param']['MVA_2D'][pid]['powerset_filter']
                mask_powerset, text_powerset, path_powerset = filter_constructor(filters=filters, X_RAW=X_RAW, ids_RAW=ids_RAW)

                for m in range(mask_powerset.shape[0]):
                    plot_helper(mask=mask_powerset[m,:], pick_ind=pick_ind, sublabel=text_powerset[m], pathlabel=path_powerset[m], savestats=True)
            else:
                plot_helper(mask=mask, pick_ind=pick_ind, sublabel='inclusive', pathlabel='inclusive', savestats=False)                

    return True


def filter_constructor(filters, X_RAW, ids_RAW):
    """
    Powerset filter constructor

    Returns:
        mask matrix, mask text labels
    """
    cutlist  = [filters[k]['cut']   for k in range(len(filters))]
    textlist = [filters[k]['latex'] for k in range(len(filters))]

    # Construct cuts and apply
    cuts, names   = stx.construct_columnar_cuts(X=X_RAW, ids=ids_RAW, cutlist=cutlist)
    stx.print_parallel_cutflow(cut=cuts, names=names)
    
    mask_powerset = stx.powerset_cutmask(cut=cuts)
    BMAT          = aux.generatebinary(len(cuts))

    print(textlist)

    # Loop over all powerset 2**|cuts| masked selections
    # Create a description latex strings and savepath strings
    text_powerset = []
    path_powerset = []
    for i in range(BMAT.shape[0]):
        string = ''
        for j in range(BMAT.shape[1]):
            bit = BMAT[i,j] # 0 or 1
            string += f'{textlist[j][bit]}'
            if j != BMAT.shape[1] - 1:
                string += ' '
        string += f' {BMAT[i,:]}'

        text_powerset.append(string)
        path_powerset.append((f'{BMAT[i,:]}').replace(' ', ''))

    return mask_powerset, text_powerset, path_powerset


def plot_XYZ_multiple_models(targetdir, args):

    global roc_mstats
    global roc_labels
    global roc_paths
    global ROC_binned_mstats

    # ===================================================================
    # Plot correlation coefficient comparisons

    from pprint import pprint
    pprint(corr_mstats)

    ### MVA-output 2D correlation plots
    if args['plot_param']['MVA_2D']['active']:

        for i in range(100): # Loop over plot indexes
            pid = f'plot[{i}]'
            if pid in args['plot_param']['MVA_2D']:
                if 'powerset_filter' in args['plot_param']['MVA_2D'][pid]:

                    xlim = args['plot_param']['MVA_2D'][pid]['xlim']
                    plots.plot_correlation_comparison(corr_mstats=corr_mstats, 
                        num_classes=args['num_classes'], targetdir=targetdir, xlim=xlim)
            else:
                break

    # ===================================================================
    # ** Plots for multiple model comparison **

    # -------------------------------------------------------------------
    ### Plot all ROC curves

    pprint(roc_mstats)
    
    # Direct collect:  Plot all models per powerset category
    for powerset_key in roc_mstats.keys():

        path_label = roc_paths[powerset_key]
        plots.ROC_plot(roc_mstats[powerset_key], roc_labels[powerset_key],
            title=f'category: {powerset_key}', filename=aux.makedir(targetdir + f'/ROC/--ALL--/{path_label}') + '/ROC-all-models')
    
    # Inverse collect: Plot all powerset categories ROCs per model
    dummy = 0 # We have the same number of powerset (category) entries for each model, pick the first
    for model_index in range(len(roc_mstats[list(roc_mstats)[dummy]])):
        
        rocs_       = [roc_mstats[powerset_key][model_index] for powerset_key in roc_mstats.keys()]
        labels_     = list(roc_mstats.keys())
        model_label = roc_labels[list(roc_labels)[dummy]][model_index]

        plots.ROC_plot(rocs_, labels_,
            title=f'model: {model_label}', filename=aux.makedir(targetdir + f'/ROC/{model_label}') + '/ROC-all-categories')

    ### Plot all MVA outputs (not implemented)
    #plots.MVA_plot(mva_mstats, mva_labels, title = '', filename=aux.makedir(targetdir + '/MVA/--ALL--') + '/MVA')

    ### Plot all binned ROC curves
    if args['plot_param']['ROC_binned']['active']:

        for i in range(100):
            pid = f'plot[{i}]'

            if pid in args['plot_param']['ROC_binned']:
                var   = args['plot_param']['ROC_binned'][pid]['var']
                edges = args['plot_param']['ROC_binned'][pid]['edges']
            else:
                break # No more plots 

            if len(var) == 1:

                # Over different bins
                for b in range(len(edges)-1):

                    # Over different models
                    xy,legs = [],[]
                    for k in range(len(ROC_binned_mstats[i])):
                        xy.append(ROC_binned_mstats[i][k][b])

                        # Take label for the legend
                        ID    = args['active_models'][k]
                        label = args['models'][ID]['label']
                        legs.append(label)

                    ### ROC
                    title = f'BINNED ROC: {var[0]}$ \\in [{edges[b]:0.1f}, {edges[b+1]:0.1f})$'
                    plots.ROC_plot(xy, legs, title=title, filename=targetdir + f'/ROC/--ALL--/ROC-binned[{i}]-bin[{b}]')

                    ### MVA (not implemented)
                    #title = f'BINNED MVA: {var[0]}$ \\in [{edges[b]:0.1f}, {edges[b+1]:0.1f})$'
                    #plots.MVA_plot(xy, legs, title=title, filename=targetdir + f'/MVA/--ALL--/MVA-binned[{i}]-bin[{b}]')

    return True
