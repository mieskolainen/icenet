# Common input & data reading routines
#
# Mikael Mieskolainen, 2023
# m.mieskolainen@imperial.ac.uk

import argparse
import numpy as np
import awkward as ak
import gc
import torch
import copy
from tqdm import tqdm

from importlib import import_module
from termcolor import colored, cprint
import os
import copy
import sys
import pickle
import xgboost
from yamlinclude import YamlIncludeConstructor

import icenet.deep.iceboost as iceboost
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
    parser.add_argument("--runmode",         type=str,  default='all')
    parser.add_argument("--config",          type=str,  default='tune0.yml')
    parser.add_argument("--datapath",        type=str,  default='')
    parser.add_argument("--datasets",        type=str,  default='')
    parser.add_argument("--tag",             type=str,  default='tag0')
    
    parser.add_argument("--maxevents",       type=int,  default=argparse.SUPPRESS)
    parser.add_argument("--use_conditional", type=int,  default=argparse.SUPPRESS)
    parser.add_argument("--use_cache",       type=int,  default=1)

    parser.add_argument("--grid_id",         type=int,  default=0)
    parser.add_argument("--grid_nodes",      type=int,  default=1)
    
    parser.add_argument("--inputmap",        type=str,  default=None)
    parser.add_argument("--modeltag",        type=str,  default=None)
    
    cli      = parser.parse_args()
    cli_dict = vars(cli)

    return cli, cli_dict


def read_config(config_path='configs/xyz/', runmode='all'):
    """
    Commandline and YAML configuration reader
    """
    cwd = os.getcwd()

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
    with open(f'{cwd}/{config_path}/{config_yaml_file}', 'r') as f:
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

        with open(f'{cwd}/{config_path}/{file}', 'r') as f:
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
    new_args['rootname']  = args['rootname']
    new_args['rngseed']   = args['rngseed']
    new_args['inputvars'] = args['inputvars']
    
    # -----------------------------------------------------
    # Runmode setup
    if   runmode == 'genesis':
        new_args.update(args['genesis_runmode'])
        new_args.update(inputmap)
    
    elif runmode == 'train':
        new_args.update(args['train_runmode'])
        new_args['plot_param'] = args['plot_param']

    elif runmode == 'eval':
        new_args.update(args['eval_runmode'])
        new_args['plot_param'] = args['plot_param']

    elif runmode == 'optimize':
        new_args.update(args['optimize_runmode'])
        new_args['plot_param'] = args['plot_param']
    
    elif runmode == 'deploy':
        new_args.update(args['genesis_runmode'])
        new_args.update(args['deploy_runmode'])
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
    ## Create a hash based on:
    # "rngseed", "maxevents", "inputvars", "genesis" and "inputmap" fields of yaml
    
    hash_args = {}

    inputvars_path = f'{cwd}/{config_path}/{args["inputvars"]}.py'
    if os.path.exists(inputvars_path):
        hash_args['__hash__inputvars'] = io.make_hash_sha256_file(inputvars_path)
    else:
        raise Exception(__name__ + f".read_config: Did not find: {inputvars_path}")

    hash_args.update(old_args['genesis_runmode']) # This first !
    hash_args['rngseed']   = args['rngseed']
    hash_args['maxevents'] = args['maxevents']
    hash_args['inputvars'] = args['inputvars']
    
    hash_args.update(inputmap)

    args['__hash_genesis__'] = io.make_hash_sha256_object(hash_args)
    
    cprint(__name__ + f'.read_config: Generated config hashes', 'magenta')
    cprint(f'[__hash_genesis__]      : {args["__hash_genesis__"]}     ', 'magenta')
    
    ## Second level hash (depends on all previous) + other parameters
    
    if runmode == 'train' or runmode == 'eval':
        hash_args['use_conditional']  = args['use_conditional']
        args['__hash_post_genesis__'] = args['__hash_genesis__'] + '__' + io.make_hash_sha256_object(hash_args)
        
        cprint(f'[__hash_post_genesis__] : {args["__hash_post_genesis__"]}', 'magenta')
    
    # -------------------------------------------------------------------
    ## Create new variables to args dictionary (and create directories)

    args["config"]     = cli_dict['config']
    args["modeltag"]   = cli_dict['modeltag']
    
    args['datadir']    = aux.makedir(f'{cwd}/output/{args["rootname"]}')
    
    if runmode != 'genesis':
        args['modeldir'] = aux.makedir(f'{cwd}/checkpoint/{args["rootname"]}/config__{io.safetxt(cli_dict["config"])}/modeltag__{cli_dict["modeltag"]}')
        args['plotdir']  = aux.makedir(f'{cwd}/figs/{args["rootname"]}/config__{io.safetxt(cli_dict["config"])}/inputmap__{io.safetxt(cli_dict["inputmap"])}--modeltag__{cli_dict["modeltag"]}')
    
    args['root_files'] = io.glob_expand_files(datasets=cli.datasets, datapath=cli.datapath)    
    
    # Technical
    args['__use_cache__']       = bool(cli_dict['use_cache'])
    args['__raytune_running__'] = False

    # Distributed computing
    for key in ['grid_id', 'grid_nodes']:
        args[key] = cli_dict[key]
    
    # -------------------------------------------------------------------
    ## Create aux
    aux.makedir('tmp')
    
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


def generic_flow(rootname, func_loader, func_factor):
    """
    Generic (data -- train -- evaluation) workflow
    
    Args:
        rootname:     name of the workflow config folder
        func_loader:  data loader (function handle)
        func_factor:  data transformer (function handle)
    """
    cli, cli_dict  = read_cli()
    runmode        = cli_dict['runmode']
    
    args, cli      = read_config(config_path=f'configs/{rootname}', runmode=runmode)
      
    if runmode == 'genesis':

        read_data(args=args, func_loader=func_loader, runmode=runmode) 
        
    if runmode == 'train' or runmode == 'eval':

        data = read_data_processed(args=args, func_loader=func_loader,
            func_factor=func_factor, mvavars=f'configs.{rootname}.mvavars', runmode=runmode)
        
    if runmode == 'train':
        
        prints.print_variables(X=data['trn']['data'].x, W=data['trn']['data'].w, ids=data['trn']['data'].ids,
                               exclude_vals=[args['imputation_param']['fill_value']])
        make_plots(data=data['trn'], args=args)
        train_models(data_trn=data['trn'], data_val=data['val'], args=args)

    if runmode == 'eval':

        prints.print_variables(X=data['tst']['data'].x, W=data['tst']['data'].w, ids=data['tst']['data'].ids,
                               exclude_vals=[args['imputation_param']['fill_value']])
        evaluate_models(data=data['tst'], info=data['info'], args=args)

    return args, runmode


def read_data(args, func_loader, runmode):
    """
    Load input data and return full dataset arrays
    
    Args:
        args:         main argument dictionary
        func_loader:  application specific root file loader function
    """
    
    def get_chunk_ind(N):
        chunks = int(np.ceil(N / args['pickle_size']))
        return aux.split_start_end(range(N), chunks)

    cache_directory = aux.makedir(f'{args["datadir"]}/data_{args["__hash_genesis__"]}')

    if args['__use_cache__'] == False or (not os.path.exists(f'{cache_directory}/output_0.pkl')):

        if runmode != "genesis":
            raise Exception(__name__ + f'.read_data: Data "{cache_directory}" not found (or __use_cache__ == False) but --runmode is not "genesis"')

        # func_loader does the multifile processing
        load_args = {'entry_start': 0, 'entry_stop': None, 'maxevents': args['maxevents'], 'args': args}
        predata   = func_loader(root_path=args['root_files'], **load_args)

        X    = predata['X']
        Y    = predata['Y']
        W    = predata['W']
        ids  = predata['ids']
        info = predata['info']

        cprint(__name__ + f'.read_data: Saving to path: "{cache_directory}"', 'yellow')
        C = get_chunk_ind(N=len(X))
        
        for i in tqdm(range(len(C))):
            with open(f'{cache_directory}/output_{i}.pkl', 'wb') as handle:
                pickle.dump([X[C[i][0]:C[i][-1]], Y[C[i][0]:C[i][-1]], W[C[i][0]:C[i][-1]], ids, info, args], \
                    handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return predata
        
    else:
        
        if runmode == "genesis": # Genesis mode does not need this
            return
        
        num_files = io.count_files_in_dir(cache_directory)
        cprint(__name__ + f'.read_data: Loading from path: "{cache_directory}"', 'yellow')
        
        for i in tqdm(range(num_files)):
            
            with open(f'{cache_directory}/output_{i}.pkl', 'rb') as handle:
                X_, Y_, W_, ids, info, genesis_args = pickle.load(handle)
                        
                if i > 0:
                    X = np.concatenate((X, X_), axis=0) # awkward will cast numpy automatically
                    Y = np.concatenate((Y, Y_), axis=0)
                    W = np.concatenate((W, W_), axis=0)
                else:
                    X,Y,W = copy.deepcopy(X_), copy.deepcopy(Y_), copy.deepcopy(W_)
                
                gc.collect() # important!
                
    
        return {'X':X, 'Y':Y, 'W':W, 'ids':ids, 'info':info}


def read_data_processed(args, func_loader, func_factor, mvavars, runmode):
    """
    Read/write (MVA) data and return full processed dataset
    """

    # --------------------------------------------------------------------
    # 'PREDATA': Raw input reading and processing
    
    cache_filename = f'{args["datadir"]}/data_{runmode}_{args["__hash_genesis__"]}.pkl'
    
    if args['__use_cache__'] == False or (not os.path.exists(cache_filename)):

        # Read it
        predata = read_data(args=args, func_loader=func_loader, runmode=runmode) 
        
        with open(cache_filename, 'wb') as handle:
            cprint(__name__ + f'.read_data_processed: Saving <DATA> to a file: "{cache_filename}"', 'yellow')
            
            # Disable garbage collector for speed
            gc.disable()
            pickle.dump(predata, handle, protocol=pickle.HIGHEST_PROTOCOL)
            gc.enable()
    else:
        with open(cache_filename, 'rb') as handle:
            cprint(__name__ + f'.read_data_processed: Loading <DATA> from a file: "{cache_filename}"', 'yellow')
            
            # Disable garbage collector for speed
            gc.disable()
            predata = pickle.load(handle)
            gc.enable()

    # --------------------------------------------------------------------
    # 'DATA': Further processing step
    
    cache_filename = f'{args["datadir"]}/processed_data_{runmode}_{args["__hash_post_genesis__"]}.pkl'
    
    if args['__use_cache__'] == False or (not os.path.exists(cache_filename)):

        # Process it
        processed_data = process_data(args=args, predata=predata, func_factor=func_factor, mvavars=mvavars, runmode=runmode)
        
        with open(cache_filename, 'wb') as handle:
            cprint(__name__ + f'.read_data_processed: Saving <PROCESSED DATA> to a file: "{cache_filename}"', 'yellow')
            
            # Disable garbage collector for speed
            gc.disable()
            pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            gc.enable()
    else:
        with open(cache_filename, 'rb') as handle:
            cprint(__name__ + f'.read_data_processed: Loading <PROCESSED DATA> from a file: "{cache_filename}"', 'yellow')
            
            # Disable garbage collector for speed
            gc.disable()
            processed_data = pickle.load(handle)
            gc.enable()
    
    return processed_data


def process_data(args, predata, func_factor, mvavars, runmode):
    """
    Process data further
    """

    X    = predata['X']
    Y    = predata['Y']
    W    = predata['W']
    ids  = predata['ids']
    info = predata['info']
    
    # ----------------------------------------------------------
    # Pop out conditional variables if they exist and no conditional training is used
    if args['use_conditional'] == False:
        
        if ids is None: # Awkward type, ids within X
            ids_ = X.fields
        else:
            ids_ = ids  # Separate fields
        
        index  = []
        idxvar = []
        for i in range(len(ids_)):
            if 'MODEL_' not in ids_[i]:
                index.append(i)
                idxvar.append(ids_[i])
            else:
                print(__name__ + f'.process_data: Removing conditional variable "{ids_[i]}" (use_conditional == False)')
        
        if   isinstance(X, np.ndarray):
            X = X[:,np.array(index, dtype=int)]
            ids = [ids[j] for j in index]
        
        elif isinstance(X, ak.Array):
            X = X[idxvar]
        
    # ----------------------------------------------------------
    
    # Split into training, validation, test
    trn, val, tst = io.split_data(X=X, Y=Y, W=W, ids=ids, frac=args['frac'])
    
    # ----------------------------------------------------------
    if args['imputation_param']['active']:
        module = import_module(mvavars, 'configs.subpkg')

        var    = args['imputation_param']['var']
        if var is not None:
            impute_vars = getattr(module, var)
        else:
            impute_vars = None
    # ----------------------------------------------------------
    
    ### Split and factor data
    output = {'info': info}

    if   runmode == 'train':

        ### Compute reweighting weights (before funcfactor because we need all the variables !)
        if args['reweight']:
            
            fmodel = args["modeldir"] + '/' + args['reweight_file']
            
            if args['reweight_mode'] == 'load':
                cprint(__name__ + f'.process_data: Loading reweighting model from: {fmodel} (runmode == {runmode})', 'green')
                pdf = pickle.load(open(fmodel, 'rb'))
            else:
                pdf = None # Compute it now
            
            trn.w, pdf = reweight.compute_ND_reweights(pdf=pdf, x=trn.x, y=trn.y, w=trn.w, ids=trn.ids, args=args['reweight_param'])
            val.w,_    = reweight.compute_ND_reweights(pdf=pdf, x=val.x, y=val.y, w=val.w, ids=val.ids, args=args['reweight_param'])
            
            if args['reweight_mode'] == 'write':
                cprint(__name__ + f'.process_data: Saving reweighting model to: {fmodel} (runmode == {runmode})', 'green')
                pickle.dump(pdf, open(fmodel, 'wb'))
        
        # Compute different data representations
        output['trn'] = func_factor(x=trn.x, y=trn.y, w=trn.w, ids=trn.ids, args=args)
        output['val'] = func_factor(x=val.x, y=val.y, w=val.w, ids=val.ids, args=args)
        
        ## Imputate
        if args['imputation_param']['active']:
            
            output['trn']['data'], imputer = impute_datasets(data=output['trn']['data'], features=impute_vars, args=args['imputation_param'], imputer=None)
            output['val']['data'], imputer = impute_datasets(data=output['val']['data'], features=impute_vars, args=args['imputation_param'], imputer=imputer)
            
            outputfile = args["modeldir"] + f'/imputer.pkl'
            cprint(__name__ + f'.process_data: Saving imputer to: {outputfile}', 'green')
            pickle.dump(imputer, open(outputfile, 'wb'))
        
    elif runmode == 'eval':
        
        ### Compute reweighting weights (before func_factor because we need all the variables !)
        if args['reweight']:
            
            fmodel = args["modeldir"] + '/' + args['reweight_file']
            
            if args['reweight_mode'] == 'load':
                cprint(__name__ + f'.process_data: Loading reweighting model from: {fmodel} (runmode = {runmode})', 'green')
                pdf = pickle.load(open(fmodel, 'rb'))
            else:
                pdf = None # Compute it now
            
            tst.w, pdf = reweight.compute_ND_reweights(pdf=pdf, x=tst.x, y=tst.y, w=tst.w, ids=tst.ids, args=args['reweight_param'])
            
            if args['reweight_mode'] == 'write':
                cprint(__name__ + f'.process_data: Saving reweighting model to: {fmodel} (runmode = {runmode})', 'green')
                pickle.dump(pdf, open(fmodel, 'wb'))
        
        # Compute different data representations
        output['tst'] = func_factor(x=tst.x, y=tst.y, w=tst.w, ids=tst.ids, args=args)
        
        ## Imputate
        if args['imputation_param']['active']:
            
            inputfile = args["modeldir"] + f'/imputer.pkl'
            cprint(__name__ + f'.process_data: Loading imputer from: {inputfile}', 'green')
            imputer = pickle.load(open(inputfile, 'rb'))
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
        gc.collect()
        io.showmem()
        
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
                model = train.raytune_main(inputs=inputs, train_func=iceboost.train_xgb)
            else:
                model = iceboost.train_xgb(**inputs)

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

            inputs = {'data_trn':    data_trn['data'],
                      'data_val':    data_val['data'],
                      'args':        args,
                      'param':       param}
            
            if ID in args['raytune']['param']['active']:
                model = train.raytune_main(inputs=inputs, train_func=train.train_cutset)
            else:
                model = train.train_cutset(**inputs)

        else:
            raise Exception(__name__ + f'.Unknown param["train"] = {param["train"]} for ID = {ID}')

        # --------------------------------------------------------
        # If distillation
        if ID == args['distillation']['source']:
            
            if len(args['primary_classes']) != 2:
                raise Exception(__name__ + f'.train_models: Distillation supported now only for 2-class classification')
            
            cprint(__name__ + f'.train.models: Computing distillation soft targets from the source <{ID}> ', 'yellow')

            if   param['train'] == 'xgb':
                XX, XX_ids = aux.red(data_trn['data'].x, data_trn['data'].ids, param)
                y_soft = model.predict(xgboost.DMatrix(data=XX, feature_names=XX_ids))
                if len(y_soft.shape) > 1: y_soft = y_soft[:, args['signalclass']]
            
            elif param['train'] == 'torch_graph':
                y_soft = model.softpredict(data_trn['data_graph'])[:, args['signalclass']]
            else:
                raise Exception(__name__ + f".train_models: Unsupported distillation source <{param['train']}>")
        # --------------------------------------------------------

    cprint(__name__ + f'.train_models: [done]', 'yellow')
    
    return


def evaluate_models(data=None, info=None, args=None):
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
    global roc_paths
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

    ROC_binned_mstats = {}
    ROC_binned_mlabel = {}

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

        X       = copy.deepcopy(data['data'].x)
        ids     = copy.deepcopy(data['data'].ids)

        # These will be left untouched by e.g. z-score normalization
        X_RAW   = copy.deepcopy(data['data'].x)
        ids_RAW = copy.deepcopy(data['data'].ids)

        # Add extra variables to the end if not already there
        # (certain plotting routines pull them from X_RAW)
        if data['data_kin'] is not None:
            for var in data['data_kin'].ids:
                
                if var not in ids_RAW:
                    ids_RAW = ids_RAW + [var]
                    
                    index   = np.where(np.isin(data['data_kin'].ids, var))[0]
                    X_RAW   = np.concatenate([X_RAW, data['data_kin'].x[:,index]], axis=1)
        
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
            func_predict = predict.pred_xgb(args=args, param=param, feature_names=aux.red(X,ids,param,'ids'))
            if args['plot_param']['contours']['active']:
                plots.plot_contour_grid(pred_func=func_predict, X=aux.red(X,ids,param,'X'), y=y, ids=aux.red(X,ids,param,'ids'), transform='numpy', 
                    targetdir=aux.makedir(f'{args["plotdir"]}/eval/2D-contours/{param["label"]}/'))
            
            plot_XYZ_wrap(func_predict = func_predict, x_input=aux.red(X,ids,param,'X'), y=y, **inputs)

        elif param['predict'] == 'xgb_logistic':
            func_predict = predict.pred_xgb_logistic(args=args, param=param, feature_names=aux.red(X,ids,param,'ids'))
            
            if args['plot_param']['contours']['active']:
                plots.plot_contour_grid(pred_func=func_predict, X=aux.red(X,ids,param,'X'), y=y, ids=aux.red(X,ids,param,'ids'), transform='numpy', 
                    targetdir=aux.makedir(f'{args["plotdir"]}/eval/2D-contours/{param["label"]}/'))
            
            plot_XYZ_wrap(func_predict = func_predict, x_input=aux.red(X,ids,param,'X'), y=y, **inputs)

        elif param['predict'] == 'torch_vector':
            func_predict = predict.pred_torch_generic(args=args, param=param)

            if args['plot_param']['contours']['active']:
                plots.plot_contour_grid(pred_func=func_predict, X=aux.red(X_ptr,ids,param,'X'), y=y, ids=aux.red(X_ptr,ids,param,'ids'),
                    targetdir=aux.makedir(f'{args["plotdir"]}/eval/2D-contours/{param["label"]}/'), transform='torch')

            plot_XYZ_wrap(func_predict = func_predict, x_input=aux.red(X_ptr,ids,param,'X'), y=y, **inputs)

        elif param['predict'] == 'torch_scalar':
            func_predict = predict.pred_torch_scalar(args=args, param=param)

            if args['plot_param']['contours']['active']:
                plots.plot_contour_grid(pred_func=func_predict, X=aux.red(X_ptr,ids,param,'X'), y=y, ids=aux.red(X_ptr,ids,param,'ids'), transform='torch', 
                    targetdir=aux.makedir(f'{args["plotdir"]}/eval/2D-contours/{param["label"]}/'))

            plot_XYZ_wrap(func_predict = func_predict, x_input=aux.red(X_ptr,ids,param,'X'), y=y, **inputs)

        elif param['predict'] == 'torch_flow':
            func_predict = predict.pred_flow(args=args, param=param, n_dims=X_ptr.shape[1])

            if args['plot_param']['contours']['active']:
                plots.plot_contour_grid(pred_func=func_predict, X=aux.red(X_ptr,ids,param,'X'), y=y, ids=aux.red(X_ptr,ids,param,'ids'), transform='torch', 
                    targetdir=aux.makedir(f'{args["plotdir"]}/eval/2D-contours/{param["label"]}/'))

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
                    targetdir=aux.makedir(f'{args["plotdir"]}/eval/2D-contours/{param["label"]}/'))
        else:
            raise Exception(__name__ + f'.Unknown param["predict"] = {param["predict"]} for ID = {ID}')

    ## Multiple model comparisons
    plot_XYZ_multiple_models(targetdir=targetdir, args=args)

    ## Pickle results to output

    resdict = {'roc_mstats':        roc_mstats,
               'roc_labels':        roc_labels,
               'roc_paths':         roc_paths,
               'corr_mstats':       corr_mstats,
               'ROC_binned_mstats': ROC_binned_mstats,
               'ROC_binned_mlabel': ROC_binned_mlabel,
               'info':              info}
    
    targetfile = targetdir + '/eval_results.pkl'
    print(__name__ + f'.evaluate_models: Saving pickle output to "{targetfile}"')
    pickle.dump(resdict, open(targetfile, 'wb'))
    return


def make_plots(data, args):
    """
    Basic Q/A-plots
    """
    
    ### Plot variables
    if args['plot_param']['basic']['active']:
        
        param = copy.deepcopy(args['plot_param']['basic'])
        param.pop('active')
        
        ### Specific variables
        if data['data_kin'] is not None:
            targetdir = aux.makedir(f'{args["plotdir"]}/reweight/1D-kinematic/')
            plots.plotvars(X = data['data_kin'].x, y = data['data_kin'].y, weights = data['data_kin'].w, ids = data['data_kin'].ids,           
                targetdir=targetdir, title=f"training re-weight reference class: {args['reweight_param']['reference_class']}",
                **param)
        
        ### Plot MVA input variable plots
        targetdir = aux.makedir(f'{args["plotdir"]}/train/1D-distributions/')
        plots.plotvars(X = data['data'].x, y = data['data'].y, weights = data['data'].w,  ids = data['data'].ids,
            targetdir=targetdir, title=f"training re-weight reference class: {args['reweight_param']['reference_class']}",
            **param)
    
    ### Correlations
    if args['plot_param']['corrmat']['active']:

        targetdir = aux.makedir(f'{args["plotdir"]}/train/')
        fig,ax    = plots.plot_correlations(X=data['data'].x, weights=data['data'].w, ids=data['data'].ids, y=data['data'].y, targetdir=targetdir)


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
            metric = aux.Metric(y_true=y[mask], y_pred=y_pred[mask],
                weights=weights[mask] if weights is not None else None,
                num_bootstrap=args['plot_param']['ROC']['num_bootstrap'])

            if sublabel not in roc_mstats:
                roc_mstats[sublabel] = []
                roc_labels[sublabel] = []
                roc_paths[sublabel]  = []

            roc_mstats[sublabel].append(metric)
            roc_labels[sublabel].append(label)
            roc_paths[sublabel].append(pathlabel)
        
        # ** All inclusive **
        mask = np.ones(len(y_pred), dtype=bool)
        plot_helper(mask=mask, sublabel='inclusive', pathlabel='inclusive')

        # ** Powerset filtered **
        if 'set_filter' in args['plot_param']['ROC']:

            filters = args['plot_param']['ROC']['set_filter']
            mask_powerset, text_powerset, path_powerset = stx.filter_constructor(filters=filters, X=X_RAW, ids=ids_RAW)

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
            if len(var) == 1:

                met_1D, label_1D = plots.binned_1D_AUC(y_pred=y_pred, y=y, weights=weights, X_kin=X_kin, \
                    ids_kin=VARS_kin, X=X_RAW, ids=ids_RAW, edges=edges, VAR=var[0], \
                    num_bootstrap=args['plot_param']['ROC_binned']['num_bootstrap'])

                if label not in ROC_binned_mstats: # not yet created
                    ROC_binned_mstats[label] = {}
                    ROC_binned_mlabel[label] = {}
                
                # Save for multiple comparison
                ROC_binned_mstats[label][i] = met_1D
                ROC_binned_mlabel[label][i] = label_1D

                # Plot this one
                plots.ROC_plot(met_1D, label_1D, xmin=args['plot_param']['ROC_binned']['xmin'], title = f'{label}', filename=aux.makedir(f'{targetdir}/ROC/{label}') + f'/ROC-binned[{i}]')
                plots.MVA_plot(met_1D, label_1D, title = f'{label}', filename=aux.makedir(f'{targetdir}/MVA/{label}') + f'/MVA-binned[{i}]')

            ## 2D
            elif len(var) == 2:
                fig, ax, met = plots.binned_2D_AUC(y_pred=y_pred, y=y, weights=weights, X_kin=X_kin, \
                    ids_kin=VARS_kin, X=X_RAW, ids=ids_RAW, edges=edges, label=label, VAR=var)
                plt.savefig(aux.makedir(f'{targetdir}/ROC/{label}') + f'/ROC-binned[{i}].pdf', bbox_inches='tight')

            else:
                print(var)
                raise Exception(__name__ + f'.plot_AUC_wrap: Unknown dimensionality {len(var)}')

    # ----------------------------------------------------------------
    ### MVA-output 1D-plot
    if args['plot_param']['MVA_output']['active']:

        def plot_helper(mask, sublabel, pathlabel):
            hist_edges = args['plot_param'][f'MVA_output']['edges']
            inputs = {'y_pred': y_pred[mask], 'y': y[mask],
                'weights': weights[mask] if weights is not None else None, 'class_ids': None,
                'hist_edges': hist_edges, 'label': f'{label}/{sublabel}', 'path': f'{targetdir}/MVA/{label}/{pathlabel}'}

            plots.density_MVA_wclass(**inputs)

        # ** All inclusive **
        mask  = np.ones(len(y_pred), dtype=bool)
        plot_helper(mask=mask, sublabel='inclusive', pathlabel='inclusive')

        # ** Set filtered **
        if 'set_filter' in args['plot_param']['MVA_output']:

            filters = args['plot_param']['MVA_output']['set_filter']
            mask_powerset, text_powerset, path_powerset = stx.filter_constructor(filters=filters, X=X_RAW, ids=ids_RAW)

            for m in range(mask_powerset.shape[0]):
                plot_helper(mask=mask_powerset[m,:], sublabel=text_powerset[m], pathlabel=path_powerset[m])

    # ----------------------------------------------------------------
    ### MVA-output 2D correlation plots
    if args['plot_param']['MVA_2D']['active']:

        for i in range(100): # Loop over plot types
            
            pid = f'plot[{i}]'
            if pid in args['plot_param']['MVA_2D']:        
                var   = args['plot_param']['MVA_2D'][pid]['var']
                edges = args['plot_param']['MVA_2D'][pid]['edges']
            else:
                break # No more this type of plots
            
            def plot_helper(mask, pick_ind, sublabel='inclusive', pathlabel='inclusive', savestats=False):

                # Two step
                XX = X_RAW[mask, ...]
                XX = XX[:, pick_ind]
                
                inputs = {
                    'y_pred':      y_pred[mask],
                    'weights':     weights[mask] if weights is not None else None,
                    'X':           XX,
                    'ids':         np.array(ids_RAW, dtype=np.object_)[pick_ind].tolist(),
                    'class_ids':   None,
                    'label':       f'{label}/{sublabel}', 'hist_edges': edges,
                    'path':        f'{targetdir}/COR/{label}/{pathlabel}'}
                
                output = plots.density_COR_wclass(y=y[mask], **inputs)
                #plots.density_COR(**inputs)

                # Save output
                if i not in corr_mstats.keys():
                    corr_mstats[i] = {}
                if label not in corr_mstats[i].keys():    
                    corr_mstats[i][label] = {}
                
                corr_mstats[i][label][sublabel] = output

            # Pick chosen variables based on regular expressions
            var_names = aux.process_regexp_ids(all_ids=ids_RAW, ids=var)
            pick_ind  = np.array(np.where(np.isin(ids_RAW, var_names))[0], dtype=int)
            
            # ** All inclusive **
            mask      = np.ones(len(y_pred), dtype=bool)

            plot_helper(mask=mask, pick_ind=pick_ind, sublabel='inclusive', pathlabel='inclusive')
            
            # ** Set filtered **
            if 'set_filter' in args['plot_param']['MVA_2D'][pid]:
                
                filters = args['plot_param']['MVA_2D'][pid]['set_filter']
                mask_powerset, text_powerset, path_powerset = stx.filter_constructor(filters=filters, X=X_RAW, ids=ids_RAW)

                for m in range(mask_powerset.shape[0]):
                    plot_helper(mask=mask_powerset[m,:], pick_ind=pick_ind, sublabel=text_powerset[m], pathlabel=path_powerset[m])
            
    return True


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
                
                try: # if user provided xlim
                    xlim = args['plot_param']['MVA_2D'][pid]['xlim']
                except:
                    xlim = None
                
                plots.plot_correlation_comparison(corr_mstats=corr_mstats[i], targetdir=targetdir, xlim=xlim)

            else:
                break

    # ===================================================================
    # ** Plots for multiple model comparison **

    # -------------------------------------------------------------------
    ### Plot all ROC curves

    pprint(roc_mstats)
    
    # We have the same number of filterset (category) entries for each model, pick the first
    dummy = 0

    # Direct collect:  Plot all models per powerset category
    for filterset_key in roc_mstats.keys():

        path_label = roc_paths[filterset_key][dummy]
        plots.ROC_plot(roc_mstats[filterset_key], roc_labels[filterset_key],
            xmin=args['plot_param']['ROC']['xmin'],
            title=f'category: {filterset_key}', filename=aux.makedir(targetdir + f'/ROC/--ALL--/{path_label}') + '/ROC-all-models')

    # Inverse collect: Plot all powerset categories ROCs per model
    for model_index in range(len(roc_mstats[list(roc_mstats)[dummy]])):
        
        rocs_       = [roc_mstats[filterset_key][model_index] for filterset_key in roc_mstats.keys()]
        labels_     = list(roc_mstats.keys())
        model_label = roc_labels[list(roc_labels)[dummy]][model_index]
        
        plots.ROC_plot(rocs_, labels_,
            xmin=args['plot_param']['ROC']['xmin'],
            title=f'model: {model_label}', filename=aux.makedir(targetdir + f'/ROC/{model_label}') + '/ROC-all-categories')

    ### Plot all MVA outputs (not implemented)
    #plots.MVA_plot(mva_mstats, mva_labels, title = '', filename=aux.makedir(targetdir + '/MVA/--ALL--') + '/MVA')

    ### Plot all 1D-binned ROC curves
    if args['plot_param']['ROC_binned']['active']:

        for i in range(100):
            pid = f'plot[{i}]'

            if pid in args['plot_param']['ROC_binned']:
                var   = args['plot_param']['ROC_binned'][pid]['var']
                edges = args['plot_param']['ROC_binned'][pid]['edges']
            else:
                break # No more plots 
            
            if len(var) == 1: # 1D
                
                # Over different bins
                for b in range(len(edges)-1):
                    
                    # Over different models
                    xy,legs = [],[]
                    for k in range(len(ROC_binned_mstats)):
                        
                        # Take label for the legend
                        ID    = args['active_models'][k]
                        label = args['models'][ID]['label']
                        
                        legs.append(label)
                        xy.append(ROC_binned_mstats[label][i][b])
                    
                    ### ROC
                    title = f'BINNED ROC: {var[0]}$ \\in [{edges[b]:0.1f}, {edges[b+1]:0.1f})$'
                    plots.ROC_plot(xy, legs, xmin=args['plot_param']['ROC_binned']['xmin'], title=title, filename=targetdir + f'/ROC/--ALL--/ROC-binned[{i}]-bin[{b}]')
                    
                    ### MVA (not implemented)
                    #title = f'BINNED MVA: {var[0]}$ \\in [{edges[b]:0.1f}, {edges[b+1]:0.1f})$'
                    #plots.MVA_plot(xy, legs, title=title, filename=targetdir + f'/MVA/--ALL--/MVA-binned[{i}]-bin[{b}]')

    return True
