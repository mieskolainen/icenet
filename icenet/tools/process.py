# Common input & data reading routines
#
# m.mieskolainen@imperial.ac.uk, 2024

import argparse
import numpy as np
import awkward as ak
import gc
import torch
import torch_geometric
import socket
import copy
import glob
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

from importlib import import_module
import os
import pickle
import xgboost

from yamlinclude import YamlIncludeConstructor

import icenet.deep.iceboost as iceboost
import icenet.deep.train as train
import icenet.deep.predict as predict

from icenet.tools import iceprint, stx, io, prints, aux, reweight, plots, supertune

# ------------------------------------------
from icenet import print
# ------------------------------------------


# ******** GLOBALS *********
from icenet import LOGGER, icelogger

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
    parser.add_argument("--runmode",           type=str,  default='all')
    parser.add_argument("--config",            type=str,  default='tune0.yml')
    parser.add_argument("--datapath",          type=str,  default='')
    parser.add_argument("--datasets",          type=str,  default='')
    
    parser.add_argument("--maxevents",         type=int,  default=argparse.SUPPRESS) # GLOBAL POWER CONTROL
    parser.add_argument("--use_conditional",   type=int,  default=argparse.SUPPRESS) # GLOBAL POWER CONTROL
    
    parser.add_argument("--compute",           type=int,  default=1)    # allow to skip train/eval computations
    parser.add_argument("--use_cache",         type=int,  default=1)
    parser.add_argument("--fastplot",          type=int,  default=0)
    
    parser.add_argument("--hash_genesis",      type=str,  default=None) # override control
    parser.add_argument("--hash_post_genesis", type=str,  default=None)
    
    parser.add_argument("--grid_id",           type=int,  default=0)    # Condor/Oracle execution variables
    parser.add_argument("--grid_nodes",        type=int,  default=1)    # Condor/Oracle
    
    parser.add_argument("--inputmap",          type=str,  default=None)
    parser.add_argument("--modeltag",          type=str,  default=None) # Use this for multiple parallel runs
    parser.add_argument("--evaltag",           type=str,  default=None) # Use this for custom output plot directory for evaluation
    
    parser.add_argument("--run_id",            type=str,  default='latest')
    
    parser.add_argument("--num_cpus",          type=int,  default=0)    # Fixed number of CPUs
    parser.add_argument("--supertune",         type=str,  default=None) # Generic cli override
    
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
    with open(os.path.join(cwd, config_path, config_yaml_file), 'r') as f:
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

        with open(os.path.join(cwd, config_path, file), 'r') as f:
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
    new_args['num_cpus']  = args['num_cpus']
    new_args['inputvars'] = args['inputvars']
    
    # -----------------------------------------------------
    # Runmode setup
    
    print(f'runmode = "{runmode}"', 'magenta')
    
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
    
    # Fast plot mode
    if cli_dict['fastplot']:
        new_args['plot_param']['basic']['active']      = False
        new_args['plot_param']['corrmat']['active']    = False
        new_args['plot_param']['contours']['active']   = False
        new_args['plot_param']['ROC_binned']['active'] = False
        new_args['plot_param']['MVA_2D']['active']     = False
        
        print(f'fastplot mode on (turning off slow plots)')
    
    old_args = copy.deepcopy(args)
    args     = copy.deepcopy(new_args)

    # -------------------------------------------------------------------
    ## Specific commandline override of yaml variables
    for key in cli_dict.keys():
        if key in args:
            print(f'{config_yaml_file} "{key}" default value cli-override with value {cli_dict[key]}', 'green')
            args[key] = cli_dict[key]
    
    # -------------------------------------------------------------------
    ## Supertune generic commandline override of yaml (dictionary) content
    
    if cli_dict['supertune'] is not None:
        print('')
        print(f'{config_yaml_file} default value cli-override with --supertune syntax:', 'green')
        print('')
        args = supertune.supertune(d=args, config_string=cli_dict['supertune'])
        print('')
    
    # -------------------------------------------------------------------
    ## 1. Create the first level hash
    
    hash_args = {}

    # Critical Python files content
    files = {'cuts':      os.path.join(cwd, config_path, 'cuts.py'),
             'filter':    os.path.join(cwd, config_path, 'filter.py'),
             'inputvars': os.path.join(cwd, config_path, f'{args["inputvars"]}.py')}
    
    for key in files.keys():
        if os.path.exists(files[key]):
            hash_args[f'__hash__{key}'] = io.make_hash_sha256_file(files[key])
        else:
            print(f"Did not find: {files[key]} [may cause crash]", 'red')
    
    # Genesis parameters as the first one
    hash_args.update(old_args['genesis_runmode'])
    
    # These are universal and need to be hashed
    hash_args['rngseed']   = args['rngseed']
    hash_args['maxevents'] = args['maxevents']
    hash_args['inputvars'] = args['inputvars']
    hash_args.update(inputmap)
    
    # Finally create the hash
    args['__hash_genesis__'] = io.make_hash_sha256_object(hash_args)
    
    print(f'Generated config hashes', 'magenta')
    print(f'[__hash_genesis__]      : {args["__hash_genesis__"]}     ', 'magenta')
    
    # -------------------------------------------------------------------
    ## 2. Create the second level hash (depends on all previous) + other parameters
    
    if runmode == 'train' or runmode == 'eval':
        
        # First include all
        hash_args.update(args)
        
        # Then exclude non-input dependent
        hash_args.pop("plot_param")
        hash_args.pop("modeltag")
        hash_args.pop("models")
        hash_args.pop("active_models")
        
        if runmode == 'train':
            try:
                hash_args.pop("outlier_param")
            except:
                True
            try:
                hash_args.pop("raytune")
            except:
                True
            try:
                hash_args.pop("distillation")
            except:
                True
            try:
                hash_args.pop("batch_train_param")
            except:
                True

        # Finally create hash
        args['__hash_post_genesis__'] = args['__hash_genesis__'] + '__' + io.make_hash_sha256_object(hash_args)

        print(f'[__hash_post_genesis__] : {args["__hash_post_genesis__"]}', 'magenta')

    # -------------------------------------------------------------------
    ## Update variables to args dictionary (and create directories)

    args["config"]  = cli_dict['config']
    args['datadir'] = aux.makedir(f'{cwd}/output/{args["rootname"]}')
    
    if runmode != 'genesis':
        
        args['modeldir'] = aux.makedir(f'{cwd}/checkpoint/{args["rootname"]}/config__{io.safetxt(args["config"])}/modeltag__{args["modeltag"]}')
        args['plotdir']  = aux.makedir(f'{cwd}/figs/{args["rootname"]}/config__{io.safetxt(args["config"])}/inputmap__{io.safetxt(cli_dict["inputmap"])}--modeltag__{args["modeltag"]}')
        
        # Add conditional tag
        conditional_tag   = f'--use_conditional__{args["use_conditional"]}' if args["use_conditional"] else ""
        args['modeldir'] += conditional_tag
        args['plotdir']  += conditional_tag
        
        # Add runtime and hostname tag
        run_id     = f'{aux.get_datetime()}_{socket.gethostname().split(".")[0]}'
        run_id_now = copy.deepcopy(run_id)
        
        if runmode == 'train':
            
            if cli_dict['run_id'] != 'latest':
                run_id = cli_dict['run_id']
        
        elif runmode == 'eval' or runmode == 'deploy':
            
            if cli_dict['run_id'] == 'latest':
                
                # Find the latest training
                list_of_files = glob.glob(f"{args['modeldir']}/*")

                if len(list_of_files) == 0:
                    raise Exception(__name__ + f'.read_config: Could not find any trained models -- run training first')

                run_id = max(list_of_files, key=os.path.getctime).split('/')[-1]

            # Use specified run_id
            else:
                run_id = cli_dict['run_id']

        elif runmode == 'optimize':
            
            if cli_dict['run_id'] == 'latest':
                
                # Find the latest evaluation
                list_of_files = glob.glob(f"{args['plotdir']}/*")

                if len(list_of_files) == 0:
                    raise Exception(__name__ + f'.read_config: Could not find any evaluation run -- run evaluation first')

                run_id = max(list_of_files, key=os.path.getctime).split('/')[-1]

            # Use specified run_id
            else:
                run_id = cli_dict['run_id']
        
        # Store it
        args['run_id']     = run_id
        args['run_id_now'] = run_id_now
        
        ## ** Create and set folders **
        args['modeldir'] = aux.makedir(f"{args['modeldir']}/{run_id}")
        args['plotdir']  = aux.makedir(f"{args['plotdir']}/{run_id}")
        
        if runmode == 'eval' and cli_dict['evaltag'] is not None:
            args['plotdir'] = aux.makedir(f"{args['plotdir']}/evaltag__{cli_dict['evaltag']}")
            print(f'Changing eval plotdir to: {args["plotdir"]}', 'red')
        
        # ----------------------------------------------------------------
        ## Save args to yaml as a checkpoint of the run configuration
        dir = aux.makedir(f'{args["plotdir"]}/{runmode}')
        aux.yaml_dump(data=args, filename=f'{dir}/args.yml')
        # ----------------------------------------------------------------
    
    # "Simplified" data reader
    args['root_files'] = io.glob_expand_files(datasets=cli.datasets, datapath=cli.datapath)    
    
    
    # -------------------------------------------------------------------
    # Technical
    
    args['__runmode__']         = cli_dict['runmode']
    args['__use_cache__']       = bool(cli_dict['use_cache'])
    args['__compute__']         = bool(cli_dict['compute'])
    args['__raytune_running__'] = False

    # Override hashes
    if cli_dict['hash_genesis'] is not None:
        print(f"Override 'hash_genesis' with: {cli_dict['hash_genesis']}", 'red')
        args['__hash_genesis__'] = cli_dict['hash_genesis']
    
    if cli_dict['hash_post_genesis'] is not None:
        print(f"Override 'hash_post_genesis' with: {cli_dict['hash_post_genesis']}", 'red')
        args['__hash_post_genesis__'] = cli_dict['hash_post_genesis']
    
    # Distributed computing
    for key in ['grid_id', 'grid_nodes']:
        args[key] = cli_dict[key]
    
    
    # -------------------------------------------------------------------
    ## Create aux dirs
    aux.makedir('tmp')
    # -------------------------------------------------------------------
    
    # -------------------------------------------------------------------
    # Set random seeds for reproducability and train-validate-test splits
    
    print('')
    print(" torch.__version__: " + torch.__version__)

    print(f'Setting random seed', 'yellow')
    aux.set_random_seed(args['rngseed'])
    
    # ------------------------------------------------
    print(f'Created arguments dictionary with runmode = <{runmode}> :')    
    # ------------------------------------------------
    
    return args, cli


def generic_flow(rootname, func_loader, func_factor):
    """
    Generic (read data -- train models -- evaluate models) workflow
    
    Args:
        rootname:     name of the workflow config folder
        func_loader:  data loader (function handle)
        func_factor:  data transformer (function handle)
    """
    cli, cli_dict = read_cli()
    runmode       = cli_dict['runmode']
    
    args, cli     = read_config(config_path=f'configs/{rootname}', runmode=runmode)

    try:
        
        if runmode == 'genesis':
            
            icelogger.set_global_log_file(f'{args["datadir"]}/genesis_{args["__hash_genesis__"]}.log')
            print(cli) # for output log
            process_raw_data(args=args, func_loader=func_loader) 
            
        if runmode == 'train' or runmode == 'eval':
            
            icelogger.set_global_log_file(f'{args["plotdir"]}/{runmode}/execution.log')
            print(cli) # for output log
            data = train_eval_data_processor(args=args,
                    func_factor=func_factor, mvavars=f'configs.{rootname}.mvavars', runmode=runmode)
            
        if runmode == 'train':

            output_file = f'{args["plotdir"]}/train/stats_train.log'
            prints.print_variables(X=data['trn']['data'].x, W=data['trn']['data'].w, ids=data['trn']['data'].ids, output_file=output_file)
            make_plots(data=data['trn'], args=args, runmode=runmode)
            
            if args['__compute__']:
                train_models(data_trn=data['trn'], data_val=data['val'], args=args)

        if runmode == 'eval':
            
            output_file = f'{args["plotdir"]}/eval/stats_evaluate.log'
            prints.print_variables(X=data['tst']['data'].x, W=data['tst']['data'].w, ids=data['tst']['data'].ids, output_file=output_file)        
            make_plots(data=data['tst'], args=args, runmode=runmode)
            
            if args['__compute__']:
                evaluate_models(data=data['tst'], info=data['info'], args=args)

    except Exception as e:
        print(e)
        raise Exception(e)
    
    return args, runmode

# -------------------------------------------------------------------

def concatenate_data(data, max_batch_size: int=32):
    """
    Helper function to concatenate arrays with a specified maximum batch size
    """
    X_all, Y_all, W_all = [], [], []
    
    print('Appending arrays to lists ...')
    tic = time.time()
    
    N = 0
    for X_, Y_, W_ in tqdm(data):
        
        X_all.append(X_)
        Y_all.append(Y_)
        W_all.append(W_)

        N += len(X_)
    
    toc = time.time() - tic
    print(f'Appending took {toc:0.2f} sec')
    
    print('Executing array concatenation ...')    
    tic = time.time()
    
    X = aux.recursive_concatenate(X_all, max_batch_size=max_batch_size, axis=0)
    Y = aux.recursive_concatenate(Y_all, max_batch_size=max_batch_size, axis=0)
    W = aux.recursive_concatenate(W_all, max_batch_size=max_batch_size, axis=0)
    
    N_final = len(X)

    if N_final != N:
        msg = f'Error with N ({N}) != N_final ({N_final})'
        print(msg)
        raise Exception(f'Error occured in concatenation: {msg}')
    
    toc = time.time() - tic
    print(f'Concatenation took {toc:0.2f} sec')

    return X, Y, W


def load_file_wrapper(index, filepath):
    """
    Helper function
    """
    with open(filepath, 'rb') as handle:
        return index, pickle.load(handle)

# -------------------------------------------------------------------

@iceprint.icelog(LOGGER)
def process_raw_data(args, func_loader):
    """
    Load raw input from the disk -- this is executed only by 'genesis'
    """
    
    num_cpus = args['num_cpus']
    
    def get_chunk_ind(N):
        chunks = int(np.ceil(N / args['pickle_size']))
        return aux.split_start_end(range(N), chunks)

    cache_directory = aux.makedir(os.path.join(args["datadir"], f'data__{args["__hash_genesis__"]}'))

    # Check do we have already computed pickles ready
    if (os.path.exists(os.path.join(cache_directory, 'output_0.pkl')) and args['__use_cache__']):
        print(f'Found existing pickle data under: {cache_directory} and --use_cache 1. [done] ', 'green')
        return
    
    # func_loader does the multifile processing
    load_args = {'entry_start': 0, 'entry_stop': None, 'maxevents': args['maxevents'], 'args': args}
    data      = func_loader(root_path=args['root_files'], **load_args)
    
    X    = data['X']
    Y    = data['Y']
    W    = data['W']
    ids  = data['ids']
    info = data['info']

    C = get_chunk_ind(N=len(X))
    
    print(f'Saving {len(C)} pickle files to path: "{cache_directory}"', 'yellow')

    ## New code
    
    def save_pickle(i):
        with open(os.path.join(cache_directory, f'output_{i}.pkl'), 'wb') as handle:
            pickle.dump([X[C[i][0]:C[i][-1]], Y[C[i][0]:C[i][-1]], W[C[i][0]:C[i][-1]], ids, info, args], 
                        handle, protocol=pickle.HIGHEST_PROTOCOL)

    tic = time.time()
    
    # Create a thread pool
    max_workers = multiprocessing.cpu_count() // 2 if num_cpus == 0 else num_cpus
    max_workers = min(len(C), max_workers)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map the save_pickle function to each index in the range
        list(tqdm(executor.map(save_pickle, range(len(C))), total=len(C)))

    toc = time.time() - tic
    print(f'Saving took {toc:0.2f} sec')
    
    # Save args
    aux.yaml_dump(data=args, filename=os.path.join(args["datadir"], f'data__{args["__hash_genesis__"]}.yml'))
    
    """
    # OLD code
    
    tic = time.time()
    
    for i in tqdm(range(len(C))):
        with open(os.path.join(cache_directory, f'output_{i}.pkl'), 'wb') as handle:
            pickle.dump([X[C[i][0]:C[i][-1]], Y[C[i][0]:C[i][-1]], W[C[i][0]:C[i][-1]], ids, info, args], \
                handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    toc = time.time() - tic
    print(f'Saving took {toc:0.2f} sec')
    """
    
    gc.collect()
    io.showmem()
    
    print('[done]')
    
    return

@iceprint.icelog(LOGGER)
def combine_pickle_data(args):
    """
    Load splitted pickle data and return full dataset arrays
    
    Args:
        args: main argument dictionary
    """
    
    num_cpus = args['num_cpus']
    
    cache_directory = aux.makedir(os.path.join(args["datadir"], f'data__{args["__hash_genesis__"]}'))
    
    if not os.path.exists(os.path.join(cache_directory, 'output_0.pkl')):
        raise Exception(__name__ + f'.process_pickle_data: No genesis stage pickle data under "{cache_directory}" [execute --runmode genesis and set --maxevents N]')
    
    ## New version
    
    """    
    Using ThreadPool, not fully parallel because of GIL (Global Interpreter Lock), but
    should keep memory in control (vs. ProcessPool uses processes, but memory can be a problem)
    """
    
    files        = os.listdir(cache_directory)
    sorted_files = sorted(files, key=lambda x: int(os.path.splitext(x)[0].split('_')[1]))
    
    filepaths = [os.path.join(cache_directory, f) for f in sorted_files]
    num_files = len(filepaths)
    
    print(f'Loading {num_files} pickle files from path: "{cache_directory}"')
    print('')
    print(sorted_files)
    print('')
    
    data = [None] * num_files
    
    tic = time.time()
    
    max_workers = multiprocessing.cpu_count() // 2 if num_cpus == 0 else num_cpus
    max_workers = min(num_files, max_workers)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(load_file_wrapper, i, fp): i for i, fp in enumerate(filepaths)}
        for future in tqdm(as_completed(future_to_index), total=num_files):
            try:
                index, (X_, Y_, W_, ids, info, genesis_args) = future.result()
                data[index] = (X_, Y_, W_)
            except Exception as e:
                msg = f'Error loading file: {filepaths[future_to_index[future]]} -- {e}'
                raise Exception(msg)
            
            finally:
                del future  # Ensure the future is deleted to free memory
    
    toc = time.time() - tic
    print(f'Loading took {toc:0.2f} sec')
    
    X, Y, W = concatenate_data(data=data, max_batch_size=args['tech']['concat_max_pickle'])
    gc.collect()  # Call garbage collection once after the loop
    
    """
    ## Old version
    
    num_files = io.count_files_in_dir(cache_directory)
    print(f'Loading from path: "{cache_directory}"', 'yellow')
    
    tic = time.time()
    for i in tqdm(range(num_files)):
        
        with open(os.path.join(cache_directory, f'output_{i}.pkl'), 'rb') as handle:
            X_, Y_, W_, ids, info, genesis_args = pickle.load(handle)

            if i > 0:
                X = np.concatenate((X, X_), axis=0) # awkward will cast numpy automatically
                Y = np.concatenate((Y, Y_), axis=0)
                W = np.concatenate((W, W_), axis=0)
            else:
                X,Y,W = copy.deepcopy(X_), copy.deepcopy(Y_), copy.deepcopy(W_)
            
            gc.collect() # important!
    toc = time.time() - tic
    print(f'Took {toc:0.2f} sec')
    """
    
    io.showmem()
    print('[done]')
    
    return {'X':X, 'Y':Y, 'W':W, 'ids':ids, 'info':info}


@iceprint.icelog(LOGGER)
def train_eval_data_processor(args, func_factor, mvavars, runmode):
    """
    Read/write (MVA) data and return full processed dataset
    """

    # --------------------------------------------------------------------
    # 1. Pickle data combiner step
    
    cache_filename = os.path.join(args["datadir"], f'data__{args["__hash_genesis__"]}.pkl')
    
    if args['__use_cache__'] == False or (not os.path.exists(cache_filename)):

        print(f'File "{cache_filename}" for <DATA> does not exist, creating.', 'yellow')
        
        data = combine_pickle_data(args=args) 
        
        with open(cache_filename, 'wb') as handle:
            print(f'Saving <DATA> to a pickle file: "{cache_filename}"', 'yellow')
            
            # Disable garbage collector for speed
            gc.disable()
            tic = time.time()
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            toc = time.time() - tic
            print(f'Saving took {toc:0.2f} sec')
            gc.enable()
        
        # Save args
        fname = cache_filename.replace('.pkl', '.yml')
        aux.yaml_dump(data=args, filename=fname)
        
    else:
        with open(cache_filename, 'rb') as handle:
            print(f'Loading <DATA> from a pickle file: "{cache_filename}"', 'yellow')
            
            # Disable garbage collector for speed
            gc.disable()
            tic  = time.time()
            data = pickle.load(handle)
            toc  = time.time() - tic
            print(f'Loading took {toc:0.2f} sec')
            gc.enable()
    
    io.showmem()
    print('[done]')
    
    # --------------------------------------------------------------------
    # 2. High level data step
    
    cache_filename = os.path.join(args["datadir"], f'processed_data__{runmode}__{args["__hash_post_genesis__"]}.pkl')
    
    if args['__use_cache__'] == False or (not os.path.exists(cache_filename)):
        
        print(f'File "{cache_filename}" for <PROCESSED DATA> does not exist, creating.', 'yellow')
        
        # Process it
        processed_data = process_data(args=args, data=data, func_factor=func_factor, mvavars=mvavars, runmode=runmode)
        
        with open(cache_filename, 'wb') as handle:
            print(f'Saving <PROCESSED DATA> to a pickle file: "{cache_filename}"', 'yellow')
            
            # Disable garbage collector for speed
            gc.disable()
            tic = time.time()
            pickle.dump(processed_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            toc = time.time() - tic
            print(f'Saving took {toc:0.2f} sec')
            gc.enable()
        
        # Save args
        fname = cache_filename.replace('.pkl', '.yml')
        aux.yaml_dump(data=args, filename=fname)
        
    else:
        with open(cache_filename, 'rb') as handle:
            print(f'Loading <PROCESSED DATA> from a pickle file: "{cache_filename}"', 'yellow')
            
            # Disable garbage collector for speed
            gc.disable()
            tic = time.time()
            processed_data = pickle.load(handle)
            toc = time.time() - tic
            print(f'Loading took {toc:0.2f} sec')
            gc.enable()
    
    gc.collect()
    io.showmem()
    print('[done]')
    
    return processed_data


@iceprint.icelog(LOGGER)
def process_data(args, data, func_factor, mvavars, runmode):
    """
    Process data to high level representations and split to train/eval/test
    """
    
    X    = data['X']
    Y    = data['Y']
    W    = data['W']
    ids  = data['ids']
    info = data['info']
    
    # ----------------------------------------------------------
    # Pop out conditional variables if they exist and no conditional training is used
    if args['use_conditional'] == False:
        
        index  = []
        idxvar = []
        for i in range(len(ids)):
            if 'MODEL_' not in ids[i]:
                index.append(i)
                idxvar.append(ids[i])
            else:
                print(f'Removing conditional variable "{ids[i]}" (use_conditional == False)')
        
        if   isinstance(X, np.ndarray):
            X   = X[:,np.array(index, dtype=int)]
            ids = copy.deepcopy(idxvar)
        
        elif isinstance(X, ak.Array):
            X   = X[idxvar]
            ids = ak.fields(X)
        
        else:
            raise Exception(__name__ + f'.process_data: Unknown X type (should be numpy array or awkward array)')
    
    print(f'ids = {ids}')
    
    # ----------------------------------------------------------
    
    # 1. Split done inside common.py already
    if 'running_split' in info:
        
        print(f'Using pre-defined [train, validate, test] split', 'magenta')
        
        ind_trn = info['running_split']['trn']
        ind_val = info['running_split']['val']
        ind_tst = info['running_split']['tst']
        
        trn = io.IceXYW(x = X[ind_trn,:], y = Y[ind_trn], w=W[ind_trn], ids=ids)
        val = io.IceXYW(x = X[ind_val,:], y = Y[ind_val], w=W[ind_val], ids=ids)
        tst = io.IceXYW(x = X[ind_tst,:], y = Y[ind_tst], w=W[ind_tst], ids=ids)
    
    # 2. Split into training, validation, test here
    else:
        print(f'Splitting into [train, validate, test] = {args["frac"]}', 'magenta')
        
        permute = args['permute'] if 'permute' in args else True
        trn, val, tst = io.split_data(X=X, Y=Y, W=W, ids=ids, frac=args['frac'], permute=permute)
    
    # ----------------------------------------------------------
    # ** Re-permute train and validation events for safety **
    trn = trn.permute(np.random.permutation(len(trn.x)))
    val = val.permute(np.random.permutation(len(val.x)))
    # ----------------------------------------------------------    
    
    # ----------------------------------------------------------
    if args['imputation_param']['active']:
        module = import_module(mvavars, 'configs.subpkg')

        var    = args['imputation_param']['var']
        if var is not None:
            impute_vars = getattr(module, var)
        else:
            impute_vars = None # All chosen
    # ----------------------------------------------------------
    
    ### Split and factor data
    output = {'info': info}
    
    if runmode == 'train':
        
        ### Compute reweighting weights (before funcfactor because we need all the variables !)
        if args['reweight']:
            
            if args["reweight_file"] is None:
                fmodel = os.path.join(args["datadir"], f'reweighter__{args["__hash_genesis__"]}.pkl') 
            else:
                fmodel = os.path.join(args["datadir"], args["reweight_file"])
            
            if 'load' in args['reweight_mode']:
                print(f'Loading reweighting model from: {fmodel} [runmode = {runmode}]', 'green')
                pdf = pickle.load(open(fmodel, 'rb'))
            else:
                pdf = None # Compute it now
            
            # -----------------------
            # ** Special mode **
            skip_reweights = True if 'skip' in args['reweight_mode'] else False
            # -----------------------
            
            trn.w, pdf = reweight.compute_ND_reweights(pdf=pdf, x=trn.x, y=trn.y, w=trn.w, ids=trn.ids, args=args, x_val=val.x, y_val=val.y, w_val=val.w, skip_reweights=skip_reweights)
            val.w, _   = reweight.compute_ND_reweights(pdf=pdf, x=val.x, y=val.y, w=val.w, ids=val.ids, args=args, skip_reweights=skip_reweights)
            
            if 'write' in args['reweight_mode']:
                print(f'Saving reweighting model to: {fmodel} [runmode = {runmode}]', 'green')
                pickle.dump(pdf, open(fmodel, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
        # Compute different data representations
        print(f'Compute representations [func_factor]', 'green')
        
        tic = time.time()
        output['trn'] = func_factor(x=trn.x, y=trn.y, w=trn.w, ids=trn.ids, args=args)
        toc = time.time() - tic
        print(f'Representations [trn] took {toc:0.2f} sec')
        
        tic = time.time()
        output['val'] = func_factor(x=val.x, y=val.y, w=val.w, ids=val.ids, args=args)
        toc = time.time() - tic
        print(f'Representations [val] took {toc:0.2f} sec')
        
        ## Imputate
        if args['imputation_param']['active']:
            
            output['trn']['data'], imputer = impute_datasets(data=output['trn']['data'], features=impute_vars, args=args['imputation_param'], imputer=None)
            output['val']['data'], imputer = impute_datasets(data=output['val']['data'], features=impute_vars, args=args['imputation_param'], imputer=imputer)
            
            fmodel = os.path.join(args["modeldir"], 'imputer.pkl')
            
            print(f'Saving imputer to: {fmodel}', 'green')
            pickle.dump(imputer, open(fmodel, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
    elif runmode == 'eval':
        
        ### Compute reweighting weights (before func_factor because we need all the variables !)
        if args['reweight']:
            
            if args["reweight_file"] is None:
                fmodel = os.path.join(args["datadir"], f'reweighter__{args["__hash_genesis__"]}.pkl') 
            else:    
                fmodel = os.path.join(args["datadir"], args["reweight_file"]) 
            
            if 'load' in args['reweight_mode']:
                print(f'Loading reweighting model from: {fmodel} [runmode = {runmode}]', 'green')
                pdf = pickle.load(open(fmodel, 'rb'))
            else:
                pdf = None # Compute it now
            
            # -----------------------
            # ** Special mode **
            skip_reweights = True if 'skip' in args['reweight_mode'] else False
            # -----------------------
            
            tst.w, pdf = reweight.compute_ND_reweights(pdf=pdf, x=tst.x, y=tst.y, w=tst.w, ids=tst.ids, args=args, skip_reweights=skip_reweights)
            
            if 'write' in args['reweight_mode']:
                print(f'Saving reweighting model to: {fmodel} [runmode = {runmode}]', 'green')
                pickle.dump(pdf, open(fmodel, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
        # Compute different data representations
        print(f'Compute representations [common.func_factor]', 'green')
        
        tic = time.time()
        output['tst'] = func_factor(x=tst.x, y=tst.y, w=tst.w, ids=tst.ids, args=args)
        toc = time.time() - tic
        print(f'Representations [tst] took {toc:0.2f} sec')
        
        ## Imputate
        if args['imputation_param']['active']:
            
            fmodel = os.path.join(args["modeldir"], f'imputer.pkl')
            
            print(f'Loading imputer from: {fmodel}', 'green')
            imputer = pickle.load(open(fmodel, 'rb'))
            output['tst']['data'], _  = impute_datasets(data=output['tst']['data'], features=impute_vars, args=args['imputation_param'], imputer=imputer)
        
    io.showmem()
    
    return output

@iceprint.icelog(LOGGER)
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
        print(f'Imputing data for special values {special_values} in variables {features}', 'yellow')
        
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
        print(f'Imputing data for Inf/Nan in variables {features}', 'yellow')

        for j in dim:        
            mask  = np.logical_not(np.isfinite(data.x[:,j]))
            found = np.sum(mask)
            if found > 0:
                data.x[mask, j] = args['fill_value']    
                print(f'Column {j} Number of {found} ({found/len(data.x):0.3E}) NaN/Inf found in [{data.ids[j]}]', 'red')
    
    return data, imputer

@iceprint.icelog(LOGGER)
def train_models(data_trn, data_val, args=None):
    """
    Train ML/AI models wrapper with pre-processing.
    
    Args:
        Different datatype objects (see the code)
    
    Returns:
        Saves trained models to disk
    """

    print(f'Training models ...', 'yellow')
    print('')
    
    # -----------------------------
    # Prepare output folders

    targetdir = os.path.join(f'{args["plotdir"]}', 'train')

    subdirs = ['']
    for sd in subdirs:
        os.makedirs(os.path.join(targetdir, sd), exist_ok = True)
    # ----------------------------------
    
    # Print training stats
    output_file = os.path.join(args["plotdir"], 'train', 'stats_train_weights.log')
    prints.print_weights(weights=data_trn['data'].w, y=data_trn['data'].y, output_file=output_file)
    
    output_file = os.path.join(args["plotdir"], 'train', 'stats_validate_weights.log')
    prints.print_weights(weights=data_val['data'].w, y=data_val['data'].y, output_file=output_file)
    
    # @@ Tensor normalization @@
    if data_trn['data_tensor'] is not None and (args['varnorm_tensor'] == 'zscore'):
        
        print('')
        print('Z-score normalizing tensor variables ...')
        X_mu_tensor, X_std_tensor = io.calc_zscore_tensor(data_trn['data_tensor'])
        
        data_trn['data_tensor'] = io.apply_zscore_tensor(data_trn['data_tensor'], X_mu_tensor, X_std_tensor)
        data_val['data_tensor'] = io.apply_zscore_tensor(data_val['data_tensor'], X_mu_tensor, X_std_tensor)
        
        # Save it for the evaluation
        pickle.dump({'X_mu_tensor': X_mu_tensor, 'X_std_tensor': X_std_tensor},
                    open(os.path.join(args["modeldir"], 'zscore_tensor.pkl'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    
    # --------------------------------------------------------------------
    
    # @@ Truncate outliers (component by component) from the training set @@
    if args['outlier_param']['algo'] == 'truncate' :
        
        print(f'Truncating outlier variable values with {args["outlier_param"]}')
        
        for j in range(data_trn['data'].x.shape[1]):
            
            # Train sample
            minval = np.percentile(data_trn['data'].x[:,j], args['outlier_param']['qmin'])
            maxval = np.percentile(data_trn['data'].x[:,j], args['outlier_param']['qmax'])

            data_trn['data'].x[data_trn['data'].x[:,j] < minval, j] = minval
            data_trn['data'].x[data_trn['data'].x[:,j] > maxval, j] = maxval

            # -----
            
            # Validation sample
            if args['outlier_param']['process_validate']:
                minval = np.percentile(data_val['data'].x[:,j], args['outlier_param']['qmin'])
                maxval = np.percentile(data_val['data'].x[:,j], args['outlier_param']['qmax'])

                data_val['data'].x[data_val['data'].x[:,j] < minval, j] = minval
                data_val['data'].x[data_val['data'].x[:,j] > maxval, j] = maxval

        # Same for anomalous event weights
        if data_trn['data'].w is not None and args['outlier_param']['truncate_weights']:
            
            print(f'Truncating outlier event weights with {args["outlier_param"]}')

            # Train sample
            minval = np.percentile(data_trn['data'].w, args['outlier_param']['qmin'])
            maxval = np.percentile(data_trn['data'].w, args['outlier_param']['qmax'])

            print(f'Before: min event weight: {minval:0.3E} | max event weight: {maxval:0.3E}')

            # -----
            
            # Validation sample
            data_trn['data'].w[data_trn['data'].w < minval] = minval
            data_trn['data'].w[data_trn['data'].w > maxval] = maxval
            
            minval = np.percentile(data_trn['data'].w, args['outlier_param']['qmin'])
            maxval = np.percentile(data_trn['data'].w, args['outlier_param']['qmax'])

            print(f'After:  min event weight: {minval:0.3E} | max event weight: {maxval:0.3E}')
            
            # Validation sample
            if args['outlier_param']['process_validate']:

                minval = np.percentile(data_val['data'].w, args['outlier_param']['qmin'])
                maxval = np.percentile(data_val['data'].w, args['outlier_param']['qmax'])

                print(f'Before: min event weight: {minval:0.3E} | max event weight: {maxval:0.3E}')

                data_val['data'].w[data_val['data'].w < minval] = minval
                data_val['data'].w[data_val['data'].w > maxval] = maxval

        else:
            print(f'Not truncating outlier event weights {args["outlier_param"]}')
    
    # -------------------------------------------------------------
    # @@ Variable normalization @@
    if   args['varnorm'] == 'zscore' or args['varnorm'] == 'zscore-weighted':
        
        print('')
        print('Z-score normalizing variables ...', 'magenta')
        
        if  args['varnorm'] == 'zscore-weighted':
            print('Using events weights with Z-score ["zscore-weighted"]', 'green')
            X_mu, X_std = io.calc_zscore(X=data_trn['data'].x, weights=data_trn['data'].w)
        else:
            print('Not using event weights with Z-score [to activate use "zscore-weighted"]', 'green')
            X_mu, X_std = io.calc_zscore(X=data_trn['data'].x)

        data_trn['data'].x  = io.apply_zscore(data_trn['data'].x, X_mu, X_std)
        data_val['data'].x  = io.apply_zscore(data_val['data'].x, X_mu, X_std)

        # Save it for the evaluation
        pickle.dump({'X_mu': X_mu, 'X_std': X_std, 'ids': data_trn['data'].ids},
                    open(os.path.join(args["modeldir"], 'zscore.pkl'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
        # Print train
        output_file = os.path.join(f'{args["plotdir"]}', 'train', f'stats_train_{args["varnorm"]}.log')
        prints.print_variables(data_trn['data'].x, data_trn['data'].ids, W=data_trn['data'].w, output_file=output_file)

    elif args['varnorm'] == 'madscore' :
        
        print('')
        print('MAD-score normalizing variables ...', 'magenta')
        X_m, X_mad         = io.calc_madscore(data_trn['data'].x)
        
        data_trn['data'].x = io.apply_zscore(data_trn['data'].x, X_m, X_mad)
        data_val['data'].x = io.apply_zscore(data_val['data'].x, X_m, X_mad)

        # Save it for the evaluation
        pickle.dump({'X_m': X_m, 'X_mad': X_mad, 'ids': data_trn['data'].ids},
                    open(os.path.join(args["modeldir"], 'madscore.pkl'), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        
        # Print train
        output_file = os.path.join(f'{args["plotdir"]}', 'train', f'stats_train_{args["varnorm"]}.log')
        prints.print_variables(data_trn['data'].x, data_trn['data'].ids, W=data_trn['data'].w, output_file=output_file)

    # -------------------------------------------------------------

    def set_distillation_drain(ID, param, inputs, dtype='torch'):
        if 'distillation' in args and args['distillation']['drains'] is not None:
            if ID in args['distillation']['drains']:
                print(f'Creating soft distillation drain for the model <{ID}>', 'yellow')
                
                # By default to torch
                inputs['y_soft'] = torch.tensor(y_soft, dtype=torch.float)
                
                if dtype == 'numpy':
                    inputs['y_soft'] = inputs['y_soft'].detach().cpu().numpy()
    
    # -------------------------------------------------------------

    print(f'Training models:', 'magenta')
    print(args['active_models'], 'green')
    print('')
    
    # Loop over active models
    exceptions = 0
    
    for i in range(len(args['active_models'])):

        # Collect garbage
        gc.collect()
        io.showmem()
        
        ID    = args['active_models'][i]
        param = args['models'][ID]
        print(f'Training <{ID}> | {param} \n')

        try:
            
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
                        'param':       param,
                        'ids':         data_trn['data_deps'].ids}
                
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
                        'args':        args,
                        'param':       param,
                        'ids':         data_trn['data'].ids}

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
            if 'distillation' in args and ID == args['distillation']['source']:
                
                if len(args['primary_classes']) != 2:
                    raise Exception(__name__ + f'.train_models: Distillation supported now only for 2-class classification')
                
                print(f'Computing distillation soft targets from the source <{ID}> ', 'yellow')

                if   param['train'] == 'xgb':
                    XX, XX_ids = aux.red(data_trn['data'].x, data_trn['data'].ids, param)
                    y_soft = model.predict(xgboost.DMatrix(data=XX, feature_names=XX_ids))
                    if len(y_soft.shape) > 1:
                        y_soft = y_soft[:, args['signalclass']]
                
                elif param['train'] == 'torch_graph':
                    y_soft = model.softpredict(data_trn['data_graph'])[:, args['signalclass']]
                else:
                    raise Exception(__name__ + f".train_models: Unsupported distillation source <{param['train']}>")
            # --------------------------------------------------------
        
        except KeyboardInterrupt:
            print(f'CTRL+C catched -- continue with the next model', 'red')
        
        except Exception as e:
            print(e)
            prints.printbar('*')
            print(f'Exception occured, check the model definition! -- continue', 'red')
            prints.printbar('*')
            exceptions += 1
    
    print(f'[done]', 'yellow')
    
    if exceptions > 0:
        raise Exception(__name__ + f'.train_models: Number of fatal exceptions = {exceptions} [check your data / model definitions -- some model did not train]')

    return True

@iceprint.icelog(LOGGER)
def evaluate_models(data=None, info=None, args=None):
    """
    Evaluate ML/AI models.

    Args:
        Different datatype objects (see the code)

    Returns:
        Saves evaluation plots to the disk
    """

    print(f'Evaluating models ...', 'yellow')
    print('')
    
    # -----------------------------
    # ** GLOBALS **

    global roc_mstats
    global roc_labels
    global roc_paths
    global roc_filters
    global y_preds
    global corr_mstats
    
    roc_mstats  = {}
    roc_labels  = {}
    roc_paths   = {}
    y_preds     = []
    roc_filters = {}
    corr_mstats = {}

    global ROC_binned_mstats
    global ROC_binned_mlabel
    
    ROC_binned_mstats = {}
    ROC_binned_mlabel = {}

    # -----------------------------
    # Prepare output folders

    targetdir = os.path.join(f'{args["plotdir"]}', 'eval')

    subdirs = ['', 'ROC', 'MVA', 'COR']
    for sd in subdirs:
        os.makedirs(os.path.join(targetdir, sd), exist_ok = True)

    # --------------------------------------------------------------------
    
    # Print evaluation stats
    output_file = os.path.join(args["plotdir"], 'eval', 'stats_eval_weights.log')
    prints.print_weights(weights=data['data'].w, y=data['data'].y, output_file=output_file)
    
    
    # --------------------------------------------------------------------
    # Collect data
    
    X       = None
    X_RAW   = None
    ids_RAW = None
    
    y       = None
    weights = None
    
    if data['data'] is not None:

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
    
    X_kin  = None
    ids_kin = None
    if data['data_kin'] is not None:
        X_kin   = data['data_kin'].x
        ids_kin = data['data_kin'].ids
    # --------------------------------------------------------------------
    
    try:
        ### Tensor variable normalization
        if data['data_tensor'] is not None and (args['varnorm_tensor'] == 'zscore'):
            
            print('\nZ-score normalizing tensor variables ...', 'magenta')
            Z_data = pickle.load(open(os.path.join(args["modeldir"], 'zscore_tensor.pkl'), 'rb'))
            X_mu_tensor  = Z_data['X_mu_tensor']
            X_std_tensor = Z_data['X_std_tensor']
            
            X_2D = io.apply_zscore_tensor(X_2D, X_mu_tensor, X_std_tensor)

        ### Variable normalization
        if   args['varnorm'] == 'zscore' or args['varnorm'] == 'zscore-weighted':
            
            print('\nZ-score normalizing variables ...', 'magenta')
            Z_data = pickle.load(open(os.path.join(args["modeldir"], 'zscore.pkl'), 'rb'))
            X_mu   = Z_data['X_mu']
            X_std  = Z_data['X_std']
            
            X = io.apply_zscore(X, X_mu, X_std)
            
            output_file = os.path.join(args["plotdir"], 'eval', f'stats_variables_{args["varnorm"]}.log')
            prints.print_variables(X, ids, weights, output_file=output_file)

        elif args['varnorm'] == 'madscore':
            
            print('\nMAD-score normalizing variables ...', 'magenta')
            Z_data = pickle.load(open(os.path.join(args["modeldir"], 'madscore.pkl'), 'rb'))
            X_m    = Z_data['X_m']
            X_mad  = Z_data['X_mad']
            
            X = io.apply_madscore(X, X_m, X_mad)
            
            output_file = os.path.join(args["plotdir"], 'eval', f'stats_variables_{args["varnorm"]}.log')
            prints.print_variables(X, ids, weights, output_file=output_file)
        
    except Exception as e:
        print(e)
        print(f'Exception occured in variable normalization. Continue without!', 'red')
    
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
    
    print(f'Evaluating models:', 'magenta')
    print(args['active_models'], 'green')
    print('')
    
    exceptions = 0
    
    try:
        
        for i in range(len(args['active_models'])):
            
            ID    = args['active_models'][i]
            param = args['models'][ID]
            print(f'Evaluating <{ID}> | {param} \n')
            
            inputs = {'weights': weights, 'label': param['label'],
                    'targetdir': targetdir, 'args':args, 'X_kin': X_kin, 'ids_kin': ids_kin, 'X_RAW': X_RAW, 'ids_RAW': ids_RAW}
            
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
            
            elif param['predict'] == 'xgb_scalar':
                func_predict = predict.pred_xgb_scalar(args=args, param=param, feature_names=aux.red(X,ids,param,'ids'))
                
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
            
            elif param['predict'] == 'cut':
                func_predict = predict.pred_cut(ids=ids_RAW, param=param)
                plot_XYZ_wrap(func_predict = func_predict, x_input = X_RAW, y = y, **inputs)
            
            elif param['predict'] == 'cutset':
                func_predict = predict.pred_cutset(ids=ids_RAW, param=param)
                plot_XYZ_wrap(func_predict = func_predict, x_input = X_RAW, y = y, **inputs)

                if args['plot_param']['contours']['active']:
                    plots.plot_contour_grid(pred_func=func_predict, X=X_RAW, y=y, ids=ids_RAW, transform='numpy', 
                        targetdir=aux.makedir(os.path.join(f'{args["plotdir"]}', 'eval/2D-contours', f'{param["label"]}')))
            else:
                raise Exception(__name__ + f'.Unknown param["predict"] = {param["predict"]} for ID = {ID}')
    
    except Exception as e:
        print(e)
        prints.printbar('*')
        print(f'Exception occured, check your steering cards! -- continue', 'red')
        prints.printbar('*')
        exceptions += 1
    
    ## Multiple model comparisons
    plot_XYZ_multiple_models(targetdir=targetdir, args=args)
    
    ## Pickle results to output
    resdict = {'roc_mstats':        roc_mstats,
               'roc_labels':        roc_labels,
               'roc_paths':         roc_paths,
               'roc_filters':       roc_filters,
               'y_preds':           y_preds,
               'corr_mstats':       corr_mstats,
               'data':              io.IceXYW(x=X_RAW, y=y, w=weights, ids=ids_RAW),
               'ROC_binned_mstats': ROC_binned_mstats,
               'ROC_binned_mlabel': ROC_binned_mlabel,
               'info':              info}
    
    targetfile = os.path.join(targetdir, 'eval_results.pkl')
    print(f'Saving pickle output to:')
    print(f'{targetfile}')
    
    with open(targetfile, 'wb') as file:
        pickle.dump(resdict, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f'[done]', 'yellow')
    
    if exceptions > 0:
        raise Exception(__name__ + f'.evaluate_models: Number of fatal exceptions = {exceptions} [check your input -- results or labelling may be corrupted now]')
    
    return True

@iceprint.icelog(LOGGER)
def make_plots(data, args, runmode):
    """
    Basic Q/A-plots
    """
    
    ### Plot variables
    if args['plot_param']['basic']['active']:
        
        param = copy.deepcopy(args['plot_param']['basic'])
        param.pop('active')
        
        ### Specific variables
        if data['data_kin'] is not None:
            targetdir = aux.makedir(f'{args["plotdir"]}/{runmode}/distributions/kinematic/')
            plots.plotvars(X = data['data_kin'].x, y = data['data_kin'].y, weights = data['data_kin'].w, ids = data['data_kin'].ids,           
                targetdir=targetdir, title=f"training re-weight reference class: {args['reweight_param']['reference_class']}",
                num_cpus=args['num_cpus'], **param)
        
        ### Plot MVA input variable plots
        targetdir = aux.makedir(f'{args["plotdir"]}/{runmode}/distributions/MVA-input/')
        plots.plotvars(X = data['data'].x, y = data['data'].y, weights = data['data'].w,  ids = data['data'].ids,
            targetdir=targetdir, title=f"training re-weight reference class: {args['reweight_param']['reference_class']}",
            num_cpus=args['num_cpus'], **param)

    ### Correlations
    if args['plot_param']['corrmat']['active']:
        
        targetdir = aux.makedir(f'{args["plotdir"]}/{runmode}/distributions/')
        fig,ax    = plots.plot_correlations(X=data['data'].x, weights=data['data'].w, ids=data['data'].ids, y=data['data'].y, targetdir=targetdir)

    return True

@iceprint.icelog(LOGGER)
def plot_XYZ_wrap(func_predict, x_input, y, weights, label, targetdir, args,
    X_kin, ids_kin, X_RAW, ids_RAW):
    """ 
    Arbitrary plot steering function.
    Add new plot types here, steered from plots.yml
    """

    global roc_mstats
    global roc_labels
    global roc_paths
    global roc_filters
    global y_preds
    global corr_mstats

    global ROC_binned_mstats
    global ROC_binned_mlabel
    
    # ** Compute predictions once and for all here **
    y_pred = func_predict(x_input)
    y_preds.append(copy.deepcopy(y_pred))
    
    # --------------------------------------
    ### Output score re-weighted observables
    
    if 'OBS_reweight' in args['plot_param'] and args['plot_param']['OBS_reweight']['active']:
        
        print('OBS_reweight')
        
        pick_ind, var_names = aux.pick_index(all_ids=ids_RAW, vars=args['plot_param']['OBS_reweight']['var'])
        
        # -----------------------------------------------
        
        # ** All inclusive **
        
        ## Plot over different temperature values
        sublabel = 'inclusive'
        dir      = aux.makedir(f'{targetdir}/OBS_reweight/{label}/{sublabel}')
        filename = dir + "/stats_chi2_summary.log"
        open(filename, 'w').close() # Clear content
        
        for tau in args['plot_param']['OBS_reweight']['tau_values']:
            
            chi2_table = plots.plot_AIRW(X=X_RAW, y=y, ids=ids_RAW, weights=weights, y_pred=y_pred,
                                         pick_ind=pick_ind, label=label, sublabel=sublabel,
                                         param=args['plot_param']['OBS_reweight'], tau=tau,
                                         targetdir=targetdir + '/OBS_reweight', num_cpus=args['num_cpus'])
            
            plots.table_writer(filename=filename, label=label, sublabel=sublabel, tau=tau, chi2_table=chi2_table)

            gc.collect() #!
        
        # ** Set filtered **
        if 'set_filter' in args['plot_param']['OBS_reweight']:
            
            filters = args['plot_param']['OBS_reweight']['set_filter']
            mask_filterset, text_filterset, path_filterset = stx.filter_constructor(filters=filters, X=X_RAW, y=y, ids=ids_RAW)

            for m in range(mask_filterset.shape[0]):
                
                if np.sum(mask_filterset[m,:]) == 0:
                    print(f'OBS_reweight: mask[{m}] has no events passing -- skip [{text_filterset[m]}]', 'red')
                    continue
                
                ## Plot over different temperature values
                sublabel = text_filterset[m]
                dir      = aux.makedir(f'{targetdir}/OBS_reweight/{label}/{sublabel}')
                filename = dir + "/chi2_summary.log"
                open(filename, 'w').close() # Clear content
                
                for tau in args['plot_param']['OBS_reweight']['tau_values']:
                    
                    mask = mask_filterset[m,:]
                    
                    chi2_table = plots.plot_AIRW(X=X_RAW[mask,:], y=y[mask], ids=ids_RAW, weights=weights[mask], y_pred=y_pred[mask],
                                         pick_ind=pick_ind, label=label, sublabel=sublabel,
                                         param=args['plot_param']['OBS_reweight'], tau=tau,
                                         targetdir=targetdir + '/OBS_reweight')
                    
                    plots.table_writer(filename=filename, label=label, sublabel=sublabel, tau=tau, chi2_table=chi2_table)
    
    
    # --------------------------------------
    ### ROC plots
    
    if 'ROC' in args['plot_param'] and args['plot_param']['ROC']['active']:

        print('ROC')
        
        def plot_helper(mask, sublabel, pathlabel):
            
            print(f'Computing aux.Metric for {sublabel}')
            
            metric = aux.Metric(y_true=y[mask], y_pred=y_pred[mask],
                weights=weights[mask] if weights is not None else None,
                num_bootstrap=args['plot_param']['ROC']['num_bootstrap'])

            if sublabel not in roc_mstats:
                roc_mstats[sublabel]  = []
                roc_labels[sublabel]  = []
                roc_paths[sublabel]   = []
                roc_filters[sublabel] = []
            
            roc_mstats[sublabel].append(metric)
            roc_labels[sublabel].append(label)
            roc_paths[sublabel].append(pathlabel)
            roc_filters[sublabel].append(mask)
        
        # ** All inclusive **
        mask = np.ones(len(y_pred), dtype=bool)
        plot_helper(mask=mask, sublabel='inclusive', pathlabel='inclusive')

        # ** Set filtered **
        if 'set_filter' in args['plot_param']['ROC']:

            filters = args['plot_param']['ROC']['set_filter']
            mask_filterset, text_filterset, path_filterset = stx.filter_constructor(filters=filters, X=X_RAW, y=y, ids=ids_RAW)

            for m in range(mask_filterset.shape[0]):
                
                if np.sum(mask_filterset[m,:]) == 0:
                    print(f'ROC: mask[{m}] has no events passing -- skip [{text_filterset[m]}]', 'red')
                    continue
                
                plot_helper(mask=mask_filterset[m,:], sublabel=text_filterset[m], pathlabel=path_filterset[m])

    # --------------------------------------
    ### ROC binned plots (no filterset selection supported here)
    if 'ROC_binned' in args['plot_param'] and args['plot_param']['ROC_binned']['active']:
        
        print('ROC_binned')
        
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
                    ids_kin=ids_kin, X=X_RAW, ids=ids_RAW, edges=edges, VAR=var[0], \
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
                    ids_kin=ids_kin, X=X_RAW, ids=ids_RAW, edges=edges, label=label, VAR=var)
                plt.savefig(aux.makedir(f'{targetdir}/ROC/{label}') + f'/ROC-binned[{i}].pdf', bbox_inches='tight')

            else:
                print(var)
                raise Exception(__name__ + f'.plot_AUC_wrap: Unknown dimensionality {len(var)}')

    # ----------------------------------------------------------------
    ### MVA-output 1D-plot
    if 'MVA_output' in args['plot_param'] and args['plot_param']['MVA_output']['active']:

        print('MVA_output')
        
        def plot_helper(mask, sublabel, pathlabel):
            
            inputs = {'y_pred':     y_pred[mask],
                      'y':          y[mask],
                      'weights':    weights[mask] if weights is not None else None,
                      'class_ids':  None,
                      'label':      f'{label}/{sublabel}',
                      'path':       f'{targetdir}/MVA/{label}/{pathlabel}'}

            plots.density_MVA_wclass(**inputs, **args['plot_param'][f'MVA_output'])
        
        # ** All inclusive **
        mask  = np.ones(len(y_pred), dtype=bool)
        plot_helper(mask=mask, sublabel='inclusive', pathlabel='inclusive')

        # ** Set filtered **
        if 'set_filter' in args['plot_param']['MVA_output']:

            filters = args['plot_param']['MVA_output']['set_filter']
            mask_filterset, text_filterset, path_filterset = stx.filter_constructor(filters=filters, X=X_RAW, y=y, ids=ids_RAW)

            for m in range(mask_filterset.shape[0]):
                
                if np.sum(mask_filterset[m,:]) == 0:
                    print(f'MVA_output: mask[{m}] has no events passing -- skip [{text_filterset[m]}]', 'red')
                    continue
                
                plot_helper(mask=mask_filterset[m,:], sublabel=text_filterset[m], pathlabel=path_filterset[m])

    # ----------------------------------------------------------------
    ### MVA-output 2D correlation plots
    if 'MVA_2D' in args['plot_param'] and args['plot_param']['MVA_2D']['active']:

        print('MVA_2D')
        
        for i in range(100): # Loop over plot types
            
            pid = f'plot[{i}]'
            if pid in args['plot_param']['MVA_2D']:        
                var     = args['plot_param']['MVA_2D'][pid]['var']
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
                    'label':       f'{label}/{sublabel}',
                    'path':        f'{targetdir}/COR/{label}/{pathlabel}/plot-{i}'}
                
                output = plots.density_COR_wclass(y=y[mask], **inputs,
                                                  **args['plot_param']['MVA_2D'][pid])
                #plots.density_COR(**inputs, **args['plot_param']['MVA_2D'][pid])

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
                mask_filterset, text_filterset, path_filterset = stx.filter_constructor(filters=filters, X=X_RAW, y=y, ids=ids_RAW)
                
                for m in range(mask_filterset.shape[0]):
                    
                    if np.sum(mask_filterset[m,:]) == 0:
                        print(f'MVA_2D: mask[{m}] has no events passing -- skip [{text_filterset[m]}]', 'red')
                        continue
                    
                    plot_helper(mask=mask_filterset[m,:], pick_ind=pick_ind, sublabel=text_filterset[m], pathlabel=path_filterset[m])
    
    return True

@iceprint.icelog(LOGGER)
def plot_XYZ_multiple_models(targetdir, args):

    global roc_mstats
    global roc_labels
    global roc_paths
    global ROC_binned_mstats

    # ===================================================================
    # Plot correlation coefficient comparisons

    from pprint import pprint

    ### MVA-output 2D correlation plots
    if 'MVA_2D' in args['plot_param'] and args['plot_param']['MVA_2D']['active']:
        
        print('MVA_2D:')
        pprint(corr_mstats)
        
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
    if 'ROC' in args['plot_param'] and args['plot_param']['ROC']['active']:
        
        print('ROC:')
        pprint(roc_mstats)
        
        # We have the same number of filterset (category) entries for each model, pick the first
        dummy = 0

        # Direct collect:  Plot all models per filter category
        for filterset_key in roc_mstats.keys():

            print(f'Plot ROC curve [{filterset_key}]', 'green')

            path_label = roc_paths[filterset_key][dummy]
            plots.ROC_plot(roc_mstats[filterset_key], roc_labels[filterset_key],
                xmin=args['plot_param']['ROC']['xmin'],
                title=f'category: {filterset_key}', filename=aux.makedir(targetdir + f'/ROC/--ALL--/{path_label}') + '/ROC-all-models')

        # Inverse collect: Plot all filter categories ROCs per model
        for model_index in range(len(roc_mstats[list(roc_mstats)[dummy]])):
            
            print(f'Plot ROC curve for the model [{model_index}]', 'green')

            rocs_       = [roc_mstats[filterset_key][model_index] for filterset_key in roc_mstats.keys()]
            labels_     = list(roc_mstats.keys())
            model_label = roc_labels[list(roc_labels)[dummy]][model_index]
            
            plots.ROC_plot(rocs_, labels_,
                xmin=args['plot_param']['ROC']['xmin'],
                title=f'model: {model_label}', filename=aux.makedir(targetdir + f'/ROC/{model_label}') + '/ROC-all-categories')

    ### Plot all MVA outputs (not implemented)
    # plots.MVA_plot(mva_mstats, mva_labels, title = '', filename=aux.makedir(targetdir + '/MVA/--ALL--') + '/MVA')

    ### Plot all 1D-binned ROC curves
    if 'ROC_binned' in args['plot_param'] and args['plot_param']['ROC_binned']['active']:
        
        print('ROC_binned:')
        
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
                    # title = f'BINNED MVA: {var[0]}$ \\in [{edges[b]:0.1f}, {edges[b+1]:0.1f})$'
                    # plots.MVA_plot(xy, legs, title=title, filename=targetdir + f'/MVA/--ALL--/MVA-binned[{i}]-bin[{b}]')

    print(f'[done]')

    return True
