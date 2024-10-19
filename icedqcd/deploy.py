# Deploy MVA-model to data (or MC) files (work in progress, not all models supported ...)
#
# This code is compatible with grid (SGE, HTCondor) submission
#
# m.mieskolainen@imperial.ac.uk, 2024

import os
import numpy as np
import awkward as ak
import pickle
import uproot
import logging
import copy
import socket
import gc

from importlib import import_module
from datetime import datetime

from icedqcd import common
from icenet.tools import iceroot, io, aux, process
from icenet.deep import predict
from tqdm import tqdm

# ------------------------------------------
#from icenet import print # (reservation)
# ------------------------------------------

def f2s(value, decimals=2):
    """
    Convert e.g. floating point "1.5" to "1p5"
    """
    return str(np.round(value, decimals)).replace('.', 'p')

def generate_cartesian_param(ids, mc_param, decimals=2):
    """
    Generate cartesian N-dim array for the theory model parameter conditional sampling
    """
    arr = []
    for key in mc_param.keys():
        arr.append(np.round(np.array(mc_param[key]), decimals))

    CAX    = aux.cartesian_product(*arr)
    pindex = np.zeros(len(mc_param.keys()), dtype=int)
    
    for i, key in enumerate(mc_param.keys()):
        pindex[i] = ids.index(f'MODEL_{key}')
    
    return CAX, pindex

def create_ID_label(ID, mc_param, lattice_values, decimals=2):
    """
    Create parameter label for the conditional model parameters
    """
    ID_label = f'{ID}_'
    
    for i, key in enumerate(mc_param.keys()):
        ID_label += f'_{key}_{f2s(lattice_values[i], decimals=decimals)}'
    
    return ID_label

def zscore_normalization(X, args):
    """
    Z-score normalization
    """
    if   args['varnorm'] == 'zscore':                            
        print(f'Z-score normalizing variables ...')
        X_mu, X_std = pickle.load(open(args["modeldir"] + '/zscore.pkl', 'rb'))

        print(f'X.shape = {X.shape} | X_mu.shape = {X_mu.shape} | X_std.shape = {X_std.shape}')
        X = io.apply_zscore(X, X_mu, X_std)

    return X

def process_data(args):
    """
    Main processing loop
    """
    rootname  = args["rootname"]
    inputvars = import_module("configs." + rootname + "." + args["inputvars"])
    
    print(f'Processing data', 'green')
    print(args)
    
    ## ** Special YAML loader here **
    from libs.ccorp.ruamel.yaml.include import YAML
    yaml = YAML()
    yaml.allow_duplicate_keys = True
    cwd = os.getcwd()

    with open(f'{cwd}/configs/{rootname}/{args["inputmap"]}', 'r') as f:
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
    # Phase 0: Create output path
    
    path_str = f"{cwd}/output/{rootname}/deploy/modeltag__{args['modeltag']}"
    
    # Add conditional tag
    conditional_tag = f'--use_conditional__{args["use_conditional"]}' if args["use_conditional"] else ""
    path_str += conditional_tag
    
    basepath = aux.makedir(path_str)
    
    # ------------------
    # Phase 1: Read ROOT-file to awkward-format
    
    nodestr  = (f"inputmap__{io.safetxt(args['inputmap'])}--grid_id__{args['grid_id']}--hostname__{socket.gethostname()}--time__{datetime.now()}").replace(' ', '')
    logging.basicConfig(filename=f'{basepath}/deploy--{nodestr}.log', encoding='utf-8',
        level=logging.DEBUG, format='%(asctime)s | %(message)s', datefmt='%d/%m/%Y %H:%M:%S')

    # Save to log-file
    logging.debug(__name__ + f'.process_data: {nodestr}')
    logging.debug('')
    logging.debug(f'{inputmap}')
    
    # Pick the model lattice definition
    if args['use_conditional']:    
        mc_param = args['mc_param']
    
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
        
        # ----------------------------------
        # GRID (batch) processing
        all_file_id = aux.split_start_end(range(len(rootfiles)), args['grid_nodes'])

        try:
            file_id = all_file_id[args['grid_id']]
            print(f"file_id = {file_id} (grid_id = {args['grid_id']})")
            
        except Exception as e:
            logging.debug('Subfolder already saturated -- no processing under this grid PC node -- continue')
            continue # This subfolder is already saturated on the grid processing, i.e. grid_id > len(all_file_id)
        # ----------------------------------
        
        for k in range(file_id[0], file_id[-1]):

            filename = rootfiles[k]
            param = {
                'rootfile'    : filename,
                'tree'        : 'Events',
                'entry_start' : None,
                'entry_stop'  : None,
                'maxevents'   : None,
                'ids'         : inputvars.LOAD_VARS,
                'library'     : 'ak'
            }
            
            try:
                X_nocut, ids_nocut = iceroot.load_tree(**param)

                # Write to log-file
                logging.debug(f'{filename} | Number of events: {len(X_nocut)}')
                total_num_events += len(X_nocut)

            except Exception as e:
                print(f'A fatal error in iceroot.load_tree with a file "{filename}": ' + str(e), 'red')
                
                # Write to log-file
                logging.debug(f'{filename} | A fatal error in iceroot.load_tree: ' + str(e))
                continue
            
            # -------------------------------------------------
            # Phase 1.5: Add conditional (theory MC param) variables

            if args['use_conditional']:

                print(f'Initializing conditional theory (model) parameters')
                for var in mc_param.keys():
                    # Create new 'record' (column) to ak-array and init to zero
                    col_name          = f'MODEL_{var}'
                    X_nocut[col_name] = 0.0

                ids_nocut = ak.fields(X_nocut)

            # ------------------
            # Phase 2: Apply pre-selections to get an event mask

            mask = common.process_root(X=X_nocut, args=args, return_mask=True)
            
            if np.sum(mask) == 0:
                X        = X_nocut
                logging.debug("No events left after pre-cuts -- scores will be -1 for all events")
                OK_event = False
            else:
                X        = X_nocut[mask]
                OK_event = True
            
            # ------------------
            # Phase 3: Convert to icenet dataformat

            try:
                # Dequantization always off in deployment
                data = common.splitfactor(x=X, y=None, w=None, ids=ids_nocut, args=args,
                                          skip_graph=True, use_dequantize=False)
            except Exception as e:
                
                # Something went wrong at OS level (e.g. memory), write to log-file and exit with error
                txt = f'{filename} | A fatal error in common.splitfactor -- os._exit(os.EX_OSERR): ' + str(e)
                logging.debug(txt)
                print(txt)
                os._exit(os.EX_OSERR)
            
            # ------------------
            # ** Free memory **
            del X
            del X_nocut
            gc.collect()
            # ------------------
            
            # ------------------
            # Phase 4: Apply MVA-models
            
            ALL_scores = {}

            for i in range(len(args['active_models'])):
                
                gc.collect()
            
                ID    = args['active_models'][i]
                param = args['models'][ID]
                
                # xgb / iceboost
                if param['predict'] in ['xgb', 'xgb_logistic']:
                    
                    print(f'Evaluating MVA-model "{ID}" \n')

                    ## 1. Impute data
                    if args['imputation_param']['active']:
                        
                        fmodel  = os.path.join(args["modeldir"], 'imputer.pkl')
                        imputer = pickle.load(open(fmodel, 'rb'))
                        data['data'], _  = process.impute_datasets(data=data['data'], features=None, args=args['imputation_param'], imputer=imputer)
                    
                    ## 2. Apply the input variable set reductor
                    X,ids = aux.red(X=data['data'].x, ids=data['data'].ids, param=param)

                    ## 3. Get the MVA-model
                    func_predict, model = get_predictor(args=args, param=param, feature_names=ids)

                    ## 4. ** Conditional model **
                    if args['use_conditional']:
                        
                        ## Get conditional parameters
                        CAX, pindex = generate_cartesian_param(ids=ids, mc_param=mc_param)

                        ## Run the MVA-model on all the theory model points in CAX array
                        for z in tqdm(range(len(CAX))):

                            gc.collect()

                            # Set the new conditional model parameters to X
                            lattice_values = CAX[z,:]
                            ID_label       = create_ID_label(ID=ID, mc_param=mc_param, lattice_values=lattice_values)

                            if OK_event:
                                XX            = copy.deepcopy(X)
                                XX[:, pindex] = lattice_values  # Set new values

                                # Variable normalization
                                XX = zscore_normalization(X=XX, args=args)
                                
                                # Predict
                                pred = func_predict(XX)
                                pred = aux.unmask(x=pred, mask=mask, default_value=-1)
                            else:
                                pred = (-1) * np.ones(len(mask)) # all -1

                            # Save
                            ALL_scores[io.rootsafe(ID_label)] = pred

                    # 4. ** Unconditional model **
                    else:

                        if OK_event:

                            # Variable normalization
                            XX = copy.deepcopy(X)
                            XX = zscore_normalization(X=XX, args=args)

                            # Predict
                            pred = func_predict(XX)
                            pred = aux.unmask(x=pred, mask=mask, default_value=-1)
                                               
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

            aux.makedir(basepath + '/' + filename.rsplit('/', 1)[0]) # Create dir
            outputfile = basepath + '/' + filename.replace('.root', '-icenet.root')
            
            print(f'Saving root output to "{outputfile}"')

            with uproot.recreate(outputfile, compression=uproot.ZLIB(4)) as rfile:
                rfile[f"Events"] = ALL_scores
            
            gc.collect() #! Garbage collection
        
        # Write to log-file
        logging.debug(f'Total number of events: {total_num_events}')


def get_predictor(args, param, feature_names=None):    
    """
    Load model predictor function handle
    """
    model = None

    if   param['predict'] == 'xgb':
        func_predict, model = predict.pred_xgb(args=args, param=param, feature_names=feature_names, return_model=True)

    elif param['predict'] == 'xgb_logistic':
        func_predict, model = predict.pred_xgb_logistic(args=args, param=param, feature_names=feature_names, return_model=True)

    #elif param['predict'] == 'torch_vector':
    #    func_predict = predict.pred_torch_generic(args=args, param=param)

    #elif param['predict'] == 'torch_graph':
    #    func_predict = predict.pred_torch_graph(args=args, param=param)
    
    else:
        raise Exception(__name__ + f'.get_predictor: Unknown param["predict"] = {param["predict"]}')

    return func_predict, model
