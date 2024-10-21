# Generic model evaluation wrapper functions
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import torch_geometric

import pickle
import copy

from tqdm import tqdm

# xgboost
import xgboost

# icenet
from icenet.tools import stx, aux, aux_torch

from icenet.algo  import flr
from icenet.deep  import optimize
from icenet.deep  import dbnf

# ------------------------------------------
from icenet import print
# ------------------------------------------


def pred_cut(ids, param):

    print(f'Evaluate [{param["label"]}] cut model ...')

    # Get feature name variables
    index = ids.index(param['variable'])
    
    def func_predict(x):
        # Identity
        if param['transform'] is None:             
            return param['sign'] * x[...,index]
        # Transform via sigmoid from R -> [0,1]
        elif param['transform'] == 'sigmoid':
            return aux.sigmoid(param['sign'] * x[...,index])
        # Numpy function, e.g. np.abs, np.tanh ...
        elif 'np.' in param['transform']:
            cmd = "param['sign'] * " + param['transform'] + f"(x[...,index])"
            return eval(cmd)
        else:
            raise Exception(__name__ + '.pred_cuts: Unknown transform chosen (check your syntax)')
    
    return func_predict

def pred_cutset(ids, param):

    print(f'Evaluate [{param["label"]}] fixed cutset model ...')
    cutstring = param['cutstring']
    print(f'cutstring: "{cutstring}"')

    # Expand model parameters into memory
    for p in param['model_param'].keys():
        exec(f"{p} = param['model_param']['{p}']")
    
    # Expand cut string
    variable = param['variable'] # this is needed in memory!
    expr = eval(f'f"{cutstring}"')
    
    print(f'cutstring expanded: {expr}')
    
    def func_predict(x):
        y = stx.eval_boolean_syntax(expr=expr, X=x, ids=ids)
        return y.astype(float)
    
    return func_predict

def pred_graph_xgb(args, param):
    
    print(f'Evaluate [{param["label"]}] model ...')
    
    device = param['deploy_device'] if 'deploy_device' in param else param['device']
    
    # torch
    graph_model = aux_torch.load_torch_checkpoint(path=f"{args['modeldir']}/{param['graph']['label']}", \
        label=param['graph']['label'], epoch=param['graph']['readmode'])
    
    graph_model, device = optimize.model_to_cuda(graph_model, device_type=device)
    
    graph_model.eval() # Turn on eval!
    
    # xgboost
    filename, N_trees = aux.create_model_filename_xgb(path=f"{args['modeldir']}/{param['xgb']['label']}", label=param['xgb']['label'], \
                            epoch=param['xgb']['readmode'], filetype='.pkl')
    
    with open(filename, 'rb') as file:
        xgb_model = pickle.load(file)['model']
    
    def func_predict(x):

        if isinstance(x, list):
            x_in = x
        else:
            x_in = [x]
        
        # Geometric type -> need to use batch loader
        loader  = torch_geometric.loader.DataLoader(x_in, batch_size=len(x), shuffle=False)
        for batch in loader:
            conv_x = graph_model.forward(batch.to(device), conv_only=True).detach().cpu().numpy()

        # Concatenate convolution features and global features for the BDT
        N     = conv_x.shape[0]
        dim1  = conv_x.shape[1]
        dim2  = len(x_in[0].u)
        
        x_tot = np.zeros((N, dim1+dim2))
        for i in range(N):                 # Over all events
            x_tot[i,0:dim1] = conv_x[i,:]  # Convoluted features
            x_tot[i,dim1:]  = x_in[i].u    # Global features

        pred = xgb_model.predict(xgboost.DMatrix(data = x_tot), iteration_range=(0,N_trees))
        if len(pred.shape) > 1: pred = pred[:, args['signal_class']]
        return pred
    
    return func_predict

def pred_torch_graph(args, param, batch_size=5000, return_model=False):
    
    print(f'Evaluate [{param["label"]}] model ...')
    
    model = aux_torch.load_torch_checkpoint(path=f"{args['modeldir']}/{param['label']}",
                label=param['label'], epoch=param['readmode'])
    
    device = param['deploy_device'] if 'deploy_device' in param else param['device']
    model, device = optimize.model_to_cuda(model, device_type=device)
    
    model.eval() # ! Turn on eval mode!
    
    def func_predict(x):

        if isinstance(x, list):
            x_in = x
        else:
            x_in = [x]
        
        # Geometric type -> need to use batch loader
        loader = torch_geometric.loader.DataLoader(x_in, batch_size=batch_size, shuffle=False)

        # Predict in smaller batches not to overflow GPU memory
        for i, batch in tqdm(enumerate(loader)):
            
            if 'raw_logit' in param:
                y = model.forward(batch.to(device))[:, args['signal_class']].detach().cpu().numpy()
            else:
                y = model.softpredict(batch.to(device))[:, args['signal_class']].detach().cpu().numpy()
            
            y_tot = copy.deepcopy(y) if (i == 0) else np.concatenate((y_tot, y), axis=0)
        
        return y_tot

    if return_model == False:
        return func_predict
    else:
        return func_predict, model


def pred_torch_generic(args, param, return_model=False):
    
    print(f'Evaluate [{param["label"]}] model ...')
    
    model = aux_torch.load_torch_checkpoint(path=f"{args['modeldir']}/{param['label']}",
                label=param['label'], epoch=param['readmode'])
    
    device = param['deploy_device'] if 'deploy_device' in param else param['device']
    model, device = optimize.model_to_cuda(model, device_type=device)
    
    model.eval() # ! Turn on eval mode!
    
    def func_predict(x):

        if not isinstance(x, dict):
            x_in = x.to(device)
        else:
            x_in = copy.deepcopy(x)
            for key in x_in.keys():
                x_in[key] = x_in[key].to(device)
        
        if 'raw_logit' in param:
            return model.forward(x_in)[:, args['signal_class']].detach().cpu().numpy()
        else:
            return model.softpredict(x_in)[:, args['signal_class']].detach().cpu().numpy()
    
    if return_model == False:
        return func_predict
    else:
        return func_predict, model


def pred_torch_scalar(args, param, return_model=False):
    
    print(f'Evaluate [{param["label"]}] model ...')
    
    model = aux_torch.load_torch_checkpoint(path=f"{args['modeldir']}/{param['label']}",
                label=param['label'], epoch=param['readmode'])
    
    device = param['deploy_device'] if 'deploy_device' in param else param['device']
    model, device = optimize.model_to_cuda(model, device_type=device)
    
    model.eval() # ! Turn on eval mode!
    
    def func_predict(x):
        
        if not isinstance(x, dict):
            x_in = x.to(device)
        else:
            x_in = copy.deepcopy(x)
            for key in x_in.keys():
                x_in[key] = x_in[key].to(device)
        
        if 'raw_logit' in param:
            return model.forward(x_in).squeeze().detach().cpu().numpy()
        else:
            return model.softpredict(x_in).squeeze().detach().cpu().numpy()
    
    if return_model == False:
        return func_predict
    else:
        return func_predict, model


def pred_flow(args, param, n_dims, return_model=False):

    print(f'Evaluate [{param["label"]}] model ...')
    
    # Load models
    param['model_param']['n_dims'] = n_dims # Set input dimension
    
    modelnames = []
    for i in args['primary_classes']:
        modelnames.append(f'{param["label"]}_class_{i}')
    
    device = param['deploy_device'] if 'deploy_device' in param else param['device']
    models, device = dbnf.load_models(param=param, modelnames=modelnames,
                        modeldir=f"{args['modeldir']}/{param['label']}", device=device)
    
    # Turn on eval!
    for i in range(len(models)):
        models[i].eval()
    
    def func_predict(x):
        return dbnf.predict(x.to(device), models)

    if return_model == False:
        return func_predict
    else:
        return func_predict, models


def pred_xgb(args, param, feature_names=None, return_model=False):
    
    print(f'Evaluate [{param["label"]}] model ...')
    
    filename, N_trees = aux.create_model_filename_xgb(path=f"{args['modeldir']}/{param['label']}",
                            label=param['label'], epoch=param['readmode'], filetype='.pkl')
    
    with open(filename, 'rb') as file:
        model = pickle.load(file)['model']
    
    def func_predict(x):
        pred = model.predict(xgboost.DMatrix(data = x, feature_names=feature_names, nthread=-1), 
                             iteration_range=(0,N_trees))
    
        if len(pred.shape) > 1: pred = pred[:, args['signal_class']]
        return pred

    if return_model == False:
        return func_predict
    else:
        return func_predict, model


def pred_xgb_scalar(args, param, feature_names=None, return_model=False):
    
    print(f'Evaluate [{param["label"]}] model ...')
    
    filename, N_trees = aux.create_model_filename_xgb(path=f"{args['modeldir']}/{param['label']}",
                            label=param['label'], epoch=param['readmode'], filetype='.pkl')
    
    with open(filename, 'rb') as file:
        model = pickle.load(file)['model']
    
    def func_predict(x):
        pred = model.predict(xgboost.DMatrix(data = x, feature_names=feature_names, nthread=-1), 
                             iteration_range=(0,N_trees))
        return pred
    
    if return_model == False:
        return func_predict
    else:
        return func_predict, model


def pred_xgb_logistic(args, param, feature_names=None, return_model=False):
    """
    Same as pred_xgb_scalar but a sigmoid function applied
    """
    
    print(f'Evaluate [{param["label"]}] model ...')
    
    filename, N_trees = aux.create_model_filename_xgb(path=f"{args['modeldir']}/{param['label']}",
                            label=param['label'], epoch=param['readmode'], filetype='.pkl')
    
    with open(filename, 'rb') as file:
        model = pickle.load(file)['model']
    
    def func_predict(x):
        
        logits = model.predict(xgboost.DMatrix(data = x, feature_names=feature_names, nthread=-1), 
                             iteration_range=(0,N_trees))  
        return aux.sigmoid(logits)
    
    if return_model == False:
        return func_predict
    else:
        return func_predict, model


def pred_flr(args, param):

    print(f'Evaluate [{param["label"]}] model ...')
    
    with open(f"{args['modeldir']}/{param['label']}/{param['label']}_0.pkl", 'rb') as file:
        model = pickle.load(file)
    
    b_pdfs    = model['b_pdfs']
    s_pdfs    = model['s_pdfs']
    bin_edges = model['bin_edges']
    
    def func_predict(x):
        return flr.predict(x, b_pdfs, s_pdfs, bin_edges)
    
    return func_predict

