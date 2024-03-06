# Generic model evaluation wrapper functions
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import torch_geometric

import pickle
import copy

# xgboost
import xgboost

# icenet
from icenet.tools import stx
from icenet.tools import aux
from icenet.tools import aux_torch

from icenet.algo  import flr
from icenet.deep  import bnaf
from icenet.deep  import optimize
from icenet.deep  import dbnf


def pred_cut(args, param):

    print(__name__ + f'.pred_cut: Evaluate <{param["label"]}> cut model ...')

    # Get feature name variables
    index = args['features'].index(param['variable'])
    
    def func_predict(x):
        # Identity
        if param['transform'] is None:             
            return param['sign'] * x[...,index]
        # Transform via sigmoid from R -> [0,1]
        elif param['transform'] == 'sigmoid':
            return 1 / (1 + np.exp(param['sign'] * x[...,index]))
        # Numpy function, e.g. np.abs, np.tanh ...
        elif 'np.' in param['transform']:
            cmd = "param['sign'] * " + param['transform'] + f"(x[...,index])"
            return eval(cmd)
        else:
            raise Exception(__name__ + '.pred_cuts: Unknown transform chosen (check your syntax)')
    
    return func_predict

def pred_cutset(args, param):

    print(__name__ + f'.pred_cutset: Evaluate <{param["label"]}> fixed cutset model ...')
    cutstring = param['cutstring']
    print(f'cutstring: "{cutstring}"')

    # Expand model parameters into memory
    for p in param['model_param'].keys():
        exec(f"{p} = param['model_param']['{p}']")
    
    # Expand cut string
    variable = param['variable']
    expr = eval(f'f"{cutstring}"')

    print(f'cutstring expanded: {expr}')
    
    # Get feature name variables
    ids = args['features']
    
    def func_predict(x):
        y = stx.eval_boolean_syntax(expr=expr, X=x, ids=ids)
        return y.astype(float)
    
    return func_predict

def pred_graph_xgb(args, param, device='cpu'):
    
    print(__name__ + f'.pred_graph_xgb: Evaluate <{param["label"]}> model ...')
    
    graph_model = aux_torch.load_torch_checkpoint(path=args['modeldir'], \
        label=param['graph']['label'], epoch=param['graph']['readmode']).to(device)
    graph_model.eval() # Turn eval mode one!
    
    xgb_model   = pickle.load(open(aux.create_model_filename(path=args['modeldir'], \
        label=param['xgb']['label'], epoch=param['xgb']['readmode'], filetype='.dat'), 'rb'))

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

        pred = xgb_model.predict(xgboost.DMatrix(data = x_tot))
        if len(pred.shape) > 1: pred = pred[:, args['signal_class']]
        return pred
    
    return func_predict

def pred_torch_graph(args, param, batch_size=5000, return_model=False):
    
    print(__name__ + f'.pred_torch_graph: Evaluate <{param["label"]}> model ...')
    model         = aux_torch.load_torch_checkpoint(path=args['modeldir'], label=param['label'], epoch=param['readmode'])
    model, device = optimize.model_to_cuda(model, device_type=param['device'])
    
    model.eval() # ! Turn on eval mode!
    
    def func_predict(x):

        if isinstance(x, list):
            x_in = x
        else:
            x_in = [x]
        
        # Geometric type -> need to use batch loader
        loader = torch_geometric.loader.DataLoader(x_in, batch_size=batch_size, shuffle=False)

        # Predict in smaller batches not to overflow GPU memory
        for i, batch in enumerate(loader):
            y = model.softpredict(batch.to(device))[:, args['signal_class']].detach().cpu().numpy()
            y_tot = copy.deepcopy(y) if (i == 0) else np.concatenate((y_tot, y), axis=0)
        
        return y_tot

    if return_model == False:
        return func_predict
    else:
        return func_predict, model


def pred_torch_generic(args, param, return_model=False):
    
    print(__name__ + f'.pred_torch_generic: Evaluate <{param["label"]}> model ...')
    model         = aux_torch.load_torch_checkpoint(path=args['modeldir'], label=param['label'], epoch=param['readmode'])
    model, device = optimize.model_to_cuda(model, device_type=param['device'])
    model.eval() # ! Turn on eval mode!
    
    def func_predict(x):

        if not isinstance(x, dict):
            x_in = x.to(device)
        else:
            x_in = copy.deepcopy(x)
            for key in x_in.keys():
                x_in[key] = x_in[key].to(device)
        
        return model.softpredict(x_in)[:, args['signal_class']].detach().cpu().numpy()
    
    if return_model == False:
        return func_predict
    else:
        return func_predict, model


def pred_torch_scalar(args, param, return_model=False):
    
    print(__name__ + f'.pred_torch_scalar: Evaluate <{param["label"]}> model ...')
    model         = aux_torch.load_torch_checkpoint(path=args['modeldir'], label=param['label'], epoch=param['readmode'])
    model, device = optimize.model_to_cuda(model, device_type=param['device'])
    model.eval() # ! Turn on eval mode!
    
    def func_predict(x):
        
        if not isinstance(x, dict):
            x_in = x.to(device)
        else:
            x_in = copy.deepcopy(x)
            for key in x_in.keys():
                x_in[key] = x_in[key].to(device)
        
        return model.softpredict(x_in).detach().cpu().numpy()
    
    if return_model == False:
        return func_predict
    else:
        return func_predict, model

'''
def pred_xtx(args, param):

# Not implemented
'''

def pred_xgb(args, param, feature_names=None, return_model=False):
    
    print(__name__ + f'.pred_xgb: Evaluate <{param["label"]}> model ...')
    filename  = aux.create_model_filename(path=args['modeldir'], label=param['label'], epoch=param['readmode'], filetype='.dat')
    model = pickle.load(open(filename, 'rb'))
    
    def func_predict(x):
        pred = model.predict(xgboost.DMatrix(data = x, feature_names=feature_names, nthread=-1))
        if len(pred.shape) > 1: pred = pred[:, args['signal_class']]
        return pred

    if return_model == False:
        return func_predict
    else:
        return func_predict, model


def pred_xgb_scalar(args, param, feature_names=None, return_model=False):
    
    print(__name__ + f'.pred_xgb_scalar: Evaluate <{param["label"]}> model ...')
    filename  = aux.create_model_filename(path=args['modeldir'], label=param['label'], epoch=param['readmode'], filetype='.dat')
    model = pickle.load(open(filename, 'rb'))
    
    def func_predict(x):
        pred = model.predict(xgboost.DMatrix(data = x, feature_names=feature_names, nthread=-1))
        return pred
    
    if return_model == False:
        return func_predict
    else:
        return func_predict, model


def pred_xgb_logistic(args, param, feature_names=None, return_model=False):
    
    print(__name__ + f'.pred_xgb_logistic: Evaluate <{param["label"]}> model ...')
    filename  = aux.create_model_filename(path=args['modeldir'], label=param['label'], epoch=param['readmode'], filetype='.dat')
    model = pickle.load(open(filename, 'rb'))
    
    def func_predict(x):
        # Apply sigmoid function    
        return 1 / (1 + np.exp(- model.predict(xgboost.DMatrix(data = x, feature_names=feature_names, nthread=-1))))
    
    if return_model == False:
        return func_predict
    else:
        return func_predict, model


def pred_flow(args, param, n_dims, return_model=False):

    print(__name__ + f'.pred_flow: Evaluate <{param["label"]}> model ...')

    # Load models
    param['model_param']['n_dims'] = n_dims # Set input dimension
    
    modelnames = []
    for i in range(args['num_classes']):
        modelnames.append(f'{param["label"]}_class_{i}')
    
    models = dbnf.load_models(param=param, modelnames=modelnames, modeldir=args['modeldir'])
    
    def func_predict(x):
        return dbnf.predict(x, models)

    if return_model == False:
        return func_predict
    else:
        return func_predict, models


def pred_flr(args, param):

    print(__name__ + f'.pred_flr: Evaluate <{param["label"]}> model ...')
    
    b_pdfs, s_pdfs, bin_edges = pickle.load(open(args['modeldir'] + f'/{param["label"]}_0_.dat', 'rb'))
    def func_predict(x):
        return flr.predict(x, b_pdfs, s_pdfs, bin_edges)
    
    return func_predict

