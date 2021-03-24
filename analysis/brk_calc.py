# B/RK analysis [CALCULATION] code
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

# icenet system paths
import _icepaths_

import pickle
import xgboost
import matplotlib as plt
import os
import torch
import argparse

import numpy as np
import numba
from numba import jit

# icenet
import iceplot
from icenet.tools import aux
from icenet.tools import io
from icenet.tools import prints
from icenet.deep import dopt

# icebrk
from icebrk import common
from icebrk import loop
from icebrk import histos
from icebrk import features


# Main function
#
def main() :
    
    ### Get input
    paths, args, cli, iodir = common.init()
    VARS = features.generate_feature_names(args['MAXT3'])
    modeldir = f'./checkpoint/brk/{args["config"]}/'; os.makedirs(modeldir, exist_ok = True)

    # ====================================================================
    print('\nLoading AI/ML models ...')

    if args['varnorm'] == 'zscore':
        X_mu, X_std = pickle.load(open(modeldir + '/zscore.dat', 'rb'))

    # ** Standardize input **
    def standardize(X):
        if args['varnorm'] == 'zscore':
            return io.apply_zscore(io.checkinfnan(X), X_mu, X_std)
        else:
            return X
    
    # ====================================================================
    # Evaluate models

    ### DEEPSETS
    DEPS_model = aux.load_torch_checkpoint(path=modeldir, label=args['deps_param']['label'], epoch=args['readmode'])

    DEPS_model, device = dopt.model_to_cuda(DEPS_model, device_type=args['deps_param']['device'])
    DEPS_model.eval() # Turn on eval mode!

    def func_predict_A(X):

        M = args['MAXT3']
        D = features.getdimension()

        # Transform to matrix shape
        Y  = standardize(X)
        X_ = aux.longvec2matrix(Y, M, D)

        X_ptr = torch.from_numpy(X_).type(torch.FloatTensor).to(device)
        y = DEPS_model.softpredict(X_ptr).detach().cpu().numpy()
        return io.checkinfnan(y)
    
    ### MAXOUT
    DMAX_model = aux.load_torch_checkpoint(path=modeldir, label=args['dmax_param']['label'], epoch=args['readmode'])
    
    DMAX_model, device = dopt.model_to_cuda(DMAX_model, device_type=args['dmax_param']['device'])
    DMAX_model.eval() # Turn on eval mode!
    
    def func_predict_B(X):
        X_ptr = torch.from_numpy(standardize(X)).type(torch.FloatTensor).to(device)
        y = DMAX_model.softpredict(X_ptr).detach().cpu().numpy()
        return io.checkinfnan(y)

    ### XGB
    XGB_model = pickle.load(open(modeldir + '/XGB_model.dat', 'rb'))
    def func_predict_C(X):
        y = XGB_model.predict(xgboost.DMatrix(data = standardize(X)))
        return io.checkinfnan(y)


    func_predict = [func_predict_A, func_predict_B, func_predict_C]

    # ========================================================================
    ### Event loop

    ## Binary matrix
    BMAT = aux.generatebinary(args['MAXT3'], args['MAXN'])
    print(f'POWERSET [NCLASS = {BMAT.shape[0]}]:')
    prints.print_colored_matrix(BMAT)
    
    
    # ========================================================================
    # MC as simulation

    output = loop.process(paths=paths, func_predict=[],
        isMC=True, MAXT3=args['MAXT3'], MAXN=args['MAXN'], WNORM=args['WNORM'],
        MAXEVENTS=args['MAXEVENTS'], VERBOSE=args['VERBOSE'], SUPERSETS=args['SUPERSETS'], BMAT=BMAT, hd5dir=iodir, outputXY=True)
    
    # Save it for the evaluation
    pickle.dump(output, open(iodir + 'MC_output.pkl', 'wb'))
    
    
    # ========================================================================
    # MC as synthetic DATA

    output = loop.process(paths=paths, func_predict=func_predict,
        isMC=False, MAXT3=args['MAXT3'], MAXN=args['MAXN'], WNORM=args['WNORM'],
        MAXEVENTS=args['MAXEVENTS'], VERBOSE=args['VERBOSE'], SUPERSETS=args['SUPERSETS'], BMAT=BMAT, hd5dir=iodir, outputXY=False, outputP=True)
        
    # Save it for the evaluation
    pickle.dump(output, open(iodir + 'DA_output.pkl', 'wb'))

    # ========================================================================


    print('\n' + __name__+ ' DONE')



if __name__ == '__main__' :

    main()
