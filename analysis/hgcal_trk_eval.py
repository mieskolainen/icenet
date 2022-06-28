# HGCAL [TRAINING] steering code
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

# icenet system paths
import sys
sys.path.append(".")

import pickle

# icenet
from icenet.tools import process
from icenet.tools import aux
from icenet.tools import plots
from icenet.tools import io

# icehgcal
from icehgcal import common
from configs.hgcal.mvavars import *

# Main function
#
def main() :
    
    args, cli = process.read_config(config_path='./configs/hgcal')
    #data      = process.read_data(args=args, func_loader=common.load_root_file, func_factor=common.splitfactor,
    #    train_mode=True, imputation_vars=globals()[args['imputation_param']['var']])

    ## Read data
    filename = "output/" + f"{cli.tag}.pkl"
    with open(filename, "rb") as fp:
        X = pickle.load(fp)

    X_trn, X_val, X_tst = io.split_data_simple(X=X, frac=args['frac'])
    data = {}
    data['tst'] = {'data': None, 'data_kin': None, 'data_deps': None, 'data_tensor': None, 'data_graph': X_tst}
    
    #process.make_plots(data=data['trn'], args=args)

    process.evaluate_models(data=data['tst'], args=args)
    
    print(__name__ + ' [done]')

if __name__ == '__main__' :
   main()
