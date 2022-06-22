# DQCD [TRAINING] steering code
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

# icenet system paths
import sys
sys.path.append(".")

# icenet
from icenet.tools import process
from icenet.tools import prints


# icedqcd
from icedqcd import common
#from configs.dqcd.mvavars import *

# Main function
#
def main() :
    
    args, cli = process.read_config(config_path='./configs/dqcd')
    data      = process.read_data(args=args, func_loader=common.load_root_file, func_factor=common.splitfactor, 
        train_mode=True, imputation_vars=None)

    ### Print ranges
	prints.print_variables(X=data['trn']['data'].x, W=data['trn']['data'].w, ids=data['trn']['data'].ids)
    
    #process.make_plots(data=data['trn'], args=args)
    process.train_models(data_trn=data['trn'], data_val=data['val'], args=args)

    print(__name__ + ' [done]')

if __name__ == '__main__' :
   main()