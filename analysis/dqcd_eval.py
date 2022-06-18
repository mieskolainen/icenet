# DQCD [EVALUATION] steering code
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

# icenet system paths
import sys
sys.path.append(".")

# icenet
from icenet.tools import process

# icedqcd
from icedqcd import common
#from configs.dqcd.mvavars import *

# Main function
#
def main() :
    
    args, cli = process.read_config(config_path='./configs/dqcd')
    data      = process.read_data(args=args, func_loader=common.load_root_file, func_factor=common.splitfactor, 
        train_mode=False, imputation_vars=None)

    process.evaluate_models(data=data['tst'], args=args)

    print(__name__ + ' [done]')

if __name__ == '__main__' :
   main()