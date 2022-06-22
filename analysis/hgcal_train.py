# HGCAL [TRAINING] steering code
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

# icenet system paths
import sys
sys.path.append(".")

# icenet
from icenet.tools import process
from icenet.tools import aux
from icenet.tools import plots

# icehgcal
from icehgcal import common
from configs.hgcal.mvavars import *

# Main function
#
def main() :
    
    args, cli = process.read_config(config_path='./configs/hgcal')
    data      = process.read_data(args=args, func_loader=common.load_root_file, func_factor=common.splitfactor,
        train_mode=True, imputation_vars=globals()[args['imputation_param']['var']])

    process.make_plots(data=data['trn'], args=args)
    process.train_models(data_trn=data['trn'], data_val=data['val'], args=args)
    
    print(__name__ + ' [done]')

if __name__ == '__main__' :
   main()
