# HGCAL [EVALUATION] code
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

# icenet system paths
import sys
sys.path.append(".")

# icenet
from icenet.tools import process

# icehgcal
from icehgcal import common
from configs.hgcal.mvavars import *

import numpy as np

# Main function
#
def main() :

    args, cli = process.read_config(config_path='./configs/hgcal')
    data      = process.read_data(args=args, func_loader=common.load_root_file, func_factor=common.splitfactor,
        train_mode=False, imputation_vars=globals()[args['imputation_param']['var']])
    
    process.evaluate_models(data=data['tst'], args=args)

    print(__name__ + ' [Done]')

if __name__ == '__main__' :
    main()
