# Electron ID [EVALUATION] code
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

# icenet system paths
import sys
sys.path.append(".")

# icenet
from icenet.tools import process

# iceid
from iceid import common
from configs.eid.mvavars import *

import numpy as np

def ele_mva_classifier(data, weights=None, args=None):
    """
    External classifier directly from the root tree
    """
    varname = 'ele_mva_value_depth15'

    print(f'\nEvaluate <{varname}> classifier ...')
    try:
        y    = np.array(data.y, dtype=float)
        yhat = np.array(data.x[:, data.ids.index(varname)], dtype=float)

        return aux.Metric(y_true=y, y_soft=yhat, weights=data.w)
    except:
        raise Exception(__name__ + f'.ele_mva_classifier: Problem with <{varname}>')


# Main function
#
def main() :

    args, cli = process.read_config(config_path='./configs/eid')
    data      = process.read_data(args=args, func_loader=common.load_root_file, func_factor=common.splitfactor,
        train_mode=False, imputation_vars=globals()[args['imputation_param']['var']])

    # ----------------------------
    # Evaluate external classifiers
    met_elemva = ele_mva_classifier(data=data['tst']['data'])
    
    # Add to the stack
    process.roc_mstats.append(met_elemva)
    process.roc_labels.append('elemva15')
    # ----------------------------
    
    process.evaluate_models(data=data['tst'], args=args)

    print(__name__ + ' [Done]')

if __name__ == '__main__' :
    main()
