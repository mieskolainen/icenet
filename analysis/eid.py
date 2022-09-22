# Electron ID steering code
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

import sys
sys.path.append(".")

# Configure plotting backend
import matplotlib
matplotlib.use('Agg')

from icenet.tools import process
from iceid import common

def main():
    args = process.generic_flow(rootname='eid', func_loader=common.load_root_file, func_factor=common.splitfactor)

if __name__ == '__main__' :
    main()


def ele_mva_classifier(data, args=None):
    """
    External classifier directly from the root tree
    (TBD: update this method to be compatible with the new workflow, change to function pointer)
    """
    varname = 'ele_mva_value_depth15'
    
    print(f'\nEvaluate <{varname}> classifier ...')
    try:
        y    = np.array(data.y, dtype=float)
        yhat = np.array(data.x[:, data.ids.index(varname)], dtype=float)

        return aux.Metric(y_true=y, y_pred=yhat, weights=data.w if args['reweight'] else None)
    except:
        raise Exception(__name__ + f'.ele_mva_classifier: Problem with <{varname}>')

