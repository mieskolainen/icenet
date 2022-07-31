# Electron ID steering code
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

# icenet system paths
import sys
sys.path.append(".")

import numpy as np

# icenet
from icenet.tools import process
from icenet.tools import aux
from icenet.tools import plots
from icenet.tools import prints

# iceid
from iceid import common


def ele_mva_classifier(data, args=None):
    """
    External classifier directly from the root tree
    """
    varname = 'ele_mva_value_depth15'

    print(f'\nEvaluate <{varname}> classifier ...')
    try:
        y    = np.array(data.y, dtype=float)
        yhat = np.array(data.x[:, data.ids.index(varname)], dtype=float)

        return aux.Metric(y_true=y, y_pred=yhat, weights=data.w if args['reweight'] else None)
    except:
        raise Exception(__name__ + f'.ele_mva_classifier: Problem with <{varname}>')

# Main function
#
def main() :
    
  cli, cli_dict  = process.read_cli()
  runmode   = cli_dict['runmode']
  
  args, cli = process.read_config(config_path=f'configs/eid', runmode=runmode)
  X,Y,W,ids = process.read_data(args=args, func_loader=common.load_root_file, runmode=runmode) 

  if runmode == 'train' or runmode == 'eval':
    data = process.process_data(args=args, X=X, Y=Y, W=W, ids=ids, func_factor=common.splitfactor, mvavars='configs.eid.mvavars', runmode=runmode)
    
  if   runmode == 'train':
    prints.print_variables(X=data['trn']['data'].x, W=data['trn']['data'].w, ids=data['trn']['data'].ids)
    process.make_plots(data=data['trn'], args=args)
    process.train_models(data_trn=data['trn'], data_val=data['val'], args=args)

  elif runmode == 'eval':

    # ----------------------------
    # Evaluate external classifiers
    met_elemva = ele_mva_classifier(data=data['tst']['data_kin'], args=args)

    # Add to the stack
    process.roc_mstats.append(met_elemva)
    process.roc_labels.append('elemva15')
    # ----------------------------
    
    prints.print_variables(X=data['tst']['data'].x, W=data['tst']['data'].w, ids=data['tst']['data'].ids)
    process.evaluate_models(data=data['tst'], args=args)

  print(__name__ + ' [done]')

if __name__ == '__main__' :
   main()
