# High Level Trigger steering code
# 
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

# icenet system paths
import sys
sys.path.append(".")

# icenet
from icenet.tools import process
from icenet.tools import prints

# icetrg
from icetrg import common


# Main function
#
def main() :
        
  cli, cli_dict = process.read_cli()
  runmode     = cli_dict['runmode']
  args, cli   = process.read_config(config_path=f'configs/trg', runmode=runmode)
  
  X,Y,W,ids   = process.read_data(args=args, func_loader=common.load_root_file, runmode=runmode) 
  
  if runmode == 'train' or runmode == 'eval':
    data = process.process_data(args=args, X=X, Y=Y, W=W, ids=ids, func_factor=common.splitfactor, mvavars='configs.trg.mvavars', runmode=runmode)
        
  if   runmode == 'train':
    prints.print_variables(X=data['trn']['data'].x, W=data['trn']['data'].w, ids=data['trn']['data'].ids)
    process.make_plots(data=data['trn'], args=args)
    process.train_models(data_trn=data['trn'], data_val=data['val'], args=args)

  elif runmode == 'eval':
    prints.print_variables(X=data['tst']['data'].x, W=data['tst']['data'].w, ids=data['tst']['data'].ids)
    process.evaluate_models(data=data['tst'], args=args)
  
  print(__name__ + ' [done]')

if __name__ == '__main__' :
    main()
