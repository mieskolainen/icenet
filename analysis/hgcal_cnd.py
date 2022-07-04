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

def main() :
    
  cli, cli_dict  = process.read_cli()
  runmode   = cli_dict['runmode']
  
  args, cli = process.read_config(config_path=f'configs/hgcal', runmode=runmode)

  impute_vars = globals()[args['imputation_param']['var']] if runmode == 'genesis' else None
  data = process.read_data(args=args, func_loader=common.load_root_file, impute_vars=impute_vars, runmode=runmode) 
  
  data = process.process_data(args=args, data=data, func_factor=common.splitfactor, runmode=runmode)

  if   runmode == 'train':
      #prints.print_variables(X=data['trn']['data'].x, W=data['trn']['data'].w, ids=data['trn']['data'].ids)
      #process.make_plots(data=data['trn'], args=args)
      process.train_models(data_trn=data['trn'], data_val=data['val'], args=args)
  
  elif runmode == 'eval':
      process.evaluate_models(data=data['tst'], args=args)

  print(__name__ + ' [done]')

if __name__ == '__main__' :
   main()
