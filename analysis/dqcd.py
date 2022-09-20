# DQCD steering code
# 
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.stats import ncx2,norm

import pickle
import numpy as np

# icenet system paths
import sys
sys.path.append(".")

from pprint import pprint
# icenet
from icenet.tools import process
from icenet.tools import prints
from icenet.tools import aux
from icefit import cortools

from analysis import dqcd_apply


# icedqcd
from icedqcd import common


# Main function
#
def main():
  
  cli, cli_dict  = process.read_cli()
  runmode        = cli_dict['runmode']

  args, cli      = process.read_config(config_path=f'configs/dqcd', runmode=runmode)
  
  if runmode in ['genesis', 'train', 'mode']:  
    X,Y,W,ids,info = process.read_data(args=args, func_loader=common.load_root_file, runmode=runmode) 

  if runmode in ['train', 'eval']:
    data = process.read_data_processed(X=X,Y=Y,W=W,ids=ids,
      funcfactor=common.splitfactor,mvavars='configs.dqcd.mvavars',runmode=runmode,args=args)
  
  if   runmode == 'train':
    prints.print_variables(X=data['trn']['data'].x, W=data['trn']['data'].w, ids=data['trn']['data'].ids)
    process.make_plots(data=data['trn'], args=args)
    process.train_models(data_trn=data['trn'], data_val=data['val'], args=args)

  elif runmode == 'eval':
    prints.print_variables(X=data['tst']['data'].x, W=data['tst']['data'].w, ids=data['tst']['data'].ids)
    process.evaluate_models(data=data['tst'], info=info, args=args)
    
  elif runmode == 'optimize':
    dqcd_apply.optimize_selection(args=args)
  
  print(__name__ + ' [done]')

if __name__ == '__main__' :
   main()
