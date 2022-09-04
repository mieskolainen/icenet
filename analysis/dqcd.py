# DQCD steering code
# 
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

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

# icedqcd
from icedqcd import common


# -----------------------------------------------------------------------
def optimize_selection(info, args):

    """
    Work in progress
    """

    # Prepare output folders
    targetfile = f'{args["plotdir"]}/eval/eval_results.pkl'
    resdict = pickle.load(open(targetfile, 'rb'))

    # Optimize
    print(__name__ + f'.optimize_models: Running ...')
    print('info')
    pprint(info)
    print('resdict')
    pprint(resdict)


    # MVA-model labels
    print(resdict['roc_labels']['inclusive'])


    # Background efficiency target
    B_target_eff = 1e-4

    L_int = 30*(1E3) # [pico(barn)^{-1}]

    print(f'L = {L_int} pb^{{-1}}')

    # --------------------------------------------------------
    # Total background estimate
    B_tot = 0
    c     = 0
    
    # Loop over sub-process classes
    for name in info[f"class_{c}"].keys():

      print(name)
      proc    = info[f"class_{c}"][name]

      xs      = proc['yaml']["xs"]
      eff_acc = proc['eff_acc']
      N       = eff_acc * xs * L_int
      B_tot  += N
      
      print(f'xs = {xs:0.1f} pb, trigger x precuts: eff*acc={eff_acc:0.4f}, <N> = {N:0.3E}')
      print('')

    # --------------------------------------------------------------------
    # Signal estimate per model point

    MVA_model_index = 0

    c     = 1
    B     = np.zeros(len(info[f"class_{c}"].keys()))
    S     = np.zeros(len(B))
    names = len(S) * [None]

    fig,ax = plt.subplots()

    # Loop over different signal model points
    i = 0
    for name in info[f"class_{c}"].keys():

      print(name)

      names[i]    = name
      proc        = info[f"class_{c}"][name]
      #print(resdict['roc_mstats'].keys())

      # ----------------
      # Pick model parameters
      model_param = proc['yaml']['model_param']

      print(model_param)
      key     = f"$m = {model_param['m']}$ AND $c\\tau = {model_param['ctau']}$"
      roc_obj = resdict['roc_mstats'][key][MVA_model_index]

      x,y     = roc_obj.fpr,roc_obj.tpr
      y_err   = np.zeros(len(y))
      for k in range(len(y_err)):
        y_err[k] = np.std(roc_obj.tpr_bootstrap[:,k]) + 1e-3

      # -----------------
      # Interpolate ROC-curve
      """
      func        = interpolate.interp1d(x,y,'linear')
      xval = np.linspace(0,1,1000)
      plt.plot(xval, func(xval), label=f'{model_param}')
      """

      # -----------------
      # Fit the ROC-curve
      #def func(x, a, b):
      #  # Formulas 4, (14):
      #  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5570585/pdf/nihms-736507.pdf
      #  F_inv = 1/a * ncx2.ppf(x,  df=1, nc=b, loc=0, scale=1)
      #  return ncx2.cdf(F_inv, df=1, nc=a*b, loc=0, scale=1)

      def func(x, a, b):
        """
        Bi-normal function
        """
        #b = sd0 / sd1
        #a = (mu1 - mu0) / sd1
        return 1 - norm.cdf(b*norm.ppf(1-x) - a)

      popt, pcov = curve_fit(f=func, xdata=x, ydata=y, sigma=y_err)

      xval = np.logspace(-8, 0, 1000)
      plt.plot(xval, func(xval, *popt), label=f'(m: {model_param["m"]}, ctau: {model_param["ctau"]}) fit: {tuple(popt)}')
      
      print(popt)

      # -----------------
      #yhat        = func(B_target_eff)
      yhat        = func(B_target_eff, *popt)

      MVA_eff     = np.array([B_target_eff, yhat])

      print(MVA_eff)

      # ----------------
      # Compute background and signal event count
      B[i]        = B_tot * MVA_eff[0]

      # ----------------
      xs          = proc['yaml']["xs"]
      eff_acc     = proc['eff_acc']
      # ----------------

      S[i]        = eff_acc * xs * L_int * MVA_eff[1] 

      i += 1

    plt.xscale('log')
    plt.xlim([1e-6, 1])
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.savefig('rocs.pdf')

    print('')

    # ------------------
    # Discovery significance

    for i in range(len(S)):

      ds = S[i] / np.sqrt(B[i])

      #print(f'MVA_eff = {MVA_eff} [background, signal]')
      print(names[i])
      print(f'<S>/sqrt[<B>] = {S[i]:0.1f}/sqrt[{B[i]}] = {ds:0.1f}')
      print('')


    # print(roc_obj.thresholds)


# Main function
#
def main() :
  
  cli, cli_dict  = process.read_cli()
  runmode        = cli_dict['runmode']
  
  args, cli      = process.read_config(config_path=f'configs/dqcd', runmode=runmode)
  X,Y,W,ids,info = process.read_data(args=args, func_loader=common.load_root_file, runmode=runmode) 

  if   runmode == 'train' or runmode == 'eval':
    data = process.read_data_processed(X=X,Y=Y,W=W,ids=ids,
      funcfactor=common.splitfactor,mvavars='configs.dqcd.mvavars',runmode=runmode,args=args)
    
  if   runmode == 'train':
    prints.print_variables(X=data['trn']['data'].x, W=data['trn']['data'].w, ids=data['trn']['data'].ids)
    process.make_plots(data=data['trn'], args=args)
    process.train_models(data_trn=data['trn'], data_val=data['val'], args=args)

  elif runmode == 'eval':
    prints.print_variables(X=data['tst']['data'].x, W=data['tst']['data'].w, ids=data['tst']['data'].ids)
    process.evaluate_models(data=data['tst'], args=args)
      
  elif runmode == 'optimize':
    optimize_selection(info=info, args=args)

  print(__name__ + ' [done]')

if __name__ == '__main__' :
   main()


