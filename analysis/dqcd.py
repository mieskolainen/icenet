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
from icenet.tools import aux
from icefit import cortools


# icedqcd
from icedqcd import common


"""
def func_binormal(x, a, b):
  # Formulas 4, (14):
  # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5570585/pdf/nihms-736507.pdf
  F_inv = (1/a) * ncx2.ppf(1-x,  df=1, nc=b)
  return 1-ncx2.cdf(F_inv, df=1, nc=a*b)
"""

def func_binormal(x, a, b):  
  # "binormal" function
  # https://dpc10ster.github.io/RJafrocBook/binormal-model.html
  #
  # b = sd0 / sd1
  # a = (mu1 - mu0) / sd1
  return norm.cdf(a + b*norm.ppf(x))

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

      proc    = info[f"class_{c}"][name]

      xs      = proc['yaml']["xs"]
      eff_acc = proc['eff_acc']
      N       = eff_acc * xs * L_int
      B_tot  += N

      print(f'{name} && {xs:0.1f} pb && {eff_acc:0.4f} && {N:0.3E} \\\\')

    # --------------------------------------------------------------------
    # Signal estimate per model point

    for MVA_model_index in [0,1]:

      c       = 1
      B       = np.zeros(len(info[f"class_{c}"].keys()))
      S       = np.zeros(len(B))
      MVA_eff = np.zeros((len(B), 2))
      names   = len(S) * [None]

      # Loop over different signal model points
      i = 0

      from PyPDF2 import PdfFileMerger
      #Create and instance of PdfFileMerger() class
      merger = PdfFileMerger()

      for name in info[f"class_{c}"].keys():

        #print(name)

        names[i]    = name
        proc        = info[f"class_{c}"][name]
        #print(resdict['roc_mstats'].keys())

        # ----------------
        # Pick model parameters
        model_param = proc['yaml']['model_param']
        #print(model_param)

        key       = f"$m = {model_param['m']}$ AND $c\\tau = {model_param['ctau']}$"
        roc_obj   = resdict['roc_mstats'][key][MVA_model_index]
        roc_path  = resdict['roc_paths'][key][MVA_model_index]

        #print(resdict)

        roc_label = resdict['roc_labels'][key][MVA_model_index]

        x,y     = roc_obj.fpr,roc_obj.tpr
        y_err   = np.zeros(len(y))
        for k in range(len(y_err)):
          y_err[k] = np.std(roc_obj.tpr_bootstrap[:,k]) + 1e-2

        # -----------------
        # Interpolate ROC-curve
        """
        func        = interpolate.interp1d(x,y,'linear')
        xval = np.logspace(-8, 0, 1000)
        yhat        = func(B_target_eff)
        """

        # -----------------
        # Fit the ROC-curve

        popt, pcov = curve_fit(f=func_binormal, xdata=x, ydata=y, sigma=y_err)

        # -----------------
        # Create parameter estimate text label
        param     = np.round(np.array(popt),2)
        param_err = np.round(np.sqrt(np.diag(np.array(pcov))), 2)
        fit_label = f'fit: $\\Phi(a_0 + a_1\\Phi^{{-1}}(x)), ('
        for k in range(len(param)):
          fit_label += f'a_{{{k}}} = {param[k]} \\pm {param_err[k]}'
          if k < len(param)-1: fit_label += ','
        fit_label += ')$'
        # -----------------

        xval = np.logspace(-8, 0, 1000)
        yhat = func_binormal(B_target_eff, *popt)

        # -----------------
        # Construct efficiency for background, signal

        MVA_eff[i,:] = np.array([B_target_eff, yhat])
        
        # ----------------
        # Compute background and signal event count
        B[i]         = B_tot * MVA_eff[i,0]

        # ----------------
        xs           = proc['yaml']["xs"]
        eff_acc      = proc['eff_acc']
        # ----------------

        S[i]         = eff_acc * xs * L_int * MVA_eff[i,1]


        # ------------------------------------
        fig,ax = plt.subplots(1,2)
        alpha = 0.32

        for k in [0,1]:

          plt.sca(ax[k])

          auc_CI  = cortools.prc_CI(x=roc_obj.auc_bootstrap, alpha=alpha)
          auc_err = np.abs(auc_CI - roc_obj.auc)
          plt.plot(x,y, drawstyle='steps-mid', color=f'C{i}', label=f'{roc_label}, AUC: ${roc_obj.auc:0.3f} \\pm_{{{auc_err[0]:0.3f}}}^{{{auc_err[1]:0.3f}}}$')

          tpr_lo = cortools.percentile_per_dim(x=roc_obj.tpr_bootstrap, q=100*(alpha/2))
          tpr_hi = cortools.percentile_per_dim(x=roc_obj.tpr_bootstrap, q=100*(1-alpha/2))
          plt.fill_between(x, tpr_lo, tpr_hi, step='mid', alpha=0.4, color=f'C{i}', edgecolor='none') # vertical

          plt.plot(xval, func_binormal(xval, *popt), linestyle='-', color=(0.35,0.35,0.35), label='binormal fit')
          plt.plot(xval, xval, color=(0.5,0.5,0.5), linestyle=':')

          path = aux.makedir(f'{args["plotdir"]}/eval/significance/ROCfit/{roc_label}')

          if k == 0:
            plt.xscale('linear')
            plt.xlim([0, 1])
            ax[k].set_aspect('equal', 'box')
            plt.ylabel('True positive rate $1 - \\beta$ (signal efficiency)')
            plt.title(f'{names[i]}', fontsize=7)
            plt.legend(loc='lower right', fontsize=7)

          if k == 1:
            plt.xscale('log')
            xmin = 1e-4
            plt.xlim([xmin, 1])
            ax[k].set_aspect(-np.log10(xmin))
            plt.title(fit_label, fontsize=7)
            plt.xlabel('False positive rate $\\alpha$ (background efficiency)')

          plt.ylim([0,1])

        pdf_filename = f'{path}/{roc_path}.pdf'
        plt.savefig(pdf_filename, bbox_inches='tight')
        plt.close()
        
        merger.append(pdf_filename)
        # ------------------------------------

        i += 1

      merger.write(f'{path}/ROC_all.pdf')
      print('')

      # ------------------
      # Discovery significance

      print('signal model point && MVA $B_{\\epsilon}$ && MVA $S_{\\epsilon}$ && $S/\\sqrt{B}$ \\\\')
      for i in range(len(S)):

        ds = S[i] / np.sqrt(B[i])

        #print(f'MVA_eff = {MVA_eff} [background, signal]')
        print(f'{names[i]} && {MVA_eff[i,0]:0.3E} && {MVA_eff[i,1]:0.3E} && {S[i]:0.1f}/sqrt[{B[i]:0.1f}] = {ds:0.1f} \\\\')

      # print(roc_obj.thresholds)


# Main function
#
def main():
  
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


