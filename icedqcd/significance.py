# Compute S/sqrt[B]
# 
# m.mieskolainen@imperial.ac.uk, 2023

import numpy as np
import awkward as ak
from tqdm import tqdm
import pickle
from pprint import pprint
import os

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.stats import ncx2,norm

from icenet.tools import aux
from icefit import cortools

#from PyPDF2 import PdfFileMerger


latex_header = \
"""
\\documentclass{article}
% General document formatting
\\usepackage[margin=0.7in]{geometry}
\\usepackage[parfill]{parskip}
\\usepackage[utf8]{inputenc}
  
% Related to math
\\usepackage{amsmath,amssymb,amsfonts,amsthm}

\\begin{document}
"""

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

def optimize_selection(args):

  def dprint(string, mode='a'):
    print(string)
    with open(latex_filename, mode) as f:
      f.write(f'{string} \n')

  latex_path     = aux.makedir(f'{args["plotdir"]}/eval/significance')
  latex_filename = f'{latex_path}/optimize.tex'

  """
  Work in progress
  """

  dprint('', 'w')
  dprint(latex_header)

  # --------------------------------------------------------
  ## Main parameters

  # Integrated luminosity
  L_int        = 41.0*(1000) # in inverse picobarns
  
  # Background efficiency target
  B_target_eff = 1e-4
  # --------------------------------------------------------

  dprint(f'$L = {L_int:0.0f} \\; \\text{{pb}}^{{-1}}$ \\\\')
  dprint('')

  # Prepare output folders
  targetfile = f'{args["plotdir"]}/eval/eval_results.pkl'
  resdict = pickle.load(open(targetfile, 'rb'))
  info    = resdict['info']

  print(__name__ + f'.optimize_models: Running ...')

  print('resdict:')
  pprint(resdict)


  # MVA-model labels
  MVA_model_names = resdict['roc_labels']['inclusive']


  # --------------------------------------------------------
  # Total background estimate
  B_tot        = 0
  c            = 0
  
  # Loop over sub-process classes
  B_xs_tot     = 0

  B_trg_eA_xs  = 0
  B_cut_eA_xs  = 0
  B_acc_eff_xs = 0

  dprint('\\tiny')
  dprint('\\begin{tabular}{l|l|ccc|c}')
  dprint('Background process & input xs $\\sigma$ [pb] & $\\epsilon A$(trg) & $\\epsilon$(cut) & $\\epsilon A$(trg x cut) & $\\langle B \\rangle$ \\\\')
  dprint('\\hline')

  for name in info[f"class_{c}"].keys():

    # ----------
    proc          = info[f"class_{c}"][name]
    # ----------

    xs            = proc['yaml']["xs"]    # Cross section
    
    trg_eA        = proc['cut_stats']['filterfunc']['after'] / proc['cut_stats']['filterfunc']['before']
    cut_eA        = proc['cut_stats']['cutfunc']['after']    / proc['cut_stats']['cutfunc']['before']
    eff_acc       = trg_eA * cut_eA       # combined eff x Acc
    N             = eff_acc * xs * L_int  # 
    
    B_tot        += N

    B_trg_eA_xs  += trg_eA  * xs
    B_cut_eA_xs  += cut_eA  * xs
    B_acc_eff_xs += eff_acc * xs
    B_xs_tot     += xs

    dprint(f'{name.replace("_", "-")} & {xs:0.1f} & {trg_eA:0.3f} & {cut_eA:0.3f} & {eff_acc:0.3f} & {N:0.1E} \\\\')

  B_trg_eA  = B_trg_eA_xs  / B_xs_tot
  B_cut_eA  = B_cut_eA_xs  / B_xs_tot
  B_acc_eff = B_acc_eff_xs / B_xs_tot
  
  dprint(f'\\hline')
  dprint(f'Total & {B_xs_tot:0.1f} & {B_trg_eA:0.3f} & {B_cut_eA:0.3f} & {B_acc_eff:0.3f} & {B_tot:0.1E} \\\\')
  dprint('\\end{tabular}')
  dprint('\\\\')
  dprint('')

  # --------------------------------------------------------------------
  # Signal estimate per model point

  keys = list(resdict['roc_mstats'].keys())
  num_MVA_models = len(resdict['roc_mstats'][keys[0]])

  for MVA_model_index in range(num_MVA_models):

    c  = 1 # Class
    S  = np.zeros(len(info[f"class_{c}"].keys()))
    B  = np.zeros(len(S))
    xs = np.zeros(len(S))

    S_trg_eA = np.zeros(len(S))
    S_cut_eA = np.zeros(len(S))

    MVA_eff  = np.zeros((len(B), 2))
    names    = len(S) * [None]

    # Loop over different signal model points
    i = 0

    # Create and instance of PdfFileMerger() class
    # merger = PdfFileMerger()

    for name in info[f"class_{c}"].keys():

      #print(name)

      names[i]  = name.replace('_', '-')
      proc      = info[f"class_{c}"][name]
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

      x,y       = roc_obj.fpr,roc_obj.tpr
      y_err     = np.ones(len(y))

      #for k in range(len(y_err)):
      #  y_err[k] = np.std(roc_obj.tpr_bootstrap[:,k])

      # -----------------
      # Interpolate ROC-curve
      
      func    = interpolate.interp1d(x, y, 'linear')

      ## Pick ROC-working point from the interpolation
      xval    = np.logspace(-8, 0, int(1e4))
      yhat    = func(B_target_eff)

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

      ## Pick ROC-working point from the analytic fit
      #xval = np.logspace(-8, 0, 1000)
      #yhat = func_binormal(B_target_eff, *popt)

      # -----------------
      # Construct efficiency for background, signal

      MVA_eff[i,:] = np.array([B_target_eff, yhat])
      
      # ----------------
      # Compute background event count
      B[i]         = B_tot * MVA_eff[i,0]

      # ----------------
      xs[i]        = proc['yaml']["xs"]
      eff_acc      = proc['eff_acc']

      S_trg_eA[i]  = proc['cut_stats']['filterfunc']['after'] / proc['cut_stats']['filterfunc']['before']
      S_cut_eA[i]  = proc['cut_stats']['cutfunc']['after']    / proc['cut_stats']['cutfunc']['before']
      # ----------------

      # Compute signal event count 
      S[i]         = S_trg_eA[i] * S_cut_eA[i] * xs[i] * L_int * MVA_eff[i,1]

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
          plt.ylabel('True positive rate $1 - \\beta$ (signal efficiency)', fontsize=9)
          plt.title(f'{names[i]}', fontsize=7)
          plt.legend(loc='lower right', fontsize=7)

        if k == 1:
          plt.xscale('log')
          xmin = 1e-4
          plt.xlim([xmin, 1])
          ax[k].set_aspect(-np.log10(xmin))
          plt.title(fit_label, fontsize=7)
          plt.xlabel('False positive rate $\\alpha$ (background efficiency)', fontsize=9)

        plt.ylim([0,1])
      
      pdf_filename = f'{path}/{roc_path}.pdf'
      plt.savefig(pdf_filename, bbox_inches='tight')
      plt.close()
      
      #merger.append(pdf_filename)
      # ------------------------------------

      i += 1

    #merger.write(f'{path}/ROC_all.pdf')
    print('')

    # ------------------
    # Discovery significance

    dprint(f'MVA model: {MVA_model_names[MVA_model_index]} \\\\')
    dprint('')
    dprint('\\tiny')
    dprint('\\begin{tabular}{lc||c|ccc|cc|c}')
    dprint('Signal model point & xs $\\sigma$ [pb] & $B$: $\\epsilon$(MVA) & $S$: $\\epsilon A$(trg) & $S$: $\\epsilon$(cut) & $S$: $\\epsilon$(MVA) & $\\langle B \\rangle$ & $\\langle S \\rangle$ & $\\langle S \\rangle / \\sqrt{{ \\langle B \\rangle }}$ \\\\')
    dprint('\\hline')
    
    for i in range(len(S)):

      # Gaussian limit
      ds   = S[i] / np.sqrt(B[i])

      line = f'{names[i]} & {xs[i]:0.1f} & {MVA_eff[i,0]:0.1E} & {S_trg_eA[i]:0.2f} & {S_cut_eA[i]:0.2f} & {MVA_eff[i,1]:0.2E} & {B[i]:0.1E} & {S[i]:0.1E} & {ds:0.1E} \\\\'
      dprint(line)
    
    # print(roc_obj.thresholds)
    dprint('\\end{tabular}')
    dprint('')
    
  dprint('\\end{document}')

  # Compile latex
  os.system(f'pdflatex -output-directory {latex_path} {latex_filename}')

