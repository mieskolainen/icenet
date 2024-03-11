# Efficiency flows ++ (Work-in-progress)
# 
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import awkward as ak
from tqdm import tqdm
import pickle
from pprint import pprint
import os
from termcolor import colored, cprint
from datetime import datetime
from scanf import scanf

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.stats import ncx2,norm

from icenet.tools import aux, io
from icefit import cortools
from icedqcd import limits


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

def func_binormal2(x, a, b):
    """
    Formulas 4, (14):
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5570585/pdf/nihms-736507.pdf
    """
    F_inv = (1/a) * ncx2.ppf(1-x,  df=1, nc=b)
    return 1 - ncx2.cdf(F_inv, df=1, nc=a*b)


def func_binormal(x, a, b):  
    """
    binormal-function
    https://dpc10ster.github.io/RJafrocBook/binormal-model.html
    
    b = sd0 / sd1
    a = (mu1 - mu0) / sd1
    """
    return norm.cdf(a + b*norm.ppf(x))


def mask_eff(num_mask, y_true, class_id, weights, den_mask=None):
    """
    Masked columnar selection efficiency
    """
    
    if den_mask is None:
        den_mask = np.ones_like(num_mask)

    num = np.sum(weights[y_true == class_id]*num_mask[y_true == class_id])
    den = np.sum(weights[y_true == class_id]*den_mask[y_true == class_id])

    return num / den


def plot_ROC_fit(i, fpr, tpr, tpr_err, fpr_err, roc_obj, roc_label, names, args):
    """
    Plot ROC fits
    """
    
    xval = np.logspace(-8, 0, int(1e4))
    
    # Fit the ROC-curve
    try:
        popt, pcov = curve_fit(f=func_binormal, xdata=fpr, ydata=tpr)#, sigma=tpr_err)
    except:
        cprint(__name__ + f".plot_ROC_fit: Problem with curve_fit", 'red')
        print(f'fpr = {fpr}')
        print(f'tpr = {tpr}')
        return
    
    # -----------------
    # Create parameter estimate text label
    param     = np.round(np.array(popt),2)
    param_err = np.round(np.sqrt(np.diag(np.array(pcov))), 2)
    fit_label = f'fit: $\\Phi(a_0 + a_1\\Phi^{{-1}}(x)), ('
    for k in range(len(param)):
        fit_label += f'a_{{{k}}} = {param[k]} \\pm {param_err[k]}'
        if k < len(param)-1: fit_label += ','
    fit_label += ')$'
    
    # ------------------------------------
    fig,ax = plt.subplots(1,2)
    alpha  = 0.32
    
    for k in [0,1]:

        plt.sca(ax[k])

        auc_CI  = cortools.prc_CI(x=roc_obj.auc_bootstrap, alpha=alpha)
        auc_err = np.abs(auc_CI - roc_obj.auc)
        plt.plot(fpr, tpr, drawstyle='steps-mid', color=f'C{i}', label=f'{roc_label}, AUC: ${roc_obj.auc:0.3f} \\pm_{{{auc_err[0]:0.3f}}}^{{{auc_err[1]:0.3f}}}$')

        tpr_lo = cortools.percentile_per_dim(x=roc_obj.tpr_bootstrap, q=100*(alpha/2))
        tpr_hi = cortools.percentile_per_dim(x=roc_obj.tpr_bootstrap, q=100*(1-alpha/2))
        plt.fill_between(fpr, tpr_lo, tpr_hi, step='mid', alpha=0.4, color=f'C{i}', edgecolor='none') # vertical

        plt.plot(xval, func_binormal(xval, *popt), linestyle='-', color=(0.35,0.35,0.35), label='binormal fit')
        plt.plot(xval, xval, color=(0.5,0.5,0.5), linestyle=':')

        path = aux.makedir(f'{args["plotdir"]}/eval/significance/ROC_fit/{roc_label}')

        if k == 0:
            plt.xscale('linear')
            plt.xlim([0, 1])
            ax[k].set_aspect('equal', 'box')
            plt.ylabel('True positive rate $1 - \\beta$ (signal efficiency)', fontsize=9)
            plt.title(f'{names[i]}', fontsize=5)
            plt.legend(loc='lower right', fontsize=7)

        if k == 1:
            plt.xscale('log')
            xmin = 1e-6
            plt.xlim([xmin, 1])
            ax[k].set_aspect(-np.log10(xmin))
            plt.title(fit_label, fontsize=6)
            plt.xlabel('False positive rate $\\alpha$ (background efficiency)', fontsize=9)

        plt.ylim([0,1])

    pdf_filename = f'{path}/{roc_label}.pdf'
    plt.savefig(pdf_filename, bbox_inches='tight')
    plt.close()


def find_filter(resdict, model_param, args):
    """
    Find filter strings
    """
    model_POI      = args['limits_param']['model_POI']
    box_POI        = args['limits_param']['box_POI'] 
    
    ROC_filter_tag = args['limits_param']['ROC_filter_tag']
    BOX_filter_tag = args['limits_param']['BOX_filter_tag']
    GEN_filter_tag = args['limits_param']['GEN_filter_tag']
    
    ROC_filter_key = None
    BOX_filter_key = None
    GEN_filter_key = None
    param_point    = {}
    
    cprint(__name__ + f'.find_filter: Signal: {model_param}', 'yellow')
    
    # --------------
    
    for filter in resdict['roc_mstats'].keys():
        matches = 0
        for param_name in model_POI :
            
            if f'{param_name} = {model_param[param_name]:g}' in filter and ROC_filter_tag in filter:
                matches += 1
                param_point[param_name] = model_param[param_name]
        
        if matches == len(model_POI): # all parameters match
            ROC_filter_key = filter
            
            cprint(f'Found ROC filter: {ROC_filter_key}' , 'green')
            break
    
    if ROC_filter_key is None:
        cprint(f'ERROR: No matching ROC filter for the signal: {model_param}, exit', 'red')
        exit()
    
    # --------------
    
    for filter in resdict['roc_mstats'].keys():
        
        if BOX_filter_tag in filter:
            param_value = scanf(f"{BOX_filter_tag}: box({box_POI[0]} ~ %f)", filter)[0]
            
            if np.round(param_value, 3) == np.round(param_point[box_POI[1]], 3):
                BOX_filter_key = filter
                cprint(f'Found BOX filter: {BOX_filter_key}' , 'green')
                break
    
    if BOX_filter_key is None:
        cprint(f'ERROR: No matching BOX filter for the signal: {model_param}, exit', 'red')
        exit()
    
    # --------------
    
    for filter in resdict['roc_mstats'].keys():
        matches = 0
        for param_name in model_POI:
            
            if f'{param_name} = {model_param[param_name]:g}' in filter and GEN_filter_tag in filter:
                matches += 1
        
        if matches == len(model_POI): # all parameters match
            GEN_filter_key = filter
            cprint(f'Found GEN filter: {GEN_filter_key}' , 'green')
            break
    
    if GEN_filter_key is None:
        cprint(f'ERROR: No matching GEN filter for the signal: {model_param}, exit', 'red')
        exit()
    
    return ROC_filter_key, BOX_filter_key, GEN_filter_key, param_point


def optimize_selection(args):
    """
    Main program
    """
    
    cprint(__name__ + f'.optimize_selection: Running ...', 'yellow')
    
    def dprint(string, mode='a'):
        print(string)
        with open(latex_filename, mode) as f:
            f.write(string + '\n')

    latex_path     = aux.makedir(f'{args["plotdir"]}/eval/significance')
    latex_filename = f'{latex_path}/optimize.tex'
     
    # -----------------
    ## Main parameters

    # Integrated luminosity
    L_int        = args['limits_param']['luminosity'] # in pb^{-1}
    
    # Signal production cross section
    signal_xs    = args['limits_param']['signal_xs']  # in pb
    
    # Benchmark BR
    BR           = args['limits_param']['target_r']
    
    # Background ML cut efficiency target
    B_target_eff = args['limits_param']['B_target_eff']
    # -----------------
    
    dprint('', 'w')
    dprint(latex_header)
    
    dprint('\\Large')
    dprint('Cut efficiencies and discovery significance')
    dprint('\\small')
    dprint('\\\\')
    dprint(f'github.com/mieskolainen/icenet')
    dprint('\\\\')
    dprint(f"{datetime.now():%d-%m-%Y %H:%M}")
    dprint('\\\\')
    dprint('\\vspace{0.5em}')
    dprint('\\\\')
    
    dprint(f'$L = {L_int} \\; \\text{{pb}}^{{-1}}$ \\\\')
    dprint(f'$\\sigma_S$ = {signal_xs} \\text{{pb}} \\\\')
    dprint(f'$\\text{{BR}} = {BR}$ (benchmark) \\\\')
    dprint(f'$\\langle S \\rangle \\equiv L \\times (\\sigma_S \\times \\text{{BR}}) \\times \\epsilon A_S (\\text{{trg}}) \\times \\epsilon_S (\\text{{pre-cut}}) \\times \\epsilon_S (M\\text{{-box}}) \\times \\epsilon_S (\\text{{ML}})$ \\\\')
    dprint('\\vspace{1em}')
    dprint('\\\\')
    dprint(f'Signal scenario: {args["limits_param"]["cosmetics"]["portal"]} \\\\')
    dprint('\\vspace{1em}')
    dprint('\\\\')
    
    dprint('\\tiny')
    dprint('\\begin{tabular}{l|l|ccc|c}')
    dprint('Background process & input $\\sigma_B$ [pb] & $\\epsilon A$(trg) & $\\epsilon$(pre-cut) & $\\epsilon A$(trg $\\times$ cut) & $\\langle B \\rangle$ \\\\')
    dprint('\\hline')
    
    # -----------------
    # Load evaluation data
    targetfile = f'{args["plotdir"]}/eval/eval_results.pkl'
    with open(targetfile, 'rb') as file:
        resdict = pickle.load(file)
    info = resdict['info']

    # MVA-model labels
    MVA_model_names = resdict['roc_labels']['inclusive']

    background_ID = 0
    signal_ID     = 1

    # -----------------
    # Background estimate
    
    B_tot        = 0
    B_xs_tot     = 0
    B_trg_eA_xs  = 0
    B_cut_eA_xs  = 0
    B_acc_eff_xs = 0

    # Loop over background sub-processes
    for name in info[f"class_{background_ID}"].keys():
        
        proc          = info[f"class_{background_ID}"][name]
        
        xs            = proc['yaml']["xs"]            # Cross-section
        trg_eA        = proc['cut_stats']['filterfunc']['after'] / proc['cut_stats']['filterfunc']['before']
        cut_eA        = proc['cut_stats']['cutfunc']['after']    / proc['cut_stats']['cutfunc']['before']
        
        N             = trg_eA  * cut_eA * xs * L_int # Number of events expected
        B_tot        += N
        B_trg_eA_xs  += trg_eA * xs
        B_cut_eA_xs  += cut_eA * xs
        B_acc_eff_xs += trg_eA * cut_eA * xs
        B_xs_tot     += xs
        
        name = name.replace("_", "-")
        name = (name[:85] + '..') if len(name) > 85 else name # Truncate maximum length, add ..
        
        dprint(f'{name} & {xs:0.1f} & {trg_eA:0.3f} & {cut_eA:0.3f} & {trg_eA*cut_eA:0.3f} & {N:0.1E} \\\\')
    
    B_trg_eA  = B_trg_eA_xs  / B_xs_tot
    B_cut_eA  = B_cut_eA_xs  / B_xs_tot
    B_acc_eff = B_acc_eff_xs / B_xs_tot
    
    dprint(f'\\hline')
    dprint(f'Total & {B_xs_tot:0.1f} & {B_trg_eA:0.3f} & {B_cut_eA:0.3f} & {B_acc_eff:0.3f} & {B_tot:0.1E} \\\\')
    dprint('\\end{tabular}')
    dprint('\\\\')
    dprint('')

    # -----------------
    # Signal estimate per model point

    keys = list(resdict['roc_mstats'].keys())
    num_MVA_models = len(resdict['roc_mstats'][keys[0]])

    for MVA_model_index in range(num_MVA_models):
        
        S        = np.zeros(len(info[f"class_{signal_ID}"].keys()))
        B        = np.zeros(len(S))
        
        S_trg_eA = np.zeros(len(S))       # Trigger eff x acceptance
        S_cut_eA = np.zeros(len(S))       # Basic cuts
        
        BOX_eff  = np.zeros((len(S), 2))  # Category cuts
        MVA_eff  = np.zeros((len(S), 2))  # MVA cut
        thr      = np.zeros(len(S))
        names    = len(S) * [None]

        param_values = []
        i = 0
        
        # Loop over different signal model point processes
        for name in info[f"class_{signal_ID}"].keys():
            
            names[i]  = name.replace('_', '-')
            proc      = info[f"class_{signal_ID}"][name]
            
            # -----------------
            # Pick model parameters
            model_param = proc['yaml']['model_param']
            
            # Find the matching filter strings
            ROC_filter_key, BOX_filter_key, GEN_filter_key, param_point \
                = find_filter(resdict, model_param, args)
            
            param_values.append(param_point)
            
            # -----------------
            # Construct box cut efficiency for background [0], signal [1]

            # For the background, we require only the fiducial box
            mask         = resdict['roc_filters'][BOX_filter_key][MVA_model_index]
            BOX_eff[i,0] = mask_eff(num_mask=mask, den_mask=None,
                                    y_true=resdict['data'].y, class_id=background_ID, weights=resdict['data'].w)
            
            # For the signal, we require the right GEN sample in the denominator and numerator
            den_mask     = resdict['roc_filters'][GEN_filter_key][MVA_model_index]
            num_mask     = np.logical_and(resdict['roc_filters'][GEN_filter_key][MVA_model_index],
                                          resdict['roc_filters'][BOX_filter_key][MVA_model_index])
            
            BOX_eff[i,1] = mask_eff(num_mask=num_mask, den_mask=den_mask,
                                    y_true=resdict['data'].y, class_id=signal_ID, weights=resdict['data'].w)
            
            print(np.round(BOX_eff[i,:], 3))
            
            # -----------------
            # Get the pre-computed MVA-ROC results
            
            roc_obj   = resdict['roc_mstats'][ROC_filter_key][MVA_model_index]
            roc_label = resdict['roc_labels'][ROC_filter_key][MVA_model_index]

            fpr, tpr, thresholds = roc_obj.fpr, roc_obj.tpr, roc_obj.thresholds
            tpr_err   = np.ones(len(tpr))
            fpr_err   = np.ones(len(tpr))
            
            for k in range(len(tpr_err)):
                tpr_err[k] = np.std(roc_obj.tpr_bootstrap[:,k])
                fpr_err[k] = np.std(roc_obj.fpr_bootstrap[:,k])
            
            # -----------------
            # Interpolate ROC-curve
            
            func     = interpolate.interp1d(fpr, tpr,        'linear')
            func_thr = interpolate.interp1d(fpr, thresholds, 'linear')
            
            ## Pick ROC-working point from the interpolation
            s_eff  = func(B_target_eff)
            thr[i] = func_thr(B_target_eff)

            ## Pick ROC-working point from the analytic fit
            # xval = np.logspace(-8, 0, 1000)
            # yhat = func_binormal(B_target_eff, *popt)

            # Save efficiency values
            MVA_eff[i,:] = np.array([B_target_eff, s_eff])
            
            # -----------------
            # Plot ROC fit
            
            plot_ROC_fit(i, fpr, tpr, tpr_err, fpr_err, roc_obj, roc_label, names, args)
            
            # ----------------
            # Compute and collect values
            
            B[i]        = B_tot * BOX_eff[i,0] * MVA_eff[i,0]
            
            S_trg_eA[i] = proc['cut_stats']['filterfunc']['after'] / proc['cut_stats']['filterfunc']['before']
            S_cut_eA[i] = proc['cut_stats']['cutfunc']['after']    / proc['cut_stats']['cutfunc']['before']
            
            # Expected signal event count (without BR applied !)
            S[i] = S_trg_eA[i] * S_cut_eA[i] * BOX_eff[i,1] * MVA_eff[i,1] * signal_xs * L_int
            
            i += 1
            # << == Loop over signal points ends
        
        dprint(f'\\textbf{{ML model: {MVA_model_names[MVA_model_index]} }} \\\\')
        dprint('')
        dprint('\\tiny')
        dprint('\\begin{tabular}{l||cc|cccc|cc|c}')
        dprint('Signal model point & $B$: $\\epsilon$($M$-box) & $B$: $\\epsilon$(ML) (threshold) & $S$: $\\epsilon A$(trg) & $\\epsilon$(pre-cut) & $\\epsilon$($M$-box) & $\\epsilon$(ML) & $\\langle B \\rangle$ & $\\langle S \\rangle$ & $\\langle S \\rangle / \\sqrt{{ \\langle B \\rangle }}$ \\\\')
        dprint('\\hline')
        
        output = {}
        
        # -----------------
        # Plot summary table
        
        for i in range(len(S)):
            
            # Expected discovery significance (Gaussian limit)
            ds   = BR * S[i] / np.sqrt(B[i])
            
            name = names[i]
            name = ('..' + name[40:]) if len(name) > 65 else name # Truncate maximum length, add ..
            
            line = f'{name} & {BOX_eff[i,0]:0.1E} & {MVA_eff[i,0]:0.1E} ({thr[i]:0.3f}) & {S_trg_eA[i]:0.3f} & {S_cut_eA[i]:0.2f} & {BOX_eff[i,1]:0.2f} & {MVA_eff[i,1]:0.2f} & {B[i]:0.1E} & {BR * S[i]:0.1E} & {ds:0.1f} \\\\'
            dprint(line)
            
            output[names[i]] = {'param':      param_values[i],
                                'signal':     {'counts': S[i], 'systematic': 0},
                                'background': {'counts': B[i], 'systematic': 0}}
        
        dprint('\\end{tabular}')
        dprint('')
        
        print(output)
        
        # -----------------
        # Compute upper limits
        
        limits.run_limits_vector(data=output, param=args['limits_param'],
            savepath = f'{args["plotdir"]}/eval/limits/{MVA_model_names[MVA_model_index]}')
        
        # << == Loop over models ends
    
    # -----------------
    # End the document and compile latex
    dprint('\\end{document}')
    
    os.system(f'pdflatex -output-directory {latex_path} {latex_filename}')
    
    print('[Done]')
