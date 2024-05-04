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
import copy

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.stats import ncx2,norm

from icenet.tools import aux, io
from icefit import cortools, statstools
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

def plot_ROC_fit(i, fpr, tpr, tpr_err, fpr_err, roc_obj, roc_label, names, args, savename):
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
    fig,ax = plt.subplots(1,2, figsize=(8,4))
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

        path = aux.makedir(f'{args["plotdir"]}/eval/optimize/ROC_fit/{roc_label}')
        
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

    pdf_filename = f'{path}/{savename}.pdf'
    plt.savefig(pdf_filename, bbox_inches='tight')
    plt.close()


def find_filter(rd, model_param, args):
    """
    Find filter strings
    """
    model_POI      = args['limits_param']['model_POI']
    box_POI        = args['limits_param']['box_POI'] 
    
    GEN_filter_tag = args['limits_param']['GEN_filter_tag']
    BOX_filter_tag = args['limits_param']['BOX_filter_tag']
    GBX_filter_tag = args['limits_param']['GBX_filter_tag']
    
    GEN_filter_key = None
    BOX_filter_key = None
    GBX_filter_key = None
    param_point    = {}
    
    cprint(__name__ + f'.find_filter: Signal: {model_param}', 'yellow')

    # --------------
    
    for filter in rd['roc_mstats'].keys():
        matches = 0
        for param_name in model_POI:
            
            if f'{param_name} = {model_param[param_name]:g}' in filter and GEN_filter_tag in filter:
                matches += 1
                param_point[param_name] = model_param[param_name]
        
        if matches == len(model_POI): # all parameters match
            GEN_filter_key = filter
            cprint(f'Found GEN filter: {GEN_filter_key}' , 'green')
            break
    
    if GEN_filter_key is None:
        cprint(f'ERROR: No matching GEN filter for the signal: {model_param}, exit', 'red')
        exit()
        
    # --------------
    
    for filter in rd['roc_mstats'].keys():
        
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
    
    for filter in rd['roc_mstats'].keys():
        matches = 0
        for param_name in model_POI :
            
            if f'{param_name} = {model_param[param_name]:g}' in filter and GBX_filter_tag in filter:
                matches += 1
        
        if matches == len(model_POI): # all parameters match
            GBX_filter_key = filter
            
            cprint(f'Found GBX filter: {GBX_filter_key}' , 'green')
            break
    
    if GBX_filter_key is None:
        cprint(f'ERROR: No matching GBX filter for the signal: {model_param}, exit', 'red')
        exit()
    
    # Sort alphabetically by parameter name
    param_point = dict(sorted(param_point.items()))
    
    return GEN_filter_key, BOX_filter_key, GBX_filter_key, param_point


def optimize_selection(args):
    """
    Main program
    """
    
    def dprint(string, mode='a'):
        print(string)
        with open(latex_filename, mode) as f:
            f.write(string + '\n')
    
    latex_path     = aux.makedir(f'{args["plotdir"]}/eval/optimize')
    latex_filename = f'{latex_path}/optimize.tex'
    
    # -----------------
    ## Main parameters

    # Integrated luminosity
    L_int          = args['limits_param']['luminosity'] # in pb^{-1}
    
    # Signal production cross section
    signal_xs      = args['limits_param']['signal_xs']  # in pb
    
    # Benchmark BR
    BR             = args['limits_param']['target_r']
    
    # Targets
    const_mode     = args['limits_param']['const_mode']
    const_cut      = args['limits_param']['const_cut']
    const_bgk_eff  = args['limits_param']['const_bgk_eff']
    
    # BOX emulation
    commutate_BOX  = args['limits_param']['commutate_BOX']
    
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
    dprint(f'$\\langle S \\rangle \\equiv L \\times (\\sigma_S \\times \\text{{BR}}) \\times \\epsilon A_S (\\text{{trg}}) \\times \\epsilon_S (\\text{{pre-cut}}) \\times \\epsilon_S (\\text{{ML}}) \\times \\epsilon_S (M\\text{{-box}})$ (cut ordered) \\\\')
    dprint('\\vspace{1em}')
    dprint('\\\\')
    
    if   const_mode == 'cut':
        dprint(f'Const mode: ML score cut threshold fixed at $c > {const_cut:0.2f}$ \\\\')
    elif const_mode == 'bgk_eff':
        dprint(f'Const mode: Bgk-efficiency target fixed at {const_bgk_eff:0.1E} (after trg + pre-selection) \\\\')
    else:
        raise Exception(__name__ + f".optimize_selection: Unknown const mode '{const_mode}'")
    
    dprint(f'commutate BOX cut: {commutate_BOX} (low MC stats treatment) \\\\')
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
    cprint(__name__ + f'.optimize_selection: Loading "{targetfile}" ...', 'yellow')
    
    with open(targetfile, 'rb') as file:
        rd = pickle.load(file)
    info = rd['info']

    # MVA-model labels
    ML_model_names = rd['roc_labels']['inclusive']

    # background, signal
    class_ids = args['primary_classes']

    # -----------------
    # Background estimate
    
    B_tot        = 0
    B_xs_tot     = 0
    B_trg_eA_xs  = 0
    B_cut_eA_xs  = 0
    B_acc_eff_xs = 0

    # Loop over background sub-processes
    for name in info[f"class_{class_ids[0]}"].keys():
        
        proc          = info[f"class_{class_ids[0]}"][name]
        
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
        name = (name[:65] + '..') if len(name) > 65 else name # Truncate maximum length, add ..
        
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

    keys = list(rd['roc_mstats'].keys())
    num_ML_models = len(rd['roc_mstats'][keys[0]])
    
    for ML_idx in range(num_ML_models):
        
        S           = np.zeros(len(info[f"class_{class_ids[1]}"].keys()))
        sigma_S     = np.zeros(len(S))
        B           = np.zeros(len(S))
        sigma_B     = np.zeros(len(S))
        ADS         = np.zeros(len(S))
        ds          = np.zeros(len(S))
        
        S_trg_eA    = np.zeros(len(S))          # Trigger eff x acceptance
        S_cut_eA    = np.zeros(len(S))          # Basic cuts
        
        ML_eff         = np.zeros((len(S), 2))  # ML cut
        ML_eff_err     = np.zeros((len(S), 2))  # ML cut
        
        BOX_eff        = np.zeros((len(S), 2))  # Box cut
        BOX_eff_err    = np.zeros((len(S), 2))  # Box cut
        
        ML_BOX_eff     = np.zeros((len(S), 2))  # ML x Box cut
        ML_BOX_eff_err = np.zeros((len(S), 2))  # ML x Box cut
        
        thr         = np.zeros(len(S))
        names       = len(S) * [None]
        
        param_values = []
        output       = {}
        i = 0
        
        # Loop over different signal model point processes
        for name in info[f"class_{class_ids[1]}"].keys():
            
            names[i]  = name.replace('_', '-')
            proc      = info[f"class_{class_ids[1]}"][name]
            
            # -----------------
            # Pick model parameters
            model_param = proc['yaml']['model_param']
            
            # Find the matching filter strings
            GEN_filter_key, BOX_filter_key, GBX_filter_key, param_point \
                = find_filter(rd, model_param, args)
            
            param_values.append(param_point)
            
            # -----------------
            # Get the pre-computed MVA-ROC results after combined GEN & BOX cut
            # (background uses "diplomat_class" filter under plots)
            
            if commutate_BOX:           # Low MC stats treatment
                fKEY = GEN_filter_key # Gen
            else:
                fKEY = GBX_filter_key # Gen & Box
            
            roc_obj   = rd['roc_mstats'][fKEY][ML_idx]
            roc_label = rd['roc_labels'][fKEY][ML_idx]
            roc_path  = rd['roc_paths'][fKEY][ML_idx]
            
            fpr, tpr, thresholds = roc_obj.fpr, roc_obj.tpr, roc_obj.thresholds
            fpr_err, tpr_err     = 0,0
            
            plot_ROC_fit(i=i, fpr=fpr, tpr=tpr, tpr_err=tpr_err, fpr_err=fpr_err,
                roc_obj=roc_obj, roc_label=roc_label, names=names, args=args, savename=roc_path)
            
            # -----------------
            # Construct box cut efficiency prior ML cut
            
            BOX_cut_mask = rd['roc_filters'][BOX_filter_key][ML_idx] & rd['roc_filters'][GEN_filter_key][ML_idx]
            den_mask     = copy.deepcopy(rd['roc_filters'][GEN_filter_key][ML_idx]) #! deepcopy
            
            for C in range(2):

                BOX_eff[i,C], BOX_eff_err[i,C] = statstools.columnar_mask_efficiency(
                    num_mask=BOX_cut_mask, den_mask=den_mask,
                    y_true=rd['data'].y, y_target=class_ids[C], weights=rd['data'].w)
            
            # -----------------
            # Construct ML cut efficiency such that the combined BOX & ML efficiency
            # matches the FPR level we want for the background (or use a constant cut)
            
            if   const_mode == 'cut':
                thr[i] = const_cut
            elif const_mode == 'bgk_eff':
                func   = interpolate.interp1d(fpr * BOX_eff[i,0], thresholds, 'linear')
                thr[i] = func(const_bgk_eff)
            else:
                raise Exception(__name__ + f'.optimize_selection: Unknown const_mode "{const_mode}"')
            
            # -----------------
            # ML cut
            
            ML_cut_mask = (rd['y_preds'][ML_idx] > thr[i]) & rd['roc_filters'][GEN_filter_key][ML_idx]
            den_mask    = copy.deepcopy(rd['roc_filters'][GEN_filter_key][ML_idx]) #! deepcopy
            
            for C in range(2):
                
                ML_eff[i,C], ML_eff_err[i,C] = statstools.columnar_mask_efficiency(num_mask=ML_cut_mask, den_mask=den_mask,
                    y_true=rd['data'].y, y_target=class_ids[C], weights=rd['data'].w)
            
            # -----------------
            # BOX cut

            BOX_cut_mask = ML_cut_mask & rd['roc_filters'][BOX_filter_key][ML_idx]
            den_mask     = copy.deepcopy(ML_cut_mask) #! deepcopy
            
            for C in range(2):
                
                if commutate_BOX and C == 0:
                    continue # Low MC stats treatment, computed above
                
                BOX_eff[i,C], BOX_eff_err[i,C] = statstools.columnar_mask_efficiency(
                    num_mask=BOX_cut_mask, den_mask=den_mask,
                    y_true=rd['data'].y, y_target=class_ids[C], weights=rd['data'].w)
            
            # ----------------
            # Total ML x BOX cut
            
            ML_BOX_cut_mask = ML_cut_mask & BOX_cut_mask
            den_mask        = copy.deepcopy(rd['roc_filters'][GEN_filter_key][ML_idx]) #! deepcopy
            
            for C in range(2):
                
                if commutate_BOX and C == 0: # Low MC stats treatment
                    ML_BOX_eff[i,C]     = ML_eff[i,C] * BOX_eff[i,C]
                    ML_BOX_eff_err[i,C] = statstools.prod_eprop(A=ML_eff[i,C], B=BOX_eff[i,C],
                        sigmaA=ML_eff_err[i,C], sigmaB=BOX_eff_err[i,C], sigmaAB=0)
                    continue
                
                ML_BOX_eff[i,C], ML_BOX_eff_err[i,C] = statstools.columnar_mask_efficiency(
                        num_mask=ML_BOX_cut_mask, den_mask=den_mask,
                        y_true=rd['data'].y, y_target=class_ids[C], weights=rd['data'].w)
            
            # ----------------
            # Compute values
            
            B[i]        = B_tot * ML_BOX_eff[i,0]
            sigma_B[i]  = B_tot * ML_BOX_eff_err[i,0]
            
            S_trg_eA[i] = proc['cut_stats']['filterfunc']['after'] / proc['cut_stats']['filterfunc']['before']
            S_cut_eA[i] = proc['cut_stats']['cutfunc']['after']    / proc['cut_stats']['cutfunc']['before']
            
            # Expected signal event count (without BR applied here!)
            S_tot       = (signal_xs * L_int) * S_trg_eA[i] * S_cut_eA[i]
            S[i]        = S_tot * ML_BOX_eff[i,1]
            sigma_S[i]  = S_tot * ML_BOX_eff_err[i,1]
            
            # Asimov Discovery Significance
            ADS[i] = statstools.ADS(s=BR * S[i], b=B[i], sigma_b=sigma_B[i])

            # Naive
            ds[i]  = BR * S[i] / np.sqrt(B[i])

            output[names[i]] = {'ML_model':     ML_model_names[ML_idx],
                                'model_param':  param_values[i],
                                'signal':       {'counts': S[i], 'systematic': sigma_S[i]},
                                'background':   {'counts': B[i], 'systematic': sigma_B[i]}}
            
            # ----------------
            # Diagnostics
            print(f'ML_eff:  {np.round(ML_eff[i,:], 6)}  | ML_eff_err  = {np.round(100 * ML_eff_err[i,:]  / ML_eff[i,:], 1)} %')
            print(f'BOX_eff: {np.round(BOX_eff[i,:], 6)} | BOX_eff_err = {np.round(100 * BOX_eff_err[i,:] / BOX_eff[i,:], 1)} %')
            print(f'tot_eff: {np.round(ML_eff[i,:] * BOX_eff[i,:], 6)} | tot_eff_err = {np.round(100 * ML_BOX_eff_err[i,:] / (ML_BOX_eff[i,:]), 1)} %')
            
            i += 1
            
            # << == Loop over signal points ends
        
        dprint(f'\\textbf{{ML model: {ML_model_names[ML_idx]} }} \\\\')
        dprint('')
        dprint('\\tiny')
        dprint('\\begin{tabular}{l|c|cc|cccc|cc|cc}')
        dprint('Signal model point & ML-cut & $B$: $\\epsilon$(ML) & $\\epsilon$($M$-box) & $S$: $\\epsilon A$(trg) & $\\epsilon$(pre-cut) & $\\epsilon$(ML) & $\\epsilon$($M$-box) & $\\langle B \\rangle \\pm$ [\\%] & $\\langle S \\rangle \\pm$ [\\%] & ADS & $S / \\sqrt{B}$ \\\\')
        dprint('\\hline')
        
        # -----------------
        # Plot summary table
        for i in range(len(S)):
            
            line = f'{param_values[i]} & {thr[i]:0.3f} &{ML_eff[i,0]:0.1E} & {BOX_eff[i,0]:0.1E} & {S_trg_eA[i]:0.2g} & {S_cut_eA[i]:0.2g} & {ML_eff[i,1]:0.2g} & {BOX_eff[i,1]:0.2g} & {B[i]:0.1E} $\\pm$ {100*sigma_B[i]/B[i]:0.1f} & {BR*S[i]:0.1E} $\\pm$ {100*sigma_S[i]/S[i]:0.1f} & {ADS[i]:0.2g} & {ds[i]:0.2g} \\\\'
            dprint(line)
        
        dprint('\\end{tabular}')
        dprint('')
        
        # -----------------
        # Compute upper limits
        
        limits.run_limits_vector(data=output, param=args['limits_param'],
            savepath = f'{args["plotdir"]}/eval/limits/{ML_model_names[ML_idx]}')
        
        # << == Loop over models ends
    
    # -----------------
    # End the document and compile latex
    dprint('\\end{document}')
    
    os.system(f'pdflatex -output-directory {latex_path} {latex_filename}')
    
    print('[Done]')
