# DQCD analysis signal upper limits
#
# Execute with: python icedqcd/limits.py
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
import uproot
import matplotlib.pyplot as plt
import os
from datetime import datetime
from termcolor import colored, cprint
from matplotlib.ticker import FuncFormatter
                    
from scipy.interpolate import RectBivariateSpline
from matplotlib import ticker, cm
import matplotlib as mpl

# icenet
import sys
sys.path.append(".")

from icenet.tools import aux,io
from icefit.peakfit import TH1_to_numpy
from icefit.icelimit import *

brazil_green  = np.array([0, 245, 34]) / 255
brazil_yellow = np.array([245, 245, 56]) / 255


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


def plot_brazil(x, Y, s1_color=brazil_green, s2_color=brazil_yellow,
                horizontal_line=1e-1, ylim=[9e-4, 2e-1]):
    """
    Produce classic 1D "Brazil" plot
    
    Args:
        x:     x-axis values
        Y:     Y-axis values, dim = [number of points] x [-2sigma, -1sigma, median, +1sigma, +2sigma]
    Returns:
        fig, ax
    """
    fig,ax = plt.subplots(figsize=(4.5, 3.5))
    
    plt.fill_between(x, Y[:,0], Y[:,4], color=s2_color, alpha=1.0, label='$\\pm 2\\sigma$')
    plt.fill_between(x, Y[:,1], Y[:,3], color=s1_color, alpha=0.6, label='$\\pm 1\\sigma$')
    plt.plot(x, Y[:,2], color='black', linestyle='--', label='Expected')
    plt.plot(x, horizontal_line * np.ones(len(x)), color='gray', linestyle=':') # Horizontal line

    plt.legend(fontsize=9, frameon=False)
    plt.yscale('log')
    plt.ylim(ylim)
    #plt.rcParams['axes.xmargin'] = 0
    
    return fig, ax


def find_limits(data, param_fixed, value, param_running, method):
    
    running_param_values = []
    limits_expected      = []
    limits_observed      = []
    
    for key in data.keys():
        if data[key]['model_param'][param_fixed] == value:
            running_param_values.append(data[key]['model_param'][param_running])
            limits_expected.append(data[key]['limits_expected'][method])
            limits_observed.append(data[key]['limits_observed'][method])
    
    running_param_values = np.array(running_param_values)
    limits_expected      = np.array(limits_expected)
    limits_observed      = np.array(limits_observed)

    # Sort according to parameter values
    idx = np.argsort(running_param_values)
    
    return running_param_values[idx], limits_expected[idx], limits_observed[idx]


def create_limit_plots_vector(data, param, savepath='.'):

    ylabel = fr"CL95 upper limit on {param['cosmetics']['ylabel']} (CLs)" # note raw fr
    title  = fr"{param['cosmetics']['title']}"
    portal = param['cosmetics']['portal']
    
    print(ylabel)
    
    # -----------------------------------------------------
    
    # Find all parameter tuplets
    params = list(data[list(data.keys())[0]]['model_param'].keys())
    values = dict(zip(params, [None]*len(params)))
    
    for key in data.keys():
        for p in params:
            if values[p] is None: values[p] = set()
            values[p].add(data[key]['model_param'][p])

    for p in params:
        values[p] = np.sort(np.array(list(values[p])))
    
    # -----------------------------------------------------
    
    for method in param['methods']:
        
        plot_index = 0
        
        # -------------------------------------------------
        # 1D Brazil per fixed m point with running ctau
        
        pp            = param['brazil']['ctau_fixed_m']
        param_fixed   = pp['model_POI']['fixed']
        param_running = pp['model_POI']['running']
        
        fixed_values  = values[param_fixed]
        
        for _, value in enumerate(fixed_values):
            
            running_values, expected, observed = find_limits(data=data, param_fixed=param_fixed, value=value,
                                                          param_running=param_running, method=method)

            ## Collect values
            Y = np.zeros((len(running_values), 5)) # 5 from -2,-1,0,+1,+2 sigmas
            for i in range(len(expected)):
                Y[i,:] = expected[i]
            print(Y)
            
            ## Brazil
            fig,ax = plot_brazil(x=running_values, Y=expected, ylim=pp['ylim'])        
            plt.xlim([np.min(running_values)*0.9, np.max(running_values)*1.1])

            ## Tick formatter
            ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator([0, 1, 3, 5, 8, 10, 30, 50, 80, 100, 300, 500]))
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
            ax.xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
            
            for tick in ax.xaxis.get_major_ticks():
                tick.set_pad(7) # Add padding

            ax.tick_params(axis='x', which='minor', labelsize=5)
            
            ## Cosmetics
            plt.title(f'$m_0 = {value}$ GeV, ${portal}$ $portal$ | {title}', fontsize=10)
            plt.xlabel(pp['xlabel'])
            plt.ylabel(ylabel, fontsize=7)
            plt.xscale('log')
            plt.yticks(fontsize=10)
            plt.xticks(fontsize=10)
            
            ## Save
            outputdir = aux.makedir(f'{savepath}/{method}') 
            plt.savefig(f'{outputdir}/plot_{plot_index:0=4}_1D_{param_fixed}_{value}_method_{method}.pdf', bbox_inches='tight')
            plot_index +=1
            plt.close()

        # -------------------------------------------------
        ## 1D Brazil per fixed ctau point with running m
        
        pp            = param['brazil']['m_fixed_ctau']
        param_fixed   = pp['model_POI']['fixed']
        param_running = pp['model_POI']['running']
        
        fixed_values  = values[param_fixed]
        
        for _, value in enumerate(fixed_values):
            
            running_values, expected, observed = find_limits(data=data, param_fixed=param_fixed, value=value,
                                                          param_running=param_running, method=method)
            
            ## Collect values
            Y = np.zeros((len(running_values), 5)) # 5 from -2,-1,0,+1,+2 sigmas
            for i in range(len(expected)):
                Y[i,:] = expected[i]
            print(Y)
            
            ## Brazil
            fig,ax = plot_brazil(x=running_values, Y=Y, ylim=pp['ylim'])        
            plt.xlim([np.min(running_values)*0.9, np.max(running_values)*1.02])
            
            ## Tick formatter
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3g'))
            
            ## Cosmetics
            plt.title(f'$c\\tau_0 = {value}$ mm, ${portal}$ $portal$ | {title}', fontsize=10)
            plt.xlabel(pp['xlabel'])
            plt.ylabel(ylabel, fontsize=7)
            plt.xscale('linear')
            plt.yticks(fontsize=10)
            plt.xticks(fontsize=10)

            ## Save
            outputdir = aux.makedir(f'{savepath}/{method}')
            plt.savefig(f'{outputdir}/plot_{plot_index:0=4}_1D_{param_fixed}_{value}_method_{method}.pdf', bbox_inches='tight')
            plot_index += 1
            plt.close()
        
        # -------------------------------------------------
        ## 2D-contours over (mass, ctau)
        
        special_value = 0.01
        
        fig,ax        = plt.subplots(figsize=(4.5, 3.5))
        
        labels        = [f'$\\pm2 \\sigma$ ($r = {special_value}$)', f'$\\pm1 \\sigma$ ($r = {special_value}$)', 'Median']
        linestyles    = ['dotted', 'dashed', 'solid', 'dashed', 'dotted']
        
        pp            = param['brazil']['m_ctau_contour']
        kx,ky         = pp['kx'], pp['ky']
        
        running_param = pp['model_POI']['running']
        m_values      = values[running_param[0]]
        ctau_values   = values[running_param[1]]
        
        ylim = param['brazil']['m_ctau_contour']['ylim']
        
        for ind in range(5):
            
            # Read limit values
            R = np.zeros((len(m_values), len(ctau_values)))
            for i in range(len(m_values)):
                for j in range(len(ctau_values)):
                    
                    for key in data.keys():
                        if data[key]['model_param']['m'] == m_values[i] and \
                           data[key]['model_param']['ctau'] == ctau_values[j]:
                            R[i,j] = data[key]['limits_expected'][method][ind]
            
            ## Interpolate
            interp_spline    = RectBivariateSpline(m_values, ctau_values, R, kx=kx, ky=ky)

            m_values_fine    = np.linspace(np.min(m_values),    np.max(m_values),    1000)
            ctau_values_fine = np.linspace(np.min(ctau_values), np.max(ctau_values), 10000)
            R_fine           = interp_spline(m_values_fine, ctau_values_fine)
            
            # Post-enforce positivity (splines can go negative)
            R_fine[R_fine < 0] = 0
            
            # Transpose for the plot
            R_fine = R_fine.T

            try:
                if ind != 2: # ind == 2 is the median value
                    
                    # Draw uncertainties at specific chosen special contour
                    cp = ax.contour(m_values_fine, ctau_values_fine, R_fine, [special_value],
                                    alpha=0.4, cmap=cm.bone, linewidths=1.0, linestyles=linestyles[ind])
                else:
                    
                    levels   = np.logspace(np.log10(np.max([ylim[0], np.min(R_fine)])), np.log10(ylim[1]), 8)
                    #R[R > 1] = None # Null (draw white) out the domain > 1
                    
                    cp = ax.contourf(m_values_fine, ctau_values_fine, R_fine, cmap=cm.copper,
                        levels=levels, locator=ticker.LogLocator())
                    
                    fmt  = lambda x, pos: '{:0.2E}'.format(x) # Tick precision formatter
                    cbar = plt.colorbar(cp, format=FuncFormatter(fmt))
                    cbar.set_label(ylabel, rotation=270, labelpad=15, fontsize=7)
                    cbar.ax.tick_params(labelsize=8)
            except:
                print(__name__ + f'.create_limit_plots: Problem with the contour plot -- skipping (check the input)')
                print(R_fine)
        
        ## Tick formatter
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator([2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]))
        ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3g'))

        #ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator(ctau_values))
        ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600]))
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
        ax.yaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
        #ax.tick_params(axis='y', which='major', labelsize=10)
        ax.tick_params(axis='y', which='minor', labelsize=5)
        #ax.text(12, 300, '$-1\\sigma$', color='black', fontsize=6)
        #ax.text(12, 380, '$-2\\sigma$', color='black', fontsize=6)
        
        for tick in ax.yaxis.get_major_ticks():
            tick.set_pad(7) # Add padding

        plt.yticks(fontsize=10)
        plt.xticks(fontsize=10)

        ## Add legend boxes
        null = np.array([np.NaN, np.NaN])
        for i in range(2):
            plt.plot(null, null, color='gray', linestyle=linestyles[i], label=labels[i])
        plt.legend(frameon=False, loc='upper right', fontsize=7)

        ## Cosmetics
        plt.title(f'${portal}$ $portal$ | {title}', fontsize=10)
        ax.set_xlabel(pp['xlabel'])
        ax.set_ylabel(pp['ylabel'])
        ax.set_yscale('log')

        ## Save figure
        outputdir = aux.makedir(f'{savepath}/{method}')
        plt.savefig(f'{outputdir}/plot_{plot_index:0=4}_2D_m_ctau_method_{method}.pdf', bbox_inches='tight')
        plt.close()

def create_limit_tables(data, param, savepath):

    ylabel = fr"CL95 upper limit (CLs) on {param['cosmetics']['ylabel']}" # note raw fr
    
    for method in param['methods']:
        
        latex_path     = aux.makedir(f'{savepath}/{method}')
        latex_filename = f'{latex_path}/limits_table_method_{method}.tex'
        
        def dprint(string, mode='a', end='\n'):
            print(string)
            with open(latex_filename, mode) as f:
                f.write(f'{string} {end}')

        dprint('', 'w')
        dprint(latex_header)    

        dprint('\\Large')
        dprint(f'Signal Upper Limits')
        dprint('\\small')
        dprint('\\\\')
        dprint(f'github.com/mieskolainen/icenet')
        dprint('\\\\')
        dprint(f"{datetime.now():%d-%m-%Y %H:%M} ")
        dprint('\\\\')
        dprint('\\vspace{0.5em}')
        dprint('\\\\')
        dprint(f'ML model: {data[list(data.keys())[0]]["ML_model"]}')
        dprint('\\\\')
        
        dprint(f'Signal portal: {param["cosmetics"]["portal"]}')
        dprint('\\\\')
        dprint(f'Limit method: {method}')
        dprint('\\\\')
        dprint(f'Systematics: {param["systematics"]}')
        dprint('\\\\')
        dprint('\\vspace{5em}')
        dprint('\\\\')

        dprint(f'Expected {ylabel}')
        dprint('\\\\')
        dprint('$H_0$: background only scenario')
        dprint('\\vspace{1em}')
        dprint('\\\\')

        dprint('\\begin{tabular}{|l||cc|c|cc|}')
        dprint('\\hline')
        dprint('Signal model point & $-2\\sigma$ & $-1\\sigma$ & median $r_{up}$ & $+1\\sigma$ & $+2\\sigma$ \\\\')
        dprint('\\hline')

        for key in data.keys():
            
            dprint(f'{data[key]["model_param"]}', end='')
            
            for i in range(len(data[key]["limits_expected"][method])):
                dprint(f' & {data[key]["limits_expected"][method][i]:.2g} ', end='')
            dprint('\\\\')
        dprint('\\hline')
        
        dprint('\\end{tabular}')
        dprint('\\\\')
        dprint('\\vspace{5em}')
        dprint('\\\\')

        dprint('\\newpage')
        
        dprint('\\small')
        dprint(f'Emulated (observed) {ylabel}')
        dprint('\\\\')
        dprint(f'$H_1$: r * signal + background benchmark scenario, with $r={param["target_r"]}$')
        dprint('\\vspace{1em}')
        dprint('\\\\')
        
        dprint('\\begin{tabular}{|l||c|cc|}')
        dprint('\\hline')
        dprint('Signal model point & Observed $r_{up}$ & $\\langle B \\rangle$ (counts) & $\\langle S \\rangle$ (counts) \\\\')
        dprint('\\hline')
        
        for key in data.keys():
            
            dprint(f'{data[key]["model_param"]}', end='')
            
            for i in range(len(data[key]["limits_observed"][method])):
                
                B = data[key]['limits_statistics']['background']
                S = param["target_r"] * data[key]['limits_statistics']['signal']
                
                B = f'{float(f"{B:.2g}"):g}'
                S = f'{float(f"{S:.2g}"):g}'

                dprint(f' & {data[key]["limits_observed"][method][i]:.2g} & {B} & {S} ', end='')
            dprint('\\\\')
        dprint('\\hline')
        dprint('\\end{tabular}')
        dprint('')
    
    dprint('\\end{document}')

    # Compile latex
    os.system(f'pdflatex -output-directory {latex_path} {latex_filename}')


def limit_wrapper_dict(data, param, bg_regulator=0.0, s_regulator=0.0):
    
    for file in data.keys():
        
        # Counts (assumed Poisson)
        s_hypothesis = data[file]['signal']['counts']
        bg_expected  = data[file]['background']['counts']
        
        # Regulate MC
        if  bg_expected <= 0:
            bg_expected = bg_regulator
        if s_hypothesis <= 0:
            s_hypothesis = s_regulator
        
        # Total systematics (MC uncertainties + others would go here)
        if param['systematics'] is not None:
            s_syst_error  = data[file]['signal']['systematic']
            bg_syst_error = data[file]['background']['systematic']
        else:
            s_syst_error  = 0
            bg_syst_error = 0

        print(f'\n{file}: background: {np.round(bg_expected,5)}, signal = {np.round(s_hypothesis,5)}')
        
        limits_expected   = {'asymptotic': None, 'toys': None}
        limits_observed   = {'asymptotic': None, 'toys': None}
        limits_statistics = {'background': bg_expected, 'signal': s_hypothesis}
        
        for method in param['methods']:
            
            print(f'[method: {method}]')

            # Excepted limits
            opt = CL_single_compute(method        = method,
                                    observed      = None,
                                    s_hypothesis  = s_hypothesis,
                                    bg_expected   = bg_expected,
                                    s_syst_error  = s_syst_error,
                                    bg_syst_error = bg_syst_error,
                                    num_toys_obs  = param['num_toys_obs'])

            print(np.round(opt,4))
            limits_expected[method] = opt
            
            # Emulate observed with benchmark r
            r   = param['target_r']
            opt = CL_single_compute(method        = method,
                                    observed      = r * s_hypothesis + bg_expected,
                                    s_hypothesis  = s_hypothesis,
                                    bg_expected   = bg_expected,
                                    s_syst_error  = s_syst_error,
                                    bg_syst_error = bg_syst_error)
            
            print(np.round(opt,4))
            limits_observed[method] = opt
        
        data[file]['limits_expected']   = limits_expected
        data[file]['limits_observed']   = limits_observed
        data[file]['limits_statistics'] = limits_statistics
    
    return data


def run_limits_vector(data={}, param={}, savepath = '.'):
    """
    DQCD fast limits
    """
    
    ## Compute limits over all signal (parameter) points
    data = limit_wrapper_dict(data=data, param=param)
    
    ## Create plots
    if param['cosmetics']['portal'] == 'vector':
        create_limit_plots_vector(data=data, param=param, savepath=savepath)
    else:
        cprint(f"Plots not implemented for portal {param['cosmetics']['portal']}", 'red')
    
    ## Produce tables
    create_limit_tables(data=data, param=param, savepath=savepath)
    
    ## Combine pdfs
    for method in param['methods']:
        figpath = aux.makedir(f'{savepath}/{method}')
        os.system(f'gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -sOutputFile={figpath}/limits_doc_{method}.pdf {figpath}/*.pdf')


if __name__ == "__main__":
    True
