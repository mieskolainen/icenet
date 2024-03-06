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

from scipy.interpolate import RectBivariateSpline
from matplotlib import ticker, cm
import matplotlib as mpl

# icenet
import sys
sys.path.append(".")

from icenet.tools import aux,io
from icefit import lognormal
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


def plot_brazil(x, Y, s1_color=brazil_green, s2_color=brazil_yellow):
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
    plt.plot(x, np.ones(len(x)), color='gray', linestyle=':') # Horizontal line at 1

    plt.legend(fontsize=9, frameon=False)
    plt.yscale('log')
    
    return fig,ax


def create_limit_plots(expected_limits, observed_limits, m_values, ctau_values, methods,
    portal, experiment_label='LHC', process_label='X', contour_min_r = 1E-9, kx=2, ky=2):
    
    ylabel  = f'CL95 upper limit on {process_label} (CLs)'
    
    for method in methods:

        # -------------------------------------------------
        # 1D Brazil over mass
        plot_index = 0

        for k,m in enumerate(m_values):
            limits = [expected_limits[create_filename(m=m, ctau=ctau, portal=portal)] for ctau in ctau_values]

            # Collect values
            Y = np.zeros((len(ctau_values), 5)) # 5 from -2,-1,0,+1,+2 sigmas
            for i in range(len(limits)):
                Y[i,:] = limits[i][method]
            print(Y)
            
            # Brazil
            fig,ax = plot_brazil(x=ctau_values, Y=Y)        
            plt.xlim([np.min(ctau_values) - 1, np.max(ctau_values) + 1])


            plt.title(f'$m_0 = {m}$ GeV, ${portal}$ $portal$ | {experiment_label}', fontsize=10)
            plt.xlabel('$c\\tau_0$ (mm)')
            plt.ylabel(ylabel)
            plt.xscale('log')

            ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]))
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
            ax.xaxis.set_minor_formatter(mpl.ticker.ScalarFormatter())
            
            for tick in ax.xaxis.get_major_ticks():
                tick.set_pad(7) # Add padding

            ax.tick_params(axis='x', which='minor', labelsize=5)
            plt.yticks(fontsize=10); plt.xticks(fontsize=10)

            outputdir = aux.makedir(f'figs/icelimit/{portal}/{method}') 
            plt.savefig(f'{outputdir}/plot_{plot_index}_1D_m_{m}_method_{method}.pdf', bbox_inches='tight')
            plot_index +=1

        # -------------------------------------------------
        ## 1D Brazil over ctau

        for k,ctau in enumerate(ctau_values):
            limits = [expected_limits[create_filename(m=m, ctau=ctau, portal=portal)] for m in m_values]

            # Collect values
            Y = np.zeros((len(m_values), 5)) # 5 from -2,-1,0,+1,+2 sigmas
            for i in range(len(limits)):
                Y[i,:] = limits[i][method]
            print(Y)
            
            # Brazil
            fig,ax = plot_brazil(x=m_values, Y=Y)        
            plt.xlim([np.min(m_values)*0.9, np.max(m_values)*1.02])

            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.3g'))

            plt.title(f'$c\\tau_0 = {ctau}$ mm, ${portal}$ $portal$ | {experiment_label}', fontsize=10)
            plt.xlabel('$m_0$ (GeV)')
            plt.ylabel(ylabel)

            plt.yticks(fontsize=10); plt.xticks(fontsize=10)

            outputdir = aux.makedir(f'figs/icelimit/{portal}/{method}')
            plt.savefig(f'{outputdir}/plot_{plot_index}_1D_ctau_{ctau}_method_{method}.pdf', bbox_inches='tight')
            plot_index += 1

        # -------------------------------------------------
        ## 2D-contours over (mass, ctau)

        fig,ax = plt.subplots(figsize=(4.5, 3.5))

        labels     = ['$\\pm2 \\sigma$ ($r_{up} = 1$)', '$\\pm1 \\sigma$ ($r_{up} = 1$)', 'Median']
        linestyles = ['dotted', 'dashed', 'solid', 'dashed', 'dotted']

        for ind in range(5):
            
            # Read limit values
            R = np.zeros((len(m_values), len(ctau_values)))
            for i in range(len(m_values)):
                for j in range(len(ctau_values)):
                    limits = expected_limits[create_filename(m=m_values[i], ctau=ctau_values[j], portal=portal)]
                    R[i,j] = limits[method][ind]

            ## Interpolate
            interp_spline    = RectBivariateSpline(m_values, ctau_values, R, kx=kx, ky=ky)

            m_values_fine    = np.linspace(np.min(m_values), np.max(m_values), 1000)
            ctau_values_fine = np.linspace(np.min(ctau_values), np.max(ctau_values), 10000)
            R_fine           = interp_spline(m_values_fine, ctau_values_fine)

            # Post-enforce positivity (splines can go negative)
            R_fine[R_fine < 0] = 0

            # Transpose for the plot
            R_fine = R_fine.T

            try:
                # Plot
                if ind != 2:
                    cp = ax.contour(m_values_fine, ctau_values_fine, R_fine, [1.0], alpha=0.4, cmap=cm.bone,
                        linewidths=1.0, linestyles=linestyles[ind])
                else:
                    levels   = np.logspace(np.log10(np.max([contour_min_r, np.min(R_fine)])), np.log10(1.0), 7)
                    R[R > 1] = None # Null (draw white) out the domain > 1

                    cp     = ax.contourf(m_values_fine, ctau_values_fine, R_fine, cmap=cm.copper,
                        levels=levels, locator=ticker.LogLocator())

                    from matplotlib.ticker import FuncFormatter
                    fmt  = lambda x, pos: '{:0.2f}'.format(x) # Tick precision formatter
                    cbar = plt.colorbar(cp, format=FuncFormatter(fmt))
                    cbar.set_label(f'Median expected CL95 upper limit on {process_label} (CLs)', rotation=270, labelpad=15, fontsize=7)
                    cbar.ax.tick_params(labelsize=8)
            except:
                print(__name__ + f'.create_limit_plots: Problem with the contour plot -- skipping (check the input)')
                print(R_fine)

        plt.title(f'${portal}$ $portal$ | {experiment_label}', fontsize=10)
        ax.set_xlabel('$m_0$ (GeV)')
        ax.set_ylabel('$c\\tau_0$ (mm)')

        ax.set_yscale('log')
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

        plt.yticks(fontsize=10); plt.xticks(fontsize=10)

        # Add legend boxes
        null = np.array([np.NaN, np.NaN])
        for i in range(2):
            plt.plot(null, null, color='gray', linestyle=linestyles[i], label=labels[i])
        plt.legend(frameon=False, loc='upper left', fontsize=7)

        # Save figure
        outputdir = aux.makedir(f'figs/icelimit/{portal}/{method}')
        plt.savefig(f'{outputdir}/plot_{plot_index}_2D_m_ctau_method_{method}.pdf', bbox_inches='tight')

def create_filename(m,ctau,portal):
    return f'{portal}_m_{m}/output_vector_{m}_{ctau}_1_1.root'

def create_limit_tables(expected_limits, observed_limits, statistics, methods, portal):

    for method in methods:

        latex_path     = aux.makedir(f'figs/icelimit/{portal}/{method}')
        latex_filename = f'{latex_path}/limits_table_method_{method}.tex'
        
        def dprint(string, mode='a', end='\n'):
            print(string)
            with open(latex_filename, mode) as f:
                f.write(f'{string} {end}')

        dprint('', 'w')
        dprint(latex_header)    

        dprint('\\Large')
        dprint('DQCD Upper Limits')
        dprint('\\small')
        dprint('\\\\')
        dprint(f'm.mieskolainen@imperial.ac.uk')
        dprint('\\\\')
        dprint(f"{datetime.now():%d-%m-%Y %H:%M} ")
        dprint('\\\\')
        dprint('\\vspace{0.5em}')
        dprint('\\\\')
        
        dprint(f'Signal portal: {portal}')
        dprint('\\\\')
        dprint(f'Limit method: {method}')
        dprint('\\\\')
        dprint('\\vspace{5em}')
        dprint('\\\\')

        dprint('Expected CL95 signal upper limits on $r$')
        dprint('\\\\')
        dprint('$H0$: background only scenario')
        dprint('\\\\')

        dprint('\\begin{tabular}{|l||cc|c|cc|}')
        dprint('\\hline')
        dprint('Signal model point & $-2\\sigma$ & $-1\\sigma$ & median $r_{up}$ & $+1\\sigma$ & $+2\\sigma$ \\\\')
        dprint('\\hline')

        for key in expected_limits.keys():
            name = key.replace('_','-')
            dprint(f'{name}', end='')
            for i in range(len(expected_limits[key][method])):
                dprint(f' & {expected_limits[key][method][i]:.3g} ', end='')
            dprint('\\\\')
        dprint('\\hline')

        dprint('\\end{tabular}')
        dprint('\\\\')
        dprint('\\vspace{5em}')
        dprint('\\\\')

        dprint('\\small')
        dprint('Emulated (observed) CL95 signal upper limits on $r$')
        dprint('\\\\')
        dprint('$H1$: nominal signal ($r=1$) + background scenario')
        dprint('\\\\')

        dprint('\\begin{tabular}{|l||c|cc|}')
        dprint('\\hline')
        dprint('Signal model point & Observed $r_{up}$ & $\\langle B \\rangle$ (counts) & $\\langle S \\rangle$ (counts) \\\\')
        dprint('\\hline')

        for key in observed_limits.keys():
            name = key.replace('_','-')
            dprint(f'{name}', end='')
            for i in range(len(observed_limits[key][method])):

                B = statistics[key]['background']
                S = statistics[key]['signal']

                B = f'{float(f"{B:.2g}"):g}'
                S = f'{float(f"{S:.2g}"):g}'

                dprint(f' & {observed_limits[key][method][i]:.3g} & {B} & {S} ', end='')
            dprint('\\\\')
        dprint('\\hline')

        dprint('\\end{tabular}')
        dprint('')
    
    dprint('\\end{document}')

    # Compile latex
    os.system(f'pdflatex -output-directory {latex_path} {latex_filename}')


def limit_wrapper(files, path, methods, num_toys_obs=int(1E3), bg_regulator=0.0, s_regulator=0.0):
    
    expected_limits = {}
    observed_limits = {}
    statistics = {}

    for rootfile in files:
        
        h = {}
        for x in ['background', 'signal']:
            with uproot.open(f'{path}/{rootfile}')[x] as f:
                h[x] = TH1_to_numpy(f)

        # Counts (assumed Poisson)
        bg_expected   = np.sum(h['background']['counts'])
        s_hypothesis  = np.sum(h['signal']['counts'])

        # Regulate MC
        if  bg_expected <= 0:
            bg_expected = bg_regulator
        if s_hypothesis <= 0:
            s_hypothesis = s_regulator

        # Total systematics (MC uncertainties + others would go here)
        s_syst_error  = None #0.05 * s_hypothesis
        bg_syst_error = None #0.05 * bg_expected

        print(f'\n{rootfile}: background: {np.round(bg_expected,5)}, signal = {np.round(s_hypothesis,5)}')

        expected_limits[rootfile] = {'asymptotic': None, 'toys': None}
        observed_limits[rootfile] = {'asymptotic': None, 'toys': None}
        statistics[rootfile]      = {'background': bg_expected, 'signal': s_hypothesis}

        for method in methods:
            print(f'[method: {method}]')

            opt = CL_single_compute(method=method, observed=None, s_hypothesis=s_hypothesis,
                bg_expected=bg_expected, s_syst_error=s_syst_error, bg_syst_error=bg_syst_error, num_toys_obs=num_toys_obs)

            print(np.round(opt,4))
            expected_limits[rootfile][method] = opt

            opt = CL_single_compute(method=method, observed=s_hypothesis+bg_expected, s_hypothesis=s_hypothesis,
                                       bg_expected=bg_expected, s_syst_error=s_syst_error, bg_syst_error=bg_syst_error)
            print(np.round(opt,4))
            observed_limits[rootfile][method] = opt
    
    return expected_limits, observed_limits, statistics


def run_limits(path='/home/user/Desktop/output/', methods=['asymptotic', 'toys'], num_toys_obs=int(1E4), portal='vector'):
    """
    DQCD analysis 'KISS' limits
    
    root files path: /home/hep/khl216/CMSSW_10_2_13/src/statsForDS/output
    """

    m_values     = [2,5,10,15,20]
    ctau_values  = [10,50,100,500]

    experiment_label = '$41.6$ fb$^{-1}$, $\\sqrt{{s}} = 13$ TeV'
    process_label    = '$r = \\sigma / \\sigma_{gg \\rightarrow H \\times 0.01}$'
    

    # 1. Create all model point filenames
    files = []
    for m in m_values:
        for ctau in ctau_values:
            files.append(create_filename(m=m, ctau=ctau, portal=portal))

    # 2. Compute limits over all points
    expected_limits, observed_limits, statistics = limit_wrapper(files=files, path=path, methods=methods,
        num_toys_obs=num_toys_obs, bg_regulator=0.0, s_regulator=0.0)
    
    # 3. Create plots
    create_limit_plots(expected_limits=expected_limits,
        observed_limits=observed_limits, portal=portal, ctau_values=ctau_values, m_values=m_values, methods=methods,
        experiment_label=experiment_label, process_label=process_label, contour_min_r=1e-6, kx=2, ky=2)
    
    # 4. Produce tables
    create_limit_tables(expected_limits=expected_limits, observed_limits=observed_limits, statistics=statistics, methods=methods, portal=portal)

    ## Combine pdfs
    for method in methods:
        figpath = aux.makedir(f'figs/icelimit/{portal}/{method}')
        os.system(f'gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -sOutputFile={figpath}/limits_docs_{method}.pdf {figpath}/*.pdf')

if __name__ == "__main__":
    run_limits(path='/vols/cms/mmieskol/dqcd_limits/', methods=['asymptotic'], num_toys_obs=int(1e4), portal='vector')
