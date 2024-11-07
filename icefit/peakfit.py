# Tag & Probe efficiency (and scale factor) estimation.
# Multiprocessing via Ray.
# 
# Notes:
# 
# - Keep all pdf functions normalized in the steering yml normalized,
#   otherwise not consistent (norm: True)
# 
# - Use 'chi2' or 'huber' loss if using weighted event histograms (either MC or data)
#   Use 'nll' for unweighted Poisson count histograms
#             and a weighted count histograms via a scale transform (experimental)
# 
# - For different fit types see: /docs/pdf/peakfit.pdf 
# 
# 
# Run with: python icefit/peakfit.py --analyze --group (--test_mode)
# 
# m.mieskolainen@imperial.ac.uk, 2024

import sys
sys.path.append(".")

import numpy as np
import multiprocessing
import pickle
import uproot
import copy
from termcolor import cprint
import matplotlib.pyplot as plt
from pprint import pprint

import os
from os import listdir
from os.path import isfile, join

from icefit import statstools
from icefit import icepeak

import ray

__VERSION__ = 0.07
__AUTHOR__  = 'm.mieskolainen@imperial.ac.uk'


# ========================================================================
# Input processing

def get_rootfiles_jpsi(input_path='/', years=[2016, 2017, 2018],
                       systematics=['Nominal'], test_mode=False):
    """
    Return rootfile names for the J/psi (muon) study.
    """
    all_years = []
    
    setup = [
    
        ## Num/Den type A
        {
            'NUM_DEN': ['LooseID', 'TrackerMuons'],
            'OBS':     ['absdxy'],
            'BINS':    [[1,2,3]]
        },
        {
            'NUM_DEN': ['LooseID', 'TrackerMuons'],
            'OBS':     ['absdxy_hack', 'pt'],
            'BINS':    [[1,2,3,4], [1,2,3,4,5]]
        },
        {
            'NUM_DEN': ['LooseID', 'TrackerMuons'],
            'OBS':     ['absdxy', 'pt'],
            'BINS':    [[1,2,3], [1,2,3,4,5]]
        },
        {
            'NUM_DEN': ['LooseID', 'TrackerMuons'],
            'OBS':     ['abseta', 'pt'],
            'BINS':    [[1,2,3,4,5], [1,2,3,4,5]]
        },
        {
            'NUM_DEN': ['LooseID', 'TrackerMuons'],
            'OBS':     ['eta'],
            'BINS':    [[1,2,3,4,5,6,7,8,9,10]]
        },
        
        ## Num/Den type B
        {
            'NUM_DEN': ['MyNum', 'MyDen'],
            'OBS':     ['absdxy'],
            'BINS':    [[1,2,3]]
        },
        {
            'NUM_DEN': ['MyNum', 'MyDen'],
            'OBS':     ['absdxy_hack', 'pt'],
            'BINS':    [[1,2,3,4], [1,2,3,4,5]]
        },
        {
            'NUM_DEN': ['MyNum', 'MyDen'],
            'OBS':     ['absdxy', 'pt'],
            'BINS':    [[1,2,3], [1,2,3,4,5]]
        },
        {
            'NUM_DEN': ['MyNum', 'MyDen'],
            'OBS':     ['abseta', 'pt'],
            'BINS':    [[1,2,3,4,5], [1,2,3,4,5]]
        },
        {
            'NUM_DEN': ['MyNum', 'MyDen'],
            'OBS':     ['eta'],
            'BINS':    [[1,2,3,4,5,6,7,8,9,10]]
        }
    ]
    
    if test_mode:
        
        cprint(f'get_rootfiles_jpsi: Using small test set of files (test_mode = True)', 'red')
        
        setup = [
            {
                'NUM_DEN': ['MyNum', 'MyDen'],
                'OBS':     ['absdxy'],
                'BINS':    [[1,2,3]]
            }
        ]
    
    # Loop over datasets
    for YEAR in years:
        info = {}
        
        for GENTYPE in ['JPsi_pythia8', f'Run{YEAR}_UL']: # MC or Data
            files = []
            
            for SYST in systematics:
                for s in setup:    
                    
                    NUM_DEN = s['NUM_DEN']
                    OBS     = s['OBS']
                    BINS    = s['BINS']

                    # 1D histograms
                    if   len(OBS) == 1:
                        
                        rootfile = f'{input_path}/Run{YEAR}_UL/{GENTYPE}/{SYST}/NUM_{NUM_DEN[0]}_DEN_{NUM_DEN[1]}_{OBS[0]}.root'
                        
                        # Binning
                        for BIN0 in BINS[0]:
                            
                            doublet = {'Pass': None, 'Fail': None}
                            for PASS in doublet.keys():
                                
                                tree = f'NUM_{NUM_DEN[0]}_DEN_{NUM_DEN[1]}_{OBS[0]}_{BIN0}_{PASS}'
                                file = {'OBS1': OBS[0], 'BIN1': BIN0, 'OBS2': None, 'BIN2': None, 'SYST': SYST, 'rootfile': rootfile, 'tree': tree}

                                doublet[PASS] = file
                            files.append(doublet)

                    # 2D histograms
                    elif len(OBS) == 2:
                        
                        rootfile = f'{input_path}/Run{YEAR}_UL/{GENTYPE}/{SYST}/NUM_{NUM_DEN[0]}_DEN_{NUM_DEN[1]}_{OBS[0]}_{OBS[1]}.root'
                        
                        # Binning
                        for BIN0 in BINS[0]:
                            for BIN1 in BINS[1]:
                                
                                doublet = {'Pass': None, 'Fail': None}
                                for PASS in doublet.keys():
                                    
                                    tree = f'NUM_{NUM_DEN[0]}_DEN_{NUM_DEN[1]}_{OBS[0]}_{BIN0}_{OBS[1]}_{BIN1}_{PASS}'
                                    file = {'OBS1': OBS[0], 'BIN1': BIN0, 'OBS2': OBS[1], 'BIN2': BIN1, 'SYST': SYST, 'rootfile': rootfile, 'tree': tree}
                                
                                    doublet[PASS] = file
                                files.append(doublet)
            
            info[GENTYPE] = files

        all_years.append({'YEAR': YEAR, 'info': info})

    return all_years

@ray.remote
def fit_task_ray(f, inputparam, savepath, YEAR, GENTYPE):
    return fit_task(f, inputparam, savepath, YEAR, GENTYPE)

def fit_task(f, inputparam, savepath, YEAR, GENTYPE):
    """
    Main fitting task for 'single' histogram fits
    """
    
    print(__name__ + f'.fit_task: Executing task "{f}" ...')
    
    param   = inputparam['param']
    fitfunc = inputparam['fitfunc']
    techno  = inputparam['techno']
    
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # Over 'Pass' and 'Fail' type
    for key in f.keys():
        
        SYST = f[key]["SYST"]
        tree = f[key]["tree"]
        hist = uproot.open(f[key]["rootfile"])[tree]

        # Wrap in a dictionary
        hist_    = {f'{key}': hist}
        fitfunc_ = {f'{key}': fitfunc}
        
        # Fit and analyze
        par,cov,var2pos,m1 = icepeak.binned_1D_fit(hist=hist_, param=param, fitfunc=fitfunc_, 
                                techno=techno, par_fixed=None)
        output = icepeak.analyze_1D_fit(hist=hist, param=param, fitfunc=fitfunc,
                                            techno=techno, par=par, cov=cov, par_fixed=None)
        
        # Create savepath
        total_savepath = f'{savepath}/Run{YEAR}/{GENTYPE}/{SYST}'
        if not os.path.exists(total_savepath):
            os.makedirs(total_savepath)

        # Save the fit plot
        plt.figure(output['fig'])
        plt.savefig(f'{total_savepath}/{tree}.pdf', bbox_inches='tight')
        plt.savefig(f'{total_savepath}/{tree}.png', bbox_inches='tight', dpi=300)
        
        # Save the pull histogram
        plt.figure(output['fig_pulls'])
        plt.savefig(f'{total_savepath}/{tree}__pulls.pdf', bbox_inches='tight')
        plt.savefig(f'{total_savepath}/{tree}__pulls.png', bbox_inches='tight', dpi=300)
        
        # Save parameter outputs
        with open(f'{total_savepath}/{tree}.log', "w") as file:
            print(par, file=file)
            print(m1.params, file=file)
            print(cov, file=file)
        
        # Draw MINOS contours
        if techno['draw_mnmatrix']:  
            cprint('Draw MINOS profile likelihood scan 2D contours', 'yellow')

            m1.draw_mnmatrix(figsize=(2.5*len(par), 2.5*len(par)))
            plt.figure(plt.gcf())
            plt.savefig(f'{total_savepath}/{tree}__mnmatrix.pdf', bbox_inches='tight')
            plt.savefig(f'{total_savepath}/{tree}__mnmatrix.png', bbox_inches='tight', dpi=300)
        
        # Save the fit numerical data
        par_dict, cov_arr = icepeak.iminuit2python(par=par, cov=cov, var2pos=var2pos)
        outdict = {'par':     par_dict,
                   'cov':     cov_arr,
                   'var2pos': var2pos,
                   'param':   param,
                   'techno':  techno}
        
        outdict = outdict | output # Combine

        filename = f"{total_savepath}/{tree}.pkl"
        pickle.dump(outdict, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        cprint(f'Fit results saved to: {filename} (pickle) \n\n', 'green')

        plt.close('all') # as last
    
    return True

@ray.remote
def fit_task_ray_multi(f, inputparam, savepath, YEAR, GENTYPE):
    return fit_task_multi(f, inputparam, savepath, YEAR, GENTYPE)

def fit_task_multi(f, inputparam, savepath, YEAR, GENTYPE):
    """
    Main fitting task for 'dual' ('Pass' and 'Fail' simultaneous) histogram fits
    """
    
    print(__name__ + f'.fit_task_multi: Executing task "{f}" ...')
    
    # Collect fit parametrization setup
    param   = inputparam['param']
    fitfunc = inputparam['fitfunc']
    techno  = inputparam['techno']

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    # Collect 'Pass' and 'Fail' type histograms
    hist = {}
    for key in f.keys():
        tree = f[key]["tree"]
        hist[key] = uproot.open(f[key]["rootfile"])[tree]
    
    # --------------------------------------------------------------------
    # Set fixed parameters
    
    d_pass = icepeak.hist_decompose(icepeak.TH1_to_numpy(hist['Pass']), param=param, techno=techno)
    d_fail = icepeak.hist_decompose(icepeak.TH1_to_numpy(hist['Fail']), param=param, techno=techno)
    
    par_fixed = {'C_pass': d_pass['num_counts_in_fit'],
                 'C_fail': d_fail['num_counts_in_fit']}
    
    # --------------------------------------------------------------------
    
    # Simultaneous fit of both    
    par,cov,var2pos,m1 = icepeak.binned_1D_fit(hist=hist, param=param, fitfunc=fitfunc,
                                            techno=techno, par_fixed=par_fixed)

    # Analyze and plot Pass, Fail seperately
    for idx, key in enumerate(f.keys()):
        
        output = icepeak.analyze_1D_fit(hist=hist[key], param=param, fitfunc=fitfunc[key],
                    techno=techno, par=par, cov=cov, par_fixed=par_fixed)

        SYST = f[key]["SYST"]
        tree = f[key]["tree"]
        
        # Create savepath
        total_savepath = f'{savepath}/Run{YEAR}/{GENTYPE}/{SYST}'
        if not os.path.exists(total_savepath):
            os.makedirs(total_savepath)

        # Save the fit plot
        plt.figure(output['fig'])
        plt.savefig(f'{total_savepath}/{tree}.pdf', bbox_inches='tight')
        plt.savefig(f'{total_savepath}/{tree}.png', bbox_inches='tight', dpi=300)
        
        # Save the pull histogram
        plt.figure(output['fig_pulls'])
        plt.savefig(f'{total_savepath}/{tree}__pulls.pdf', bbox_inches='tight')
        plt.savefig(f'{total_savepath}/{tree}__pulls.png', bbox_inches='tight', dpi=300)
        
        # Save parameter outputs
        if (idx == 0):
            with open(f'{total_savepath}/{tree}.log', "w") as file:
                print(par, file=file)
                print(m1.params, file=file)
                print(cov, file=file)
        
        # Draw MINOS contours
        if techno['draw_mnmatrix'] and (idx == 0):  
            cprint('Draw MINOS profile likelihood scan 2D contours', 'yellow')
            
            m1.draw_mnmatrix(figsize=(2.5*len(par), 2.5*len(par)))
            plt.figure(plt.gcf())
            plt.savefig(f'{total_savepath}/{tree}__mnmatrix.pdf', bbox_inches='tight')
            plt.savefig(f'{total_savepath}/{tree}__mnmatrix.png', bbox_inches='tight', dpi=300)
        
        # Save the fit numerical data
        par_dict, cov_arr = icepeak.iminuit2python(par=par, cov=cov, var2pos=var2pos)
        outdict = {'par':     par_dict,
                   'cov':     cov_arr,
                   'var2pos': var2pos,
                   'param':   param,
                   'techno':  techno}
        
        outdict = outdict | output # Combine
        
        filename = f"{total_savepath}/{tree}.pkl"
        pickle.dump(outdict, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        cprint(f'Fit results saved to: {filename} (pickle) \n\n', 'green')
        
        plt.close('all') # as last
    
    return True


def run_jpsi_peakfit(inputparam, savepath):
    """
    J/psi peak fitting with multiprocessing over kinematic 'hyperbins' via Ray
    """
    
    #np.seterr(all='print') # Numpy floating point error treatment
    
    param     = inputparam['param']
    num_cpus  = param['num_cpus']
    
    all_years = get_rootfiles_jpsi(input_path=param['input_path'],
                                   years=param['years'], systematics=param['systematics'],
                                   test_mode=param['test_mode'])

    pprint(all_years)

    # ----------------------------
    # Prepare multiprocessing
    
    if num_cpus == 0:
        num_cpus = multiprocessing.cpu_count()
    
    if num_cpus > 1:
        cprint(__name__ + f'.run_jpsi_peakfit: Fitting with {num_cpus} CPU cores using Ray', 'green')
        
        tmpdir = os.getcwd() + '/tmp'
        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)
        
        ray.init(num_cpus=num_cpus, _temp_dir=tmpdir) # Start Ray
        result_ids = []
    else:
        cprint(__name__ + f'.run_jpsi_peakfit: Fitting with {num_cpus} CPU cores', 'green')
    
    # ----------------------------
    # Execute fit
    
    for y in all_years:
        YEAR = y['YEAR']

        for GENTYPE in y['info']: # Data or MC type

            for f in y['info'][GENTYPE]:
                
                # Single fits
                if param['fit_type'] == 'single':
                    
                    if num_cpus > 1:
                        result_ids.append(fit_task_ray.remote(f, inputparam, savepath, YEAR, GENTYPE))
                    else:
                        fit_task(f, inputparam, savepath, YEAR, GENTYPE)

                # Multi-fits
                else:
                    
                    if num_cpus > 1:
                        result_ids.append(fit_task_ray_multi.remote(f, inputparam, savepath, YEAR, GENTYPE))
                    else:
                        fit_task_multi(f, inputparam, savepath, YEAR, GENTYPE)
                
    if num_cpus > 1:
        results = ray.get(result_ids)
        ray.shutdown()

    return True


def run_jpsi_tagprobe(inputparam, savepath):
    """
    Tag & Probe efficiency (& scale factors)
    """
    
    param  = inputparam['param']
    
    def tagprobe_single(tree, total_savepath):

        N       = {}
        N_err   = {}
        outdict = {}
        
        for PASS in ['Pass', 'Fail']:

            filename = f"{total_savepath}/{f'{tree}_{PASS}'}.pkl"
            cprint(__name__ + f'.run_jpsi_tagprobe: Reading fit results from: {filename} (pickle)', 'green')         
            outdict[PASS] = pickle.load(open(filename, "rb"))

            # Read out signal peak fit event count yield and its uncertainty
            N[PASS]     = outdict[PASS]['N']['S']
            N_err[PASS] = outdict[PASS]['N_err']['S']

        return N, N_err, outdict
    
    # ====================================================================
    ## Read filenames
    all_years = get_rootfiles_jpsi(input_path=param['input_path'], years=param['years'],
                                   test_mode=param['test_mode'])

    ### Loop over datasets
    for y in all_years:
        
        YEAR     = y['YEAR']
        
        mc_key   = f'JPsi_pythia8'
        data_key = f'Run{YEAR}_UL'

        # Loop over observables -- pick 'data_key'
        # (both data and mc have the same observables)
        for f in y['info'][data_key]:
            
            # Just a pick (both Pass and Fail will be used)
            KEY  = 'Pass'
            SYST = f[KEY]['SYST']
            
            # Create savepath
            total_savepath = f'{savepath}/Run{YEAR}/Efficiency/{SYST}'
            if not os.path.exists(total_savepath):
                os.makedirs(total_savepath)

            tree    = f[KEY]['tree'].replace(f"_{KEY}", "")
            eff     = {}
            eff_err = {}

            # Loop over data and MC
            for GENTYPE in [data_key, mc_key]:
                
                if param['fit_type'] == 'single':
                    """
                    Single type fits
                    """
                    
                    N,N_err,d = tagprobe_single(tree=tree, total_savepath=f'{savepath}/Run{YEAR}/{GENTYPE}/{SYST}')
                    
                    # ---------------------------------------
                    ## ** Compute signal efficiency using signal counts **
                    
                    if (N['Pass'] + N['Fail']) > 0:
                        eff[GENTYPE]     = N['Pass'] / (N['Pass'] + N['Fail'])
                        eff_err[GENTYPE] = statstools.tpratio_taylor(x=N['Pass'], y=N['Fail'], x_err=N_err['Pass'], y_err=N_err['Fail'])
                    else:
                        cprint('Signal efficiency cannot be extracted (set nan)', 'red')
                        eff[GENTYPE]     = np.nan
                        eff_err[GENTYPE] = np.nan
                    # ---------------------------------------

                else:
                    """
                    Dual type fits
                    """

                    N,N_err,d = tagprobe_single(tree=tree, total_savepath=f'{savepath}/Run{YEAR}/{GENTYPE}/{SYST}')

                    # ---------------------------------------    
                    ## ** Read out efficiency directly from the fit result **
                    
                    # Pick 'Pass' (both 'Pass' and 'Fail' contain same parameter info because joint fit)
                    d       = d['Pass']
                    par     = d['par']
                    cov     = d['cov']
                    var2pos = d['var2pos']
                    
                    ID    = 'eps__S'    # Signal efficiency
                    index = var2pos[ID]
                    
                    # Use directly the efficiency parameter fit driven uncertainty
                    if cov[index][index] > 0: # unstable cov is tagged with -1
                        eff[GENTYPE]     = par[ID]
                        eff_err[GENTYPE] = np.sqrt(cov[index][index])

                    # Use counts & their error based propagation
                    elif (N['Pass'] + N['Fail']) > 0:
                        cprint('Non-positive definite covariance, using counts and Taylor expansion', 'red')
                        eff[GENTYPE]     = par[ID]
                        eff_err[GENTYPE] = statstools.tpratio_taylor(x=N['Pass'], y=N['Fail'], x_err=N_err['Pass'], y_err=N_err['Fail'])

                    else:
                        cprint('Signal efficiency cannot be extracted (set nan)', 'red')
                        eff[GENTYPE]     = np.nan
                        eff_err[GENTYPE] = np.nan
                    # ---------------------------------------
                
                ### Print out
                cprint(f'[{GENTYPE}] | fit_type = {param["fit_type"]}', 'magenta')
                print(f'N_pass:     {N["Pass"]:0.1f} +- {N_err["Pass"]:0.1f} (signal yield)')
                print(f'N_fail:     {N["Fail"]:0.1f} +- {N_err["Fail"]:0.1f} (signal yield)')
                print(f'Efficiency: {eff[GENTYPE]:0.3f} +- {eff_err[GENTYPE]:0.3f} \n')
            
            # --------------------------------------- 
            ## ** Compute scale factor Data / MC **
            
            if (eff[data_key] > 0) and (eff[mc_key] > 0):
                scale     = eff[data_key] / eff[mc_key]
                scale_err = statstools.ratio_eprop(A=eff[data_key], B=eff[mc_key], \
                                sigmaA=eff_err[data_key], sigmaB=eff_err[mc_key], sigmaAB=0)
            else:
                cprint('Scale factor cannot be extracted (set nan)', 'red')
                scale     = np.nan
                scale_err = np.nan
            
            cprint(f'Data / MC:  {scale:0.3f} +- {scale_err:0.3f} (scale factor) \n', 'yellow')
            # --------------------------------------- 
            
            ### Save results
            outdict  = {'eff': eff, 'eff_err': eff_err, 'scale': scale, 'scale_err': scale_err}
            filename = f"{total_savepath}/{tree}.pkl"
            pickle.dump(outdict, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            cprint(f'Efficiency and scale factor results saved to: {filename} (pickle)', 'green')
            print('--------------------------')

    return True


def fit_and_analyze(p):
    """
    Main analysis steering
    """

    p_orig = copy.deepcopy(p)
    
    for VARIATION in p['param']['variations']:
        
        ## 1 (mass window shifted down)
        if   VARIATION == 'MASS-WINDOW-DOWN':
            p = copy.deepcopy(p_orig)
            p['param']['fitrange'] = np.array(p['param']['fitrange']) - 0.02
        
        ## 2 (mass window shifted up)
        elif VARIATION == 'MASS-WINDOW-UP':
            p = copy.deepcopy(p_orig)
            p['param']['fitrange'] = np.array(p['param']['fitrange']) + 0.02

        ## 3 (symmetric signal shape fixed)
        elif VARIATION == 'SYMMETRIC-SIGNAL_CB_asym_RBW':
            p = copy.deepcopy(p_orig)

            # ---------------------------
            # Fixed parameter values for symmetric 'CB (*) asym_RBW', where (*) is a convolution
            pairs = {'asym': 1e-6, 'n': 1.0 + 1e-6, 'alpha': 10.0}

            # Fix the parameters
            for key in pairs.keys():

                ind = p['param']['name'].index(f'p__{key}')
                x   = pairs[key]

                p['param']['start_values'][ind] = x
                p['param']['limits'][ind] = [x, x]
                p['param']['fixed'][ind]  = True
            # ------------------------

        ## 4 (Default setup)
        elif VARIATION == 'DEFAULT':
            p = copy.deepcopy(p_orig)
        
        else:
            raise Exception(f'Undefined systematic variation chosen: {VARIATION}')
        
        ### Execute yield fit and compute tag & probe efficiency
        outputdir = os.getcwd() + f'/output/peakfit/{p["param"]["output_name"]}/fitparam_{VARIATION}'
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)
        
        run_jpsi_peakfit(inputparam=p,  savepath=outputdir)
        run_jpsi_tagprobe(inputparam=p, savepath=outputdir)

    return True

def group_systematics(p):
    """
    Group and print results of systematic fit variations
    """

    for YEAR in p['param']['years']:

        d = {}

        for SYST in p['param']['systematics']:
            
            for VARIATION in p['param']['variations']:
                
                path  = os.getcwd() + f'/output/peakfit/{p["param"]["output_name"]}/fitparam_{VARIATION}/Run{YEAR}/Efficiency/{SYST}/'
                files = [f for f in listdir(path) if isfile(join(path, f))]

                for filename in files:
                    with open(path + filename, 'rb') as f:
                        x = pickle.load(f)
                    
                    if filename not in d:
                        d[filename] = {}

                    d[filename][SYST + '_' + VARIATION] = x
        
        ## Go through results and print
        print('')
        cprint(f'YEAR: {YEAR}', 'red')
        for hyperbin in list(d.keys()):
            cprint(hyperbin, 'magenta')
            for key in d[hyperbin].keys():
                print(f"Scale factor: {d[hyperbin][key]['scale']:0.4f} +- {d[hyperbin][key]['scale_err']:0.4f} \t ({key})")
        
        ## Save collected results
        savepath = os.getcwd() + f'/output/peakfit/{p["param"]["output_name"]}'
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        filename = f'{savepath}/peakfit_systematics_YEAR_{YEAR}.pkl'
        pickle.dump(d, open(filename, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
        cprint(f'Systematics grouped results saved to: {filename} (pickle)', 'green')

    return True

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze',     help="Fit and analyze", action="store_true")
    parser.add_argument('--group',       help="Collect and group results", action="store_true")
    parser.add_argument('--test_mode',   help="Fast test mode", action="store_true")
    parser.add_argument('--inputfile',   type=str, default='configs/peakfit/tune2.yml', help="Steering input YAML file", nargs='?')
    
    # Override (optional)
    parser.add_argument('--input_path',    type=str, default=None, help="Input path", nargs='?')
    parser.add_argument('--output_name',   type=str, default=None, help="Output name", nargs='?')
    parser.add_argument('--fit_type',      type=str, default=None, help="Fit type", nargs='?')
    parser.add_argument('--num_cpus',      type=int, default=None, help="Number of CPUs (0 for automatic)", nargs='?')
    parser.add_argument('--rng_seed',      type=int, default=None, help="Random seed", nargs='?')
    parser.add_argument('--loss_type',     type=str, default=None, help="Loss type", nargs='?')
    parser.add_argument('--trials',        type=int, default=None, help="Trials", nargs='?')
    
    parser.add_argument('--draw_mnmatrix', help="Draw 2D MINOS profiles", action="store_true")
    
    args = parser.parse_args()
    print(args)
    
    # Get YAML parameters
    p = {}
    p['param'], p['fitfunc'], p['cfunc'], p['techno'] = \
        icepeak.read_yaml_input(args.inputfile, fit_type=args.fit_type)
    
    if args.test_mode:
        p['param']['test_mode'] = True
    else:
        p['param']['test_mode'] = False
    
    # -------------------------------------------------
    # Command line overrides over steering card defaults
            
    if args.input_path is not None:
        p['param']['input_path'] = args.input_path
    
    if args.output_name is not None:
        p['param']['output_name'] = args.output_name
    
    if args.num_cpus is not None:
        p['param']['num_cpus'] = args.num_cpus
    
    if args.rng_seed is not None:
        p['techno']['rng_seed'] = args.rng_seed
    
    if args.loss_type is not None:
        p['techno']['loss_type'] = args.loss_type
    
    if args.trials is not None:
        p['techno']['trials'] = args.trials
    
    if args.draw_mnmatrix:
        p['techno']['draw_mnmatrix'] = True
    # -------------------------------------------------

    print('-----------------------------')
    cprint(f'peakfit {__VERSION__} ({__AUTHOR__})', 'magenta')
    print('')
    cprint('Main parameters:', 'green')
    print(p['param'])
    print('')
    cprint('Techno parameters:', 'green')
    print(p['techno'])
    print('-----------------------------')
    
    # Set random seed
    icepeak.set_random_seeds(p['techno']['rng_seed'])
    
    if args.analyze:
        fit_and_analyze(p)
    
    if args.group:
        group_systematics(p)
