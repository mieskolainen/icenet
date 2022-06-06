# B/RK analysis [FIT] code [IMPLEMENTATION UNDER CONSTRUCTION]
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

"""
# icenet system paths
import _icepaths_

import h5py
import uproot
import uproot_methods
import pickle
import xgboost
import matplotlib.pyplot as plt
import os
import torch
import numpy as np

import numba
from numba import jit

# icenet
import iceplot
from icenet.tools import aux
from icenet.tools import io
from icenet.tools import prints

# icebrk
from icebrk import common
from icebrk import loop
from icebrk import histos
from icebrk import features


# Main function
#
def main() :
    
    ### Get input
    paths, args, cli, iodir = common.init()
    VARS = features.generate_feature_names(args['MAXT3'])
    targetdir = f'./figs/{args["rootname"]}/{cli.tag}/fit/'; os.makedirs(targetdir, exist_ok = True)
    
    # ====================================================================
    
    ## Binary matrix
    BMAT = aux.generatebinary(args['MAXT3'], args['MAXN'])
    print(f'POWERSET [NCLASS = {BMAT.shape[0]}]:')
    prints.print_colored_matrix(BMAT)

    ### Load data
    #output = pickle.load(open(iodir + 'xyz.pickle', 'rb')) # (r)ead (b)inary

    
    # ====================================================================
    ### Load HDF5 observables for unbinned fits

    def hdf5_get_events(obs, weight, filename):
        with h5py.File(filename, 'r+') as f:
            dataset = f[obs];    x = dataset[...]
            dataset = f[weight]; w = dataset[...]
        return x,w


    ### IMPLEMENTATION TO BE DONE HERE  >>>


    '''
    files = ['MC_S.hd5', 'MC_B.hd5', 'DA_S.hd5', 'DA_B.hd5']
    
    for i in range(len(files)):
        f = h5py.File(iodir + '/' + files[i], 'r+')

        for key in f.keys():
            print(f'key: {key}')
            dataset = f[key]
            data = dataset[...]
            print(data)
            print('\n')

        f.close()
    '''
    '''
    fig2, ax2 = iceplot.create_axes(**histos.obs_M, ratio_plot=False)
    bins = iceplot.stepspace(4.4, 6.0, 0.04)


    # MC SIGNAL
    x,w = hdf5_get_events(obs='M', weight='W0', filename=iodir + '/MC_S.hd5')
    counts, errs, bins, cbins = iceplot.hist(x, weights=w, bins=bins, density=False)
    ax2[0].hist(x=cbins, bins=bins, weights=counts, color=iceplot.imperial_dark_red, label='MC sig truth', alpha=0.5, **iceplot.hist_style_fill)

    # MC BACKGROUND (1,2,3 wrong assignments) TRIPLETS
    x,w = hdf5_get_events(obs='M', weight='W0', filename=iodir + '/MC_B.hd5')
    counts, errs, bins, cbins = iceplot.hist(x, weights=w, bins=bins, density=False)
    ax2[0].hist(x=cbins, bins=bins, weights=counts, color=iceplot.imperial_green, label='MC bgk truth', alpha=0.5, **iceplot.hist_style_fill)


    # MC SIGNAL
    x,w = hdf5_get_events(obs='M', weight='W0', filename=iodir + '/DA_S.hd5')
    counts, errs, bins, cbins = iceplot.hist(x, weights=w, bins=bins, density=False)
    ax2[0].hist(x=cbins, bins=bins, weights=counts, color=iceplot.imperial_dark_red, label='MC sig ($icenet$)', **iceplot.hist_style_step)

    # MC BACKGROUND (1,2,3 wrong assignments) TRIPLETS
    x,w = hdf5_get_events(obs='M', weight='W0', filename=iodir + '/DA_B.hd5')
    counts, errs, bins, cbins = iceplot.hist(x, weights=w, bins=bins, density=False)
    ax2[0].hist(x=cbins, bins=bins, weights=counts, color=iceplot.imperial_green, label='MC bgk ($icenet$)', **iceplot.hist_style_step)
    
    ax2[0].legend(frameon=False)
    fig2.savefig(f'{targetdir}/MC_decompose.pdf', bbox_inches='tight')
    
    '''
    # ====================================================================

    def set_aspect(ax, ratio=1.5):
        ratio = 1.0
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
        return

    # 2D histograms
    fig,ax = plt.subplots(nrows=1, ncols=2)

    k = 0
    for ID in ['S', 'B']:

        x,w = hdf5_get_events(obs='M',  weight='W0', filename=iodir + f'/MC_{ID}.hd5')
        y,w = hdf5_get_events(obs='q2', weight='W0', filename=iodir + f'/MC_{ID}.hd5')

        ax[k].hist2d(x, y, weights=w, bins=100)
        ax[k].set_aspect('auto')
        ax[k].set_xlabel('$M$ [GeV]')
        ax[k].set_ylabel('$q^2$ [GeV]$^2$')

        set_aspect(ax[k], ratio = 1.5)
        k +=1


    fig.tight_layout()
    fig.savefig(f'{targetdir}/h2_M_q2.pdf', bbox_inches='tight')


    print('\n' + __name__+ ' DONE')



if __name__ == '__main__' :

    main()

"""