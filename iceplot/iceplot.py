# Advanced histogramming & automated plotting functions
# 
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk


import matplotlib.pyplot as plt
import numpy as np
import math


def set_global_style(dpi=120, figsize=(4,3.75), font='serif', font_size=8, legend_fontsize=7, legend_handlelength=1):
    """ Set global plot style.
    """
    plt.rcParams['legend.fontsize']     = legend_fontsize
    plt.rcParams['legend.handlelength'] = legend_handlelength

    plt.rcParams['figure.dpi']     = dpi
    plt.rcParams['figure.figsize'] = figsize

    plt.rcParams['font.family']    = font
    plt.rcParams['font.size']      = font_size


# Colors
imperial_dark_blue  = (0, 0.24, 0.45)
imperial_light_blue = (0, 0.43, 0.69)
imperial_dark_red   = (0.65, 0.10, 0.0)
imperial_green      = (0.0, 0.54, 0.23)


""" Global marker styles

zorder : approximate plotting order
lw     : linewidth
ls     : linestyle
"""
errorbar_style  = {'zorder': 3, 'ls': ' ', 'lw': 1, 'marker': 'o', 'markersize': 2.5}
plot_style      = {'zorder': 2, 'ls': '-', 'lw': 1}
hist_style_step = {'zorder': 0, 'ls': '-', 'lw': 1, 'histtype': 'step'}
hist_style_fill = {'zorder': 0, 'ls': '-', 'lw': 1, 'histtype': 'stepfilled'}
hist_style_bar  = {'zorder': 0, 'ls': '-', 'lw': 1, 'histtype': 'bar'}


class hobj:
    """ Minimal histogram data object.
    """
    def __init__(self, counts = 0, errs = 0, bins = 0, cbins = 0):
        self.counts   = counts
        self.errs     = errs
        self.bins     = bins
        self.cbins    = cbins

        if (np.sum(counts) == 0):
            self.is_empty = True
        else:
            self.is_empty = False

    # + operator
    def __add__(self, other):

        if (self.is_empty == True): # Take the rhs
            return other

        if ((self.bins == other.bins).all() == False):
            raise(__name__ + ' + operator: cannot operator on different sized histograms')

        counts = self.counts + other.counts
        errs   = np.sqrt(self.errs**2   + other.errs**2)

        return hobj(counts, errs, bins, cbins)

    # += operator
    def __iadd__(self, other):

        if (self.is_empty == True): # Still empty
            return other

        if ((self.bins == other.bins).all() == False):
            raise(__name__ + ' += operator: cannot operator on different sized histograms')

        self.counts = self.counts + other.counts
        self.errs   = np.sqrt(self.errs**2 + other.errs**2)
        
        return self


def stepspace(start, stop, step):
    """ Linear binning edges between [start, stop]
    """
    return np.arange(start, stop + step, step)


def plot_horizontal_line(ax, color=(0.5,0.5,0.5), linewidth=0.9):
    """ For the ratio plot
    """
    xlim = ax.get_xlim()
    ax.plot(np.linspace(xlim[0], xlim[1], 2), np.array([1,1]), color=color, linewidth=linewidth)


def tick_calc(lim, step, N=6):
    """ Tick spacing calculator.
    """
    return [np.round(lim[0] + i*step, N) for i in range(1+math.floor((lim[1]-lim[0])/step))]

def set_axis_ticks(ax, ticks, dim='x'):
    """ Set ticks of the axis.
    """
    if   (dim == 'x'):
        ax.set_xticks(ticks)
        ax.set_xticklabels(list(map(str, ticks)))
    elif (dim == 'y'):
        ax.set_yticks(ticks)
        ax.set_yticklabels(list(map(str, ticks)))

def tick_creator(ax, xtick_step=None, ytick_step=None, ylim_ratio=(0.7, 1.3),
        ratio_plot=True, minorticks_on=True, ytick_ratio_step=0.15, labelsize=9,
        labelsize_ratio=8, **kwargs) :
    """ Axis tick constructor.
    """

    # Get limits
    xlim = ax[0].get_xlim()
    ylim = ax[0].get_ylim()

    # X-axis
    if (xtick_step is not None):
        ticks = tick_calc(lim=xlim, step=xtick_step)
        set_axis_ticks(ax[-1], ticks, 'x')

    # Y-axis
    if (ytick_step is not None):    
        ticks = tick_calc(lim=ylim, step=ytick_step)
        set_axis_ticks(ax[0], ticks, 'y')

    # Y-ratio-axis
    if ratio_plot:
        ax[0].tick_params(labelbottom=False)
        ax[1].tick_params(axis='y', labelsize=labelsize_ratio)

        ticks = tick_calc(lim=ylim_ratio, step=ytick_ratio_step)
        ticks = ticks[1:-1] # Remove the first and the last
        set_axis_ticks(ax[1], ticks, 'y')

    # Tick settings
    for a in ax:
        if minorticks_on: a.minorticks_on()
        a.tick_params(top=True, bottom=True, right=True, left=True, which='both', direction='in', labelsize=labelsize)

    return ax

def create_axes(xlabel='$x$', ylabel=r'Counts', ylabel_ratio='Ratio',
    xlim=(0,1), ylim=None, ylim_ratio=(0.7, 1.3),
    ratio_plot=True, figsize=(5,4), fontsize=9, units='', **kwargs):
    """ Axes creator.
    """

    # Create subplots
    N = 2 if ratio_plot else 1
    gridspec_kw = {'height_ratios': (3.333, 1) if ratio_plot else (1,), 'hspace': 0.0}
    fig, ax = plt.subplots(N,  figsize=figsize, gridspec_kw=gridspec_kw)
    ax = [ax] if (N == 1) else ax

    # Axes limits
    for a in ax: a.set_xlim(*xlim)

    if ylim is not None:
        ax[0].set_ylim(*ylim)

    # Axes labels
    if units:
        if kwargs['density']:
            ylabel = f'{ylabel} / [{units}]'
        else:
            binwidth = kwargs['bins'][1] - kwargs['bins'][0]
            ylabel = f'{ylabel} / [{binwidth:.2f} {units}]'
        xlabel = f'{xlabel} [{units}]'
    
    ax[0].set_ylabel(ylabel, fontsize=fontsize)
    ax[-1].set_xlabel(xlabel, fontsize=fontsize)

    # Ratio plot
    if ratio_plot:
        ax[1].set_ylabel(ylabel_ratio, fontsize=fontsize)
        ax[1].set_ylim(*ylim_ratio)

    # Setup ticks
    ax = tick_creator(ax=ax, ratio_plot=ratio_plot, **kwargs)

    return fig, ax

def ordered_legend(ax=None, order=None, frameon=False, unique=False, **kwargs):
    """ Ordered legends.
    """

    def unique_everseen(seq, key=None):
        seen = set()
        seen_add = seen.add
        return [x for x,k in zip(seq,key) if not (k in seen or seen_add(k))]

    if ax is None: ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()

    # Sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    
    # Sort according to a given list, which may be incomplete
    if order is not None: 
        keys=dict(zip(order,range(len(order))))
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t, keys=keys: keys.get(t[0],np.inf)))

    # Keep only the first of each handle
    if unique:  labels, handles= zip(*unique_everseen(zip(labels,handles), key = labels)) 
    ax.legend(handles, labels, frameon=frameon, **kwargs)

    return (handles, labels)

def edge2centerbins(bins) :
    """ Get centerbins from edgebins.
    """
    return (bins[1:] + bins[0:-1])/2

def ratioerr(A, B, sigma_A, sigma_B, sigma_AB = 0, EPS = 1E-15):
    """ Ratio f(A,B) = A/B error, by Taylor expansion of f.
    """
    A[np.abs(A) < EPS] = EPS
    B[np.abs(B) < EPS] = EPS
    return np.abs(A/B) * np.sqrt((sigma_A/A)**2 + (sigma_B/B)**2 - 2*sigma_AB/(A*B))


def hist(x, bins=30, weights=None, density=False):
    """ Calculate a histogram.
    
    ** Implement under/overflows !! **
    """

    # Calculate histogram
    if weights is None:
        weights = np.ones(x.shape)

    counts, bins = np.histogram(x, bins=bins, weights=weights)
    cbins = edge2centerbins(bins)

    # Input data to histogram bins
    inds = np.digitize(x, bins)
    errs = np.asarray([np.linalg.norm(weights[inds==k],2) for k in range(1, len(bins))])

    # Normalize to density, take into account non-uniform binning
    if density:
        norm = ((bins[1:] - bins[0:-1])) * weights.sum()
        counts /= norm
        errs   /= norm

    return counts, errs, bins, cbins

def hist_obj(x, bins=30, weights=None, density=False):
    """ A wrapper to return a histogram object.
    """
    counts, errs, bins, cbins = hist(x, bins=bins, weights=weights, density=density)
    return hobj(counts, errs, bins, cbins)

def generate_colormap():
    """ Default colormap.
    """
    # Take colors
    color = plt.cm.Set1(np.linspace(0,1,10))

    # Add black 
    black = np.ones((1,4))
    black[:,0:3] = 0.0
    color = np.concatenate((black, color))    

    return color

def hist_filled_error(ax, bins, cbins, y, err, color, **kwargs):
    """ Stephist style error.
    """
    new_args = kwargs.copy()
    new_args['lw'] = 0
    new_args.pop('histtype', None) # Remove

    ax.fill_between(bins[0:-1], y-err, y+err, step='post', alpha=0.3, color=color, **new_args)

    # The last bin
    ax.fill_between(bins[-2:],  (y-err)[-2:], (y+err)[-2:], step='pre', alpha=0.3, color=color, **new_args)


def superplot(data : dict, observable=None, ratio_plot=True, yscale='linear', ratio_error_plot=True, legend_counts = False, color=None):
    """ Superposition (overlaid) plotting.
    """

    if observable == None:
        observable = data[0]['obs']

    fig, ax = create_axes(**observable, ratio_plot=ratio_plot)

    if color == None:
        color = generate_colormap()


    legend_labels = []

    # Plot histograms
    for i in range(len(data)):

        if data[i]['hdata'].is_empty:
            print(__name__ + f'.superplot: Skipping empty histogram for entry {i}')
            continue

        c = data[i]['color']
        if c is None: c = color[i]

        counts = data[i]['hdata'].counts
        errs   = data[i]['hdata'].errs
        bins   = data[i]['hdata'].bins
        cbins  = data[i]['hdata'].cbins

        label = data[i]['legend']
        if legend_counts == True:
            label += f' $N={np.sum(data[i]["hdata"].counts):.1f}$'

        legend_labels.append(label)

        if   data[i]['hfunc'] == 'hist' :
            ax[0].hist(x=cbins, bins=bins, weights=counts, color=c, label=label, **data[i]['style'])
            hist_filled_error(ax=ax[0], bins=bins, cbins=cbins, y=counts, err=errs, color=c, **data[i]['style'])

        elif data[i]['hfunc'] == 'errorbar' :
            ax[0].errorbar(x=cbins, y=counts, yerr=errs, color=c, label=label, **data[i]['style'])

        elif data[i]['hfunc'] == 'plot' :
            ax[0].plot(cbins, counts, color=c, label=label, **data[i]['style'])
            
            new_args = data[i]['style'].copy()
            new_args['lw'] = 0
            ax[0].fill_between(cbins, counts-errs, counts+errs, alpha=0.3, color=c, **new_args)

    # Plot ratiohistograms
    if ratio_plot:

        plot_horizontal_line(ax[1])

        for i in range(len(data)):

            if data[i]['hdata'].is_empty:
                print(__name__ + f'.superplot: Skipping empty histogram for entry {i} (ratioplot)')
                continue

            c = data[i]['color']
            if c is None: c = color[i]

            A        = data[i]['hdata'].counts
            B        = data[0]['hdata'].counts
            sigma_A  = data[i]['hdata'].errs
            sigma_B  = data[0]['hdata'].errs
            sigma_AB = 0
            ratio_errs = ratioerr(A=A, B=B, sigma_A=sigma_A, sigma_B=sigma_B, sigma_AB=sigma_AB)

            EPS      = 1E-30
            ratio    = data[i]['hdata'].counts / (data[0]['hdata'].counts + EPS)
            bins     = data[i]['hdata'].bins
            cbins    = data[i]['hdata'].cbins

            # If no errors turned on
            if ratio_error_plot == False:
                ratio_errs = np.zeros(ratio_errs.shape)

            if   data[i]['hfunc'] == 'hist':
                ax[1].hist(x=cbins, bins=bins, weights=ratio, color=c, **data[i]['style'])        
                hist_filled_error(ax=ax[1], bins=bins, cbins=cbins, y=ratio, err=ratio_errs, color=c, **data[i]['style'])

            elif data[i]['hfunc'] == 'errorbar':
                ax[1].errorbar(x=cbins, y=ratio, yerr=ratio_errs, color=c, **data[i]['style'])

            elif data[i]['hfunc'] == 'plot':
                ax[1].plot(cbins, ratio, color=c, **data[i]['style'])

                new_args = data[i]['style'].copy()
                new_args['lw'] = 0
                ax[1].fill_between(cbins, ratio-ratio_errs, ratio+ratio_errs, alpha=0.3, color=c, **new_args)

    # Legend
    if legend_labels != []:
        ordered_legend(ax = ax[0], order=legend_labels)

    # Log y-scale
    ax[0].set_yscale(yscale)

    return fig, ax
