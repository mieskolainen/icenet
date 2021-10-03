# Simple Monte Carlo routines
#
# m.mieskolainen@imperial.ac.uk, 2021

import numpy as np
import numba
import matplotlib.pyplot as plt
from tqdm import tqdm

# icenet system paths
import sys
sys.path.append(".")

from icenet.tools import icevec


# ----------------------------------------------
c_const    = 3E8             # Speed of light (m/sec)
hbar_const = 6.582119514E-25 # [GeV x sec]
# ----------------------------------------------


@numba.njit
def U(a,b, size=(1,)):
    """ Uniform random numbers from [a,b]
    """
    return a + (b-a)*np.random.random(size=size)

#@numba.njit
def randpow(a, b, g, size=(1,)):
    """ Power-law random numbers for pdf(x) ~ x^{g-1} for a <= x <= b
    """
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    
    return (ag + (bg - ag)*r)**(1.0/g)


@numba.njit
def randexp(u, size=(1,)):
    """ Exponential random variables pdf(x) ~ 1/u exp(-x/u) with mean u
    """
    return -u * np.log(1 - np.random.random(size=size))


@numba.njit
def twobody(p, m1, m2):
    """
    2-body isotropic decay routine according to dLIPS_2 flat phase space

    Args:      
           p : 4-momentum of the mother (in the lab frame, usually)
          m1 : Daughter 1 invariant mass
          m2 : Daughter 2 invariant mass

    Returns:
       p1,p2 : Daughter 4-momentum
               in the the same frame as the mother was defined
    """

    # Mother mass
    m0   = p.m

    # Energies, and momentum absolute (back to back)
    e1   = 0.5 * (m0**2 + m1**2 - m2**2) / m0
    e2   = 0.5 * (m0**2 + m2**2 - m1**2) / m0
    pabs = 0.5 * np.sqrt( (m0 - m1 - m2) * (m0 + m1 + m2) \
               * (m0 + m1 - m2) * (m0 - m1 + m2) ) / m0

    # Isotropic angles in a spherical system
    costheta = 2.0 * np.random.rand() - 1.0 # [-1,1]
    sintheta = np.sqrt(1.0 - costheta**2)
    phi      = 2.0 * np.pi * np.random.rand()  # [0,2pi]

    # To cartesian
    pX       = pabs * sintheta * np.cos(phi)
    pY       = pabs * sintheta * np.sin(phi)
    pZ       = pabs * costheta

    # 4-momenta now defined in the mother rest frame
    p1 =  icevec.vec4( pX,  pY,  pZ, e1)
    p2 =  icevec.vec4(-pX, -pY, -pZ, e2)

    # Then boost daughters into the original frame
    sign = 1
    p1.boost(b=p, sign=sign)
    p2.boost(b=p, sign=sign)
    
    return p1,p2


def Gamma2tau(Gamma):
    """ Width to mean lifetime
    """
    return hbar_const / Gamma


def tau2Gamma(tau):
    """ Mean lifetime to width
    """
    return hbar_const / tau


def resonance_generator(M, ctau, pt2, rap, N=1000):
    """
    Simulation massive resonance with (M, ctau) parameters,
    having production kinematics (pt2, rapidity, phi ~ flat)
    
    Returns:
        p4 : Array of 4-momentum (resonance 4-momentum)
        x4 : Array of 4-position (resonance decay 4-position)
    """
    tau    = ctau / c_const # Rest frame mean lifetime

    p4_arr = []
    x4_arr = []

    for i in range(N):
        
        ### Production kinematics

        pt2_value = pt2()
        rap_value = rap()
        phi_value = U(0, 2*np.pi)[0]

        p  = icevec.vec4()
        p.setPt2RapPhiM2(pt2=pt2_value, rap=rap_value, phi=phi_value, m2=M**2)        
        
        ### Flight kinematics

        # Sample lifetime in the rest frame (sec)
        t  = randexp(tau)[0]

        # Compute decay length distance (meter) in the lab
        # beta = |p|/E, gamma = E/m
        t_lab = p.gamma * t
        d_lab = p.beta * p.gamma * c_const * t

        # Create 4-position vector of the decay vertex in the lab frame
        p3 = p.p3
        x3 = (p3 / p.p3mod) * d_lab
        x  = icevec.vec4(x=x3[0], y=x3[1], z=x3[2], t=t_lab)

        ### Save them
        p4_arr.append(p)
        x4_arr.append(x)

    return p4_arr, x4_arr


def outer_sim_loop(M_values, ctau_values, pt2, rap, acc_func, N):
    """
    Simulation helper (wrapper) loop

    Returns:
        Z: Acceptance probability matrix
    """
    Z = np.zeros((len(ctau_values), len(M_values)))

    for i in range(len(ctau_values)):
        for j in range(len(M_values)):

            ctau           = ctau_values[i]
            M              = M_values[j]
            p4_arr, x4_arr = resonance_generator(M=M, ctau=ctau, pt2=pt2, rap=rap, N=N)

            # Evaluate fiducial acceptance function
            Z[i,j] = np.max([np.sum(acc_func(p4=p4_arr, x4=x4_arr)) / N, 1/N])

    return Z


def outer_sim_loop_2(M, ctau, pt2_values, rap_values, acc_func, N):
    """
    Simulation helper (wrapper) loop

    Returns:
        Z: Acceptance probability matrix
    """
    Z = np.zeros((len(pt2_values), len(rap_values)))

    for i in range(len(pt2_values)):
        for j in range(len(rap_values)):

            pt2            = lambda : pt2_values[i]
            rap            = lambda : rap_values[j]
            p4_arr, x4_arr = resonance_generator(M=M, ctau=ctau, pt2=pt2, rap=rap, N=N)

            # Evaluate fiducial acceptance function
            Z[i,j] = np.max([np.sum(acc_func(p4=p4_arr, x4=x4_arr)) / N, 1/N])

    return Z


def spherical_acceptance(p4, x4):
    """
    Spherical (geometric) acceptance function.
    
    Args:
        p4: Array of 4-momentum
        x4: Array of 4-position

    Params:
        R : Spherical detector radius (extend to more complex) [global]
    
    Returns:
        Array of True/False for each event (accepted or not)
    """
    global R
    d = np.array([x4[i].p3mod for i in range(len(x4))])

    return d < R

def set_aspect_true_equal(ax):
    """
    Set plot square sized.
    """
    ax = ax.set_aspect(np.diff(ax.get_xlim())/np.diff(ax.get_ylim()))
    return ax

def annotate_heatmap(x, y, Z, ax):
    """
    Annotate heatmap with text (broken function, fix TBD).
    """
    plt.sca(ax) # Activate axes
    
    for i in range(len(y)-1):
        for j in range(len(x)-1):

            yy = (y[i+1] + y[i])/2
            xx = (x[j+1] + x[j])/2

            plt.text(xx, yy, f'{Z[i,j]:0.1E}',
                     horizontalalignment = 'center',
                     verticalalignment   = 'center')
    return ax


def produce_acceptance_sim(N=1000):
    """
    Simulate fiducial (geometric) acceptance
    by looping over (M,ctau) or (Pt,Rap) pairs, others being fixed.
    """
    print('produce_acceptance_sim: Simulating LLP geometric acceptance maps ...')

    global R

    ### (M,ctau) plots

    pt2_values  = np.array([2, 10, 50]) ** 2
    rap_values  = np.array([0, 2.5, 5])
    
    ctau_values = np.logspace(-3,  2, 8)
    M_values    = np.logspace(0.1, 2, 6)

    for R in [1,10]:

        fig,ax = plt.subplots(len(pt2_values), len(rap_values), figsize=(12,12))

        for i in range(len(pt2_values)):
            for j in tqdm(range(len(rap_values))):

                # Define pt2 and rapidity distributions here!!

                pt2 = lambda : pt2_values[i]
                rap = lambda : rap_values[j]

                """
                # Distributions
                
                pt2 = lambda : randexp(30)**2
                rap = lambda : U(0,4)
                """

                Z = outer_sim_loop(M_values=M_values, ctau_values=ctau_values, \
                    pt2=pt2, rap=rap, N=N, acc_func=spherical_acceptance)

                # ---------------------------
                ### Plot it
                from matplotlib.colors import LogNorm
                x,y = np.meshgrid(M_values, ctau_values)

                # Turn into millimeters
                y *= 1000

                c = ax[i,j].pcolor(x, y, Z, cmap='RdBu', norm=LogNorm(vmin=Z.min(), vmax=Z.max()), shading='auto')

                ax[i,j].axis([x.min(), x.max(), y.min(), y.max()])
                ax[i,j].set_xscale('log')
                ax[i,j].set_yscale('log')
                ax[i,j].set_title(f'[$P_t = {np.sqrt(pt2())}$ GeV, $Y = {rap()}$] | decay length $d \\leq {R:0.1f}$ m', fontsize=7)

                if j == 0:
                    ax[i,j].set_ylabel('$c \\tau_0$ (mm)')
                if i == len(pt2_values)-1:
                    ax[i,j].set_xlabel('$M$ (GeV)')

                #ax[i,j] = annotate_heatmap(x=M_values, y=ctau_values, Z=Z, ax=ax[i,j])
                
                if j == len(rap_values)-1:
                    fig.colorbar(c, ax=ax[i,j])
                
                #ax[i,j] = set_aspect_true_equal(ax=ax[i,j])
                # ---------------------------

        plt.savefig(f'./figs/LLP_geometric_acc_M_ctau_R={R:0.0f}.pdf', bbox_inches='tight')
        plt.close()

    ### (Rap, Pt) plots
    M_values    = np.array([2, 10, 20])
    ctau_values = np.array([1E-1, 1e0, 1e1])
    
    pt2_values  = np.linspace(0.1, 50, 8) ** 2
    rap_values  = np.linspace(0, 5, 6)

    for R in [1,10]:

        fig,ax = plt.subplots(len(M_values), len(ctau_values), figsize=(12,12))

        for i in range(len(M_values)):
            for j in tqdm(range(len(ctau_values))):

                M          = M_values[i]
                ctau       = ctau_values[j]

                Z = outer_sim_loop_2(M=M, ctau=ctau, \
                    pt2_values=pt2_values, rap_values=rap_values, N=N, acc_func=spherical_acceptance)

                # ---------------------------
                ### Plot it
                from matplotlib.colors import LogNorm

                x,y = np.meshgrid(rap_values, np.sqrt(pt2_values))

                c = ax[i,j].pcolor(x, y, Z, cmap='RdBu', norm=LogNorm(vmin=Z.min(), vmax=Z.max()), shading='auto')

                ax[i,j].axis([x.min(), x.max(), y.min(), y.max()])
                #ax[i,j].set_xscale('log')
                #ax[i,j].set_yscale('log')
                ax[i,j].set_title(f'[$M = {M}$ GeV, $c\\tau_0 = {ctau*1000}$ mm] | decay length $d \\leq {R:0.1f}$ m', fontsize=7)

                if j == 0:
                    ax[i,j].set_ylabel('$P_t$ (GeV)')
                if i == len(M_values)-1:
                    ax[i,j].set_xlabel('rapidity $Y$')

                #ax[i,j] = annotate_heatmap(x=rap_values, y=np.sqrt(pt2_values), Z=Z, ax=ax[i,j])

                if j == len(ctau_values)-1:
                    fig.colorbar(c, ax=ax[i,j])

                #ax[i,j] = set_aspect_true_equal(ax=ax[i,j])
                # ---------------------------

        plt.savefig(f'./figs/LLP_geometric_acc_Rap_Pt_R={R:0.0f}.pdf', bbox_inches='tight')
        plt.close()

#produce_acceptance_sim()

def test_toy_pt_spectrum():

    pt2 = randpow(a=10**2, b=13000**2, g=-0.7, size=10000)
    pt  = np.sqrt(pt2)
    
    print(f'<pt> = {np.mean(pt)}')

    fig,ax = plt.subplots(2, 1)
    
    bins = np.linspace(1, 220,100)
    ax[0].hist(pt, bins)
    ax[0].set_xlabel('$p_T$ (GeV)')
    
    
    bins = np.logspace(np.log10(1), np.log10(220), 100)
    ax[1].hist(pt, bins)
    ax[1].set_yscale('log')
    ax[1].set_xscale('log')
    ax[1].set_xlabel('$p_T$ (GeV)')

    plt.show()

