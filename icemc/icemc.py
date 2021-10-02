# Simple Monte Carlo routines
#
# m.mieskolainen@imperial.ac.uk, 2021

import numpy as np
import numba
import matplotlib.pyplot as plt
from tqdm import tqdm

from icenet.tools import icevec


c_const    = 3E8             # Speed of light (m/sec)
hbar_const = 6.582119514E-25 # [GeV x sec]


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


def sim_loop(M, ctau, pt2, rap, N=1000):

    tau = ctau / c_const

    t_values = np.zeros(N)
    d_values = np.zeros(N)
    beta_gamma_values = np.zeros(N)

    for i in range(N):

        pt2_value = pt2()
        rap_value = rap()
        phi_value = U(0, 2*np.pi)[0]

        p   = icevec.vec4()
        p.setPt2RapPhiM2(pt2=pt2_value, rap=rap_value, phi=phi_value, m2=M**2)        
        
        # Sample lifetime in the rest frame (sec)
        t   = randexp(tau)[0]

        # Compute decay distance (meter) in the lab
        # beta = |p|/E, gamma = E/m
        t_lab = p.gamma * t
        d_lab = p.beta * p.gamma * c_const * t


        t_values[i] = t_lab
        d_values[i] = d_lab
        beta_gamma_values[i] = p.gamma * p.beta


    return d_values


def outer_sim_loop(M_values, ctau_values, pt2, rap, ax, acceptance_func, N=100):

    z = np.zeros((len(ctau_values), len(M_values)))
    d = np.zeros((len(ctau_values), len(M_values)))

    for i in range(len(ctau_values)):
        for j in range(len(M_values)):

            ctau      = ctau_values[i]
            M         = M_values[j]
            decay3pos = sim_loop(M=M, ctau=ctau, pt2=pt2, rap=rap, N=N)

            # Evaluate fiducial acceptance function
            z[i,j] = np.max([np.sum(acceptance_func(decay3pos)) / N, 1/N])

    # --------------------------------------------------------------------
    # Plot it
    from matplotlib.colors import LogNorm
    x,y    = np.meshgrid(M_values, ctau_values)

    # Turn into millimeters
    y *= 1000

    z_min, z_max = np.min(z), np.max(z)
    
    c = ax.pcolormesh(x, y, z, cmap='RdBu', norm=LogNorm(vmin=z.min(), vmax=z.max()), shading='auto')
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    ax.set_xscale('log')
    ax.set_yscale('log')

    return ax,c


def outer_sim_loop_2(M, ctau, pt2_values, rap_values, ax, acceptance_func, N=100):

    z = np.zeros((len(pt2_values), len(rap_values)))
    d = np.zeros((len(pt2_values), len(rap_values)))

    for i in range(len(pt2_values)):
        for j in range(len(rap_values)):

            pt2       = lambda : pt2_values[i]
            rap       = lambda : rap_values[j]
            decay3pos = sim_loop(M=M, ctau=ctau, pt2=pt2, rap=rap, N=N)

            # Evaluate fiducial acceptance function
            z[i,j] = np.max([np.sum(acceptance_func(decay3pos)) / N, 1/N])

    # --------------------------------------------------------------------
    # Plot it
    from matplotlib.colors import LogNorm

    x,y    = np.meshgrid(rap_values, np.sqrt(pt2_values))
    z_min, z_max = np.min(z), np.max(z)

    c = ax.pcolormesh(x, y, z, cmap='RdBu', norm=LogNorm(vmin=z.min(), vmax=z.max()), shading='auto')
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    #ax.set_xscale('log')
    #ax.set_yscale('log')

    return ax,c


def test_acceptance_sim():
    """
    Simulate fiducial (geometric) acceptance
    """

    # Number of events
    N = 10000

    # --------------------------------------------------------------------
    # (M,ctau) plots

    pt2_values  = np.array([2, 10, 50]) ** 2
    rap_values  = np.array([0, 2.5, 5])
    
    ctau_values = np.logspace(-3,  2, 12)
    M_values    = np.logspace(0.1, 2, 10)

    for R in [1,10]:

        # Fiducial (geometric) acceptance function definition
        def acceptance_func(decay3pos, daughter4mom=None):
            return decay3pos < R
        
        fig,ax = plt.subplots(len(pt2_values), len(rap_values), figsize=(12,12))

        for i in range(len(pt2_values)):
            for j in range(len(rap_values)):

                # Define pt2 and rapidity distributions here!!

                pt2 = lambda : pt2_values[i]
                rap = lambda : rap_values[j]

                """
                # Distributions
                
                pt2 = lambda : randexp(30)**2
                rap = lambda : U(0,4)
                """

                ax[i,j], c = outer_sim_loop(M_values=M_values, ctau_values=ctau_values, \
                    pt2=pt2, rap=rap, N=N, acceptance_func=acceptance_func, ax=ax[i,j])

                # ---------------------------

                ax[i,j].set_title(f'[$P_t = {np.sqrt(pt2())}$ GeV, $Y = {rap()}$] | decay length $d \\leq {R:0.1f}$ m', fontsize=7)

                if j == 0:
                    ax[i,j].set_ylabel('$c \\tau_0$ (mm)')
                if i == len(pt2_values)-1:
                    ax[i,j].set_xlabel('$M$ (GeV)')

                #if i == len(pt2_values)-1 and j == len(rap_values)-1:
                #    fig.colorbar(c, ax=ax[i,j])

        plt.savefig(f'LLP_geometric_acc_M_ctau_R={R:0.0f}.pdf', bbox_inches='tight')
        plt.close()

    # --------------------------------------------------------------------
    # (rap, pt) plots

    M_values    = np.array([2, 10, 20])
    ctau_values = np.array([1E-1, 1e0, 1e1])
    
    pt2_values  = np.linspace(0.1, 50, 12) ** 2
    rap_values  = np.linspace(0, 5, 10)

    for R in [1,10]:

        # Fiducial (geometric) acceptance function definition
        def acceptance_func(decay3pos, daughter4mom=None):
            return decay3pos < R
            #return (10 <= d) & (d <= 20) # "Shell-detector"

        fig,ax = plt.subplots(len(M_values), len(ctau_values), figsize=(12,12))

        for i in range(len(M_values)):
            for j in range(len(ctau_values)):

                M          = M_values[i]
                ctau       = ctau_values[j]

                ax[i,j], c = outer_sim_loop_2(M=M, ctau=ctau, \
                    pt2_values=pt2_values, rap_values=rap_values, N=N, acceptance_func=acceptance_func, ax=ax[i,j])

                # ---------------------------

                ax[i,j].set_title(f'[$M = {M}$ GeV, $c\\tau_0 = {ctau*1000}$ mm] | decay length $d \\leq {R:0.1f}$ m', fontsize=7)

                if j == 0:
                    ax[i,j].set_ylabel('$P_t$ (GeV)')
                if i == len(M_values)-1:
                    ax[i,j].set_xlabel('rapidity $Y$')

                #if i == len(pt2_values)-1 and j == len(rap_values)-1:
                #    fig.colorbar(c, ax=ax[i,j])

        plt.savefig(f'LLP_geometric_acc_Rap_Pt_R={R:0.0f}.pdf', bbox_inches='tight')
        plt.close()
