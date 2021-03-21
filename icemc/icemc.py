# Simple Monte Carlo routines
#
# Mikael Mieskolainen, 2021
# m.mieskolainen@imperial.ac.uk

import numpy as np
import numba

from icenet.tools import icevec


@numba.njit
def U(a,b):
    """ Uniform random numbers from [a,b]
    """
    return a + (b-a)*np.random.rand()


#@numba.njit
def randpow(a, b, g, size=1):
    """
    Power-law random numbers for pdf(x) ~ x^{g-1} for a <= x <= b
    """
    r = np.random.random(size=size)
    ag, bg = a**g, b**g
    
    return (ag + (bg - ag)*r)**(1.0/g)


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

