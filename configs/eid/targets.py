# Supervised training signal class target definitions (only available in MC)
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import numpy as np
import numba


def target_standard(events):
    """ Classification signal target definition"""
    return events.array("is_e") & np.logical_not(events.array("is_egamma"))


def target_e(events):
    """ Classification signal target definition"""
    return events.array("is_e")


def target_egamma(events):
    """ Classification signal target definition"""
    return events.array("is_egamma")


# Add alternatives here
# ...