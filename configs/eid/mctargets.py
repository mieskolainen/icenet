# MC only. Supervised training signal class target definitions.
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import numpy as np
import numba


def target_e(events, entrystart=0, entrystop=None):
    """ Classification signal target definition"""
    return events.arrays("is_e", library="np", how=list, entry_start=entrystart, entry_stop=entrystop)

def target_egamma(events, entrystart=0, entrystop=None):
    """ Classification signal target definition"""
    return events.arrays("is_egamma", library="np", how=list, entry_start=entrystart, entry_stop=entrystop)


# Add alternatives here
# ...