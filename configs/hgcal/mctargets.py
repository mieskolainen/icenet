# MC only. Supervised training signal class target definitions.
#
# m.mieskolainen@imperial.ac.uk, 2021

import numpy as np
import numba


#def target_e(events, entry_start=0, entry_stop=None, new=False):
#    """ Classification signal target definition"""
#    if new:
#    	return events.arrays("is_e", library="np", how=list, entry_start=entry_start, entry_stop=entry_stop)
#    return events.array("is_e", entry_start=entry_start, entry_stop=entry_stop)
#
#def target_egamma(events, entry_start=0, entry_stop=None, new=False):
#    """ Classification signal target definition"""
#    if new:
#    	return events.arrays("is_egamma", library="np", how=list, entry_start=entry_start, entry_stop=entry_stop)
#    return events.array("is_egamma", entry_start=entry_start, entry_stop=entry_stop)

# Add alternatives here
# ...