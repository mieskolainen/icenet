# Cuts and statistics objects for B/RK analyzer (protocode)
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk


import numpy as np
import icebrk.tools as tools


def apply_cuts(d, evt_index, cutflow):
	"""Selection cuts function.

    Args:
        d:
        evt_index:
        cutflow:

    Returns:
        True or False
	
    """

	'''
	if (d['nSV'][evt_index] != 1):
		cutflow['nSV != 2'] += 1
		return False
	'''
	
	return True


def triplet_cuts(d, evt_index, cutflow):
	""" Triplet cuts """

	return True


def collect_info_stats(d, evt_index, infostats):
	"""Collect event information.

    Args:
        d:
        evt_index:
        infostats:
    """

	# Set event info
	if   (d['nSV'][evt_index] == 0): infostats['nSV == 0'] += 1
	elif (d['nSV'][evt_index] == 1): infostats['nSV == 1'] += 1
	elif (d['nSV'][evt_index] == 2): infostats['nSV == 2'] += 1
	elif (d['nSV'][evt_index] == 3): infostats['nSV == 3'] += 1
	elif (d['nSV'][evt_index] >= 4): infostats['nSV >= 4'] += 1

	return


def collect_mcinfo_stats(d, evt_index, y, qsets, MAXT3, mcinfostats):
	"""Collect MC only information.
	
    Args:

    Returns:
	
    """

	index = tools.index_of_last_signal(evt_index, d, qsets, MAXT3)
	if (index >= MAXT3): mcinfostats['last index >= MAXT3'] += 1
	
	return


def init_stat_objects():
	"""Initialize cutflow and statistics objects.

    Args:

    Returns:
        cutflow:
        infostats_BC:
        mcinfostats_BC:
	
    """
	cutflow        = {
	}
	infostats_BC   = {
		'nSV == 0' : 0,
		'nSV == 1' : 0,
		'nSV == 2' : 0,
		'nSV == 3' : 0,
		'nSV >= 4' : 0
	}
	mcinfostats_BC = {
		'last index >= MAXT3' : 0
	}
	return cutflow, infostats_BC, mcinfostats_BC

