# B/RK analyzer observables and histograms
# 
# 
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk


import bz2
import numpy as np
import iceplot

import icebrk.tools as tools


obs_M = {

# Axis limits
'xlim'    : (4.5, 6.5),
'ylim'    : None,
'xlabel'  : r'System $M$',
'ylabel'  : r'Candidates',
'units'   : r'GeV',
'label'   : r'3-body system invariant mass',
'figsize' : (4,4),

# Histogramming
'bins'    : np.arange(0.0, 10.0 + 0.1, 0.1),
'density' : False,

# Function to calculate
'func'    : None,

# Disk save
'pickle'  : True
}

obs_Pt = {

# Axis limits
'xlim'    : (0, 25.0),
'ylim'    : None,
'xlabel'  : r'System $P_t$',
'ylabel'  : r'Candidates',
'units'   : r'GeV',
'label'   : r'System tranverse momentum',
'figsize' : (4,4),

# Histogramming
'bins'    : np.arange(0.0, 25.0 + 1.0, 1.0),
'density' : False,
 
# Function to calculate
'func'    : None,

# Disk save
'pickle'  : True
}

obs_q2 = {

# Axis limits
'xlim'    : (0, 12.0),
'ylim'    : None,
'xlabel'  : r'Electron pair $q^2$',
'ylabel'  : r'Candidates',
'units'   : r'GeV$^2$',
'label'   : r'Electron system invariant squared',
'figsize' : (4,4),

# Histogramming
'bins'    : np.arange(0, 12 + 0.4, 0.4),
'density' : False,
 
# Function to calculate
'func'    : None,

# Disk save
'pickle'  : True
}

obs_pt_l1 = {

# Axis limits
'xlim'    : (0, 12.0),
'ylim'    : None,
'xlabel'  : r'Leading electron $p_t$',
'ylabel'  : r'Candidates',
'units'   : r'GeV',
'label'   : r'Leading $e$ transverse momentum',
'figsize' : (4,4),

# Histogramming
'bins'    : np.arange(0, 12 + 0.5, 0.5),
'density' : False,
 
# Function to calculate
'func'    : None,

# Disk save
'pickle'  : False
}

obs_pt_l2 = {

# Axis limits
'xlim'    : (0, 12.0),
'ylim'    : None,
'xlabel'  : r'Sub-leading electron $p_t$',
'ylabel'  : r'Candidates',
'units'   : r'GeV',
'label'   : r'Sub-leading transverse momentum',
'figsize' : (4,4),

# Histogramming
'bins'    : np.arange(0, 12 + 0.5, 0.5),
'density' : False,
 
# Function to calculate
'func'    : None,

# Disk save
'pickle'  : False
}

obs_pt_k = {

# Axis limits
'xlim'    : (0, 12.0),
'ylim'    : None,
'xlabel'  : r'Kaon $p_t$',
'ylabel'  : r'Candidates',
'units'   : r'GeV',
'label'   : r'Kaon transverse momentum',
'figsize' : (4,4),

# Histogramming
'bins'    : np.arange(0, 12 + 0.5, 0.5),
'density' : False,
 
# Function to calculate
'func'    : None,

# Disk save
'pickle'  : False
}

# ** MC ONLY **
obs_first_t3i = {

# Axis limits
'xlim'    : (-1, 20),
'ylim'    : None,
'xlabel'  : r'First signal triplet',
'ylabel'  : r'Events',
'units'   : r'MC index',
'label'   : r'First signal triplet',
'figsize' : (4,4),

# Histogramming
'bins'    : np.arange(-1, 20 + 1, 1),
'density' : False,
 
# Function to calculate
'func'    : None,

# Disk save
'pickle'  : False
}

obs_last_t3i = {

# Axis limits
'xlim'    : (-1, 20),
'ylim'    : None,
'xlabel'  : r'Last signal triplet',
'ylabel'  : r'Events',
'units'   : r'MC index',
'label'   : r'Last signal triplet',
'figsize' : (4,4),

# Histogramming
'bins'    : np.arange(-1, 20 + 0.5, 0.5),
'density' : False,
 
# Function to calculate
'func'    : None,

# Disk save
'pickle'  : False
}

obs_N_signal_t3 = {

# Axis limits
'xlim'    : (-1, 20),
'ylim'    : None,
'xlabel'  : r'Number of signal triplets',
'ylabel'  : r'Events',
'units'   : r'',
'label'   : r'Number of signal triplets',
'figsize' : (4,4),

# Histogramming
'bins'    : np.arange(-1, 20 + 0.5, 0.5),
'density' : False,
 
# Function to calculate
'func'    : None,

# Disk save
'pickle'  : False
}


# Dictionary of all batch observables
obs_all = {

	# JAGGED
	'M'           : obs_M,
	'Pt'          : obs_Pt,
	'q2'          : obs_q2,
	'pt_l1'       : obs_pt_l1,
	'pt_l2'       : obs_pt_l2,
	'pt_k'        : obs_pt_k,

	# NORMAL
	'first_t3i'   : obs_first_t3i,
	'last_t3i'    : obs_last_t3i,
	'N_signal_t3' : obs_N_signal_t3,
	'N_signal_pfpf_t3'   : obs_N_signal_t3,
	'N_signal_lowlow_t3' : obs_N_signal_t3
}


def calc_batch_observables(l1_p4, l2_p4, k_p4):
	"""JAGGED + VECTORIZED (operates on event batch) observables.

    Args:
    	l1_p4:
    	l2_p4:
    	k_p4:

    Returns:
        x: Observables
    """

	x = {
		'M'     : None,
		'Pt'    : None,
		'q2'    : None,
		'pt_l1' : None,
		'pt_l2' : None,
		'pt_k'  : None
	}
	x['M']     = (l1_p4['e'] + l2_p4['e'] + k_p4['k']).mass
	x['Pt']    = (l1_p4['e'] + l2_p4['e'] + k_p4['k']).pt
	x['q2']    = (l1_p4['e'] + l2_p4['e']).mass2
	x['pt_l1'] =  l1_p4['e'].pt
	x['pt_l2'] =  l2_p4['e'].pt
	x['pt_k']  =   k_p4['k'].pt
	
	return x


def calc_batch_MC_observables(d, l1_p4, l2_p4, k_p4):
	""" MC ONLY batch observables.
	
	Args:
		d: 
		l1_p4:
		l2_p4:
		k_p4:
	
	Returns:
		x
	"""	
	x = {
	}
	return x


def calc_observables(evt_index, d, l1_p4, l2_p4, k_p4, sets, MAXT3):
	"""NON-JAGGED (NORMAL) observables.

    Args:
    	l1_p4:
    	l2_p4:
    	k_p4:

    Returns:
        x: Observables
    """

	x = {
	}
	return x


vals = {
	'init' : False,
}
vals_pfpf = {
	'init' : False,
}
vals_lowlow = {
	'init' : False,
}


def calc_MC_observables(evt_index, d, l1_p4, l2_p4, k_p4, sets, MAXT3):
	"""MC ONLY observables.

    Args:
    	evt_index:
    	d:
    	l1_p4:
    	l2_p4:
    	k_p4:
    	sets:
    	MAXT3:

    Returns:
        x: Observables
    """

	if vals['init'] == False:
		vals['init'] = True
		for i in range(100):
			vals[str(i)] = 0

	if vals_pfpf['init'] == False:
		vals_pfpf['init'] = True
		for i in range(100):
			vals_pfpf[str(i)] = 0

	if vals_lowlow['init'] == False:
		vals_lowlow['init'] = True
		for i in range(100):
			vals_lowlow[str(i)] = 0

	x = {
		'first_t3i'          : None,
		'last_t3i'           : None,
		'N_signal_t3'        : None,
		'N_signal_pfpf_t3'   : None,
		'N_signal_lowlow_t3' : None	
	}

	# Number of signal triplets
	x['N_signal_t3'] 		= np.sum(d['_BToKEE_is_signal'][evt_index])
	x['N_signal_lowlow_t3'] = np.sum(d['_BToKEE_is_signal'][evt_index] & d['Electron_isLowPt'][d['BToKEE_l1Idx']][evt_index] & d['Electron_isLowPt'][d['BToKEE_l2Idx']][evt_index])
	x['N_signal_pfpf_t3'] 	= np.sum(d['_BToKEE_is_signal'][evt_index] & d['Electron_isPF'][d['BToKEE_l1Idx']][evt_index] 	 & d['Electron_isPF'][d['BToKEE_l2Idx']][evt_index])


	vals[str(x['N_signal_t3'])] 	          += 1
	vals_lowlow[str(x['N_signal_lowlow_t3'])] += 1	
	vals_pfpf[str(x['N_signal_pfpf_t3'])]     += 1

	# The first signal index
	x['first_t3i']   = tools.index_of_first_signal(evt_index, d, sets, MAXT3)

	# The last signal index
	x['last_t3i']    = tools.index_of_last_signal(evt_index, d, sets, MAXT3)

	#print(list(vals.values())[0:22])
	#print(list(vals_lowlow.values())[0:22])
	#print(list(vals_pfpf.values())[0:22])

	return x


def pickle_files(iodir, N_algo, label, mode='rb'):
	"""Open pickle files.

    Args:
    	iodir:
    	N_algo:
    	label:
    	mode: mode = 'rb' (read binary), 'ab' (append binary), 'wb' (write binary)

    Returns:
        x: Observables
    """

	wfile   = {'S': dict(), 'B': dict()}
	for ID in wfile.keys():
		for i in range(N_algo):
			wfile[ID][str(i)] = bz2.BZ2File(iodir + f'/{label}_{ID}_weights_{i}.bz2', mode)

	obsfile = {'S': dict(), 'B': dict()}
	for ID in obsfile.keys():
		for obs in obs_all.keys():
			if obs_all[obs]['pickle']:
				obsfile[ID][obs] = bz2.BZ2File(iodir + f'/{label}_{ID}_obs_{obs}.bz2', mode)

	return obsfile, wfile