# Fast root file processor tests
# 
# m.mieskolainen@imperial.ac.uk, 2022

import sys
import pprint
import pickle
import uproot
import time
import awkward as ak

sys.path.append(".")
from icenet.tools import iceroot
from icenet.tools import io

path     = '/home/user/travis-stash/input/icedqcd'
datasets = 'HiddenValley_vector_m_10_ctau_10_xiO_1_xiL_1_privateMC_11X_NANOAODSIM_v2_generationForBParking/output_*.root'
key      = 'Events;1'

ids = ['nsv', 'sv_.*']
#ids = ['nsv', 'sv_deltaR', 'sv_mass']
#ids = ['sv_ptrel', 'sv_deta', 'sv_dphi', 'sv_deltaR', 'sv_mass', 'sv_chi2']
#ids = None

entry_stop = 2010
rootfile   = io.glob_expand_files(datasets=datasets, datapath=path)

for library in ['ak']:

	t       = time.time()
	out     = iceroot.load_tree(rootfile=rootfile, tree=key, entry_stop=entry_stop, ids=ids, library=library)
	elapsed = time.time() - t

	print(f'Opening with <{library}> took: {elapsed} sec')
	pprint.pprint(out)
	print(len(out))

	N = 5


	def loop(X):
		for i in range(N):
			nsv = X[i].nsv
			print(f'\n[Event {i}]:')
			print(f'ak:')
			print(f'{X[i]}')

			#print(f'list:')
			#print(f'{X_list[i]}')

			for j in range(nsv):
				print(f'sv[{j}].mass:')
				print(f'{X[i].sv[j].mass}')
				#print(f'{X_list[i]}')

	##
	print(f'Testing ak under different representations')
	X    = out[:N]
	#X_list  = out[:N].tolist()

	loop(X)

	print(f'---------------------------------------------------')
	print(f'Testing nested object property based selection (not correct)')
	X.sv = X.sv[X.sv['mass'] >= 2.0]
	
	loop(X)

