# Fast root file processor tests
# 
# m.mieskolainen@imperial.ac.uk, 2022

import sys
import pprint
import pickle
import uproot
import time
import awkward as ak
import numpy as np

sys.path.append(".")
from icenet.tools import iceroot
from icenet.tools import io

path     = '/home/user/travis-stash/input/icedqcd'
datasets = 'HiddenValley_vector_m_10_ctau_10_xiO_1_xiL_1_privateMC_11X_NANOAODSIM_v2_generationForBParking/output_*.root'
key      = 'Events;1'

ids = ['nsv', 'sv_.*', 'cpf_.*']

#ids = ['nsv', 'sv_deltaR', 'sv_mass']
#ids = ['sv_ptrel', 'sv_deta', 'sv_dphi', 'sv_deltaR', 'sv_mass', 'sv_chi2']
#ids = None

entry_stop = 2010
rootfile   = io.glob_expand_files(datasets=datasets, datapath=path)

for library in ['ak']:

	t       = time.time()
	out,ids = iceroot.load_tree(rootfile=rootfile, tree=key, entry_stop=entry_stop, ids=ids, library=library)
	elapsed = time.time() - t

	print(f'Opening with <{library}> took: {elapsed} sec')
	
	print(len(out))
	print(out)

	N = 5

	# Add in a new record
	out['__model_m'] = ak.Array(np.random.rand(len(out)))
	print(len(out))
	print(out)
	
	print(ak.fields(out))
	
	def loop(X):
		for i in range(N):
			nsv = X[i].nsv
			print(f'\n[Event {i}]:')
			print(f'ak:')
			print(f'X[{i}]: {X[i]}')
			print(f'__model_m: {X[i].__model_m}')

			jetIdx = X[i].cpf.jetIdx
			mask   = (X[i].cpf.jetIdx != -1)

			print(mask)

			# Alternative indexing
			print(X[i]['cpf']['jetIdx'])
			
			p = X[i].cpf[mask]
			print(f'  px: {p.px} py: {p.py} pz: {p.pz} trackSip2dVal: {p.trackSip2dVal} trackSip3dVal: {p.trackSip3dVal}')

			# print(f'list:')
			# print(f'{X_list[i]}')

			for j in range(nsv):
				print(f'sv[{j}].mass: {X[i].sv[j].mass}')
				#print(f'{X_list[i]}')

	##
	print(f'Testing ak under different representations')

	X    = out[:N]
	print(f'{len(out)}')
	
	#X_list  = out[:N].tolist()

	loop(X)

	print(f'---------------------------------------------------')
	print(f'Testing nested object property based selection (not correct)')
	X.sv = X.sv[len(X.sv) < 2] #['mass'] >= 2.0
	
	loop(X)

