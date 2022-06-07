# Test jagged awkward array to a fixed dimensional matrix conversion
# 
# m.mieskolainen@imperial.ac.uk, 2022

import uproot
import awkward
import copy
import numpy as np
import numba

import sys
import pprint
import pickle

sys.path.append(".")
from icenet.tools import iceroot
from icenet.tools.aux import jagged2matrix

def test_jagged2matrix():

	path     = '/home/user/travis-stash/input/icedqcd'
	rootfile = 'HiddenValley_vector_m_10_ctau_10_xiO_1_xiL_1_privateMC_11X_NANOAODSIM_v2_generationForBParking/output_100.root'
	key      = 'Events;1'

	ids      = ['nsv', 'nmuon', 'sv_ptrel', 'sv_deta', 'sv_dphi', 'sv_deltaR', 'sv_mass', 'sv_chi2']
	#ids = None

	library  = 'ak'
	#library = 'np'

	X = iceroot.load_tree(rootfile=f'{path}/{rootfile}', tree=key, max_num_elements=1000, ids=ids, library=library)
	#pprint.pprint(X.keys())
	#print(len(X))
	#print(X['sv_deta'])
	mat = jagged2matrix(X, scalar_vars=['nsv', 'nmuon'], jagged_vars=['sv_deta', 'sv_dphi', 'sv_mass'], jagged_maxdim=[5, 5, 5])

	print(mat)
