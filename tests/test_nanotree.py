# Fast root file processor tests
# 
# m.mieskolainen@imperial.ac.uk, 2021

import sys
import pprint
import pickle

sys.path.append(".")
from icenet.tools import iceroot
import icenet.tools.icemap as icemap

rootfile = 'dev_nano.root'
#cuts     = 'gen_e1_l1_dr < 0.2 AND gen_e2_l1_dr < 0.2 AND e1_l1_pt >=6 AND e2_l1_pt >= 6 AND gen_e1_hlt_dr < 0.2'

ids = ['sv_ptrel', 'sv_deta', 'sv_dphi', 'sv_deltaR', 'sv_mass', 'sv_chi2']
ids = None

for key in ['Events;1']:
	
	print(key)
	X = iceroot.load_tree(rootfile=rootfile, tree=key, entry_start=0, entry_stop=10, ids=ids)
	
	pprint.pprint(X.ids)
