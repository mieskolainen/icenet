# Fast root file processor tests
# 
# m.mieskolainen@imperial.ac.uk, 2021

import sys
sys.path.append(".")
from icenet.tools import iceroot

rootfile = '/home/user/cernbox/HLT_electrons/efftest.root'
cuts     = 'gen_e1_l1_dr < 0.2 AND gen_e2_l1_dr < 0.2 AND e1_l1_pt >=6 AND e2_l1_pt >= 6'

for key in ['tree;1', 'tree;2']:

	print(key)
	X = iceroot.load_tree(rootfile=rootfile, tree=key, entry_stop=None)
	#print(X.ids)

	y1 = X[cuts + " AND gen_e1_hlt_dr < 0.2"]
	y2 = X[cuts + " AND gen_e2_hlt_dr < 0.2"]
	
	print(f'{X.x.shape[0]} {y1.shape[0]} {y1.shape[0] / X.x.shape[0]:0.3E}')
	print(f'{X.x.shape[0]} {y2.shape[0]} {y2.shape[0] / X.x.shape[0]:0.3E}')

	print('')

