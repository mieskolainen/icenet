# Fast root file processor tests
# 
# m.mieskolainen@imperial.ac.uk, 2021

import sys
sys.path.append(".")
from icenet.tools import iceroot
from icenet.tools.icemap import icemap
import numpy as np

rootfile = '/home/user/cernbox/HLT_electrons/efftest.root'
cuts     = 'gen_e1_l1_dr < 0.2 AND gen_e2_l1_dr < 0.2 AND e1_l1_pt >=6 AND e2_l1_pt >= 6 AND gen_e1_hlt_dr < 0.2'

for key in ['tree;1', 'tree;2']:
	
	print(key)
	Y = iceroot.load_tree(rootfile=rootfile, tree=key)
	X = icemap(Y)

	y1 = X[cuts]
	print(f'before: {X.x.shape[0]}, after: {y1.x.shape[0]}, ratio: {y1.x.shape[0] / X.x.shape[0]:0.3E}')
	print('')
	
	y2 = X[cuts + ' AND gen_e2_hlt_dr < 0.2']
	print(f'before: {X.x.shape[0]}, after: {y2.x.shape[0]}, ratio: {y2.x.shape[0] / X.x.shape[0]:0.3E}')
	print('')
	

Y    = iceroot.load_tree(rootfile=rootfile, tree="tree")
X    = icemap(Y) 
x    = X[cuts]
pms2 = x['e1_hlt_pms2']

for i in range(len(pms2)):
	print(f'{pms2[i]:.5f}')

