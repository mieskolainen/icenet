# Fast root file processor tests
# 
# m.mieskolainen@imperial.ac.uk, 2021

import sys
sys.path.append(".")
from icenet.tools import iceroot

rootfile = '/home/user/cernbox/HLT_electrons/efftest.root'

X = iceroot.load_tree(rootfile=rootfile, tree='tree', entry_stop=None)

cuts = 'gen_e1_l1_dr < 0.2 AND gen_e2_l1_dr < 0.2 AND gen_e1_hlt_dr < 0.2 AND e1_l1_pt >=6 AND e2_l1_pt >= 6'
y = X[cuts]

print(X.ids)
print(f'{X.x.shape[0]} {y.shape[0]} {y.shape[0] / X.x.shape[0]:0.3E}')
