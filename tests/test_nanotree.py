# Fast root file processor tests
# 
# m.mieskolainen@imperial.ac.uk, 2022

import sys
import pprint
import pickle
import uproot

sys.path.append(".")
from icenet.tools import iceroot

path     = '/home/user/travis-stash/input/icedqcd'
rootfile = 'HiddenValley_vector_m_10_ctau_10_xiO_1_xiL_1_privateMC_11X_NANOAODSIM_v2_generationForBParking/output_100.root'
key      = 'Events;1'

ids = ['sv_ptrel', 'sv_deta', 'sv_dphi', 'sv_deltaR', 'sv_mass', 'sv_chi2']
#ids = None

#library = 'ak'
library = 'np'

X = iceroot.load_tree(rootfile=f'{path}/{rootfile}', tree=key, max_num_elements=3, ids=ids, library=library)

pprint.pprint(X)
print(len(X))

