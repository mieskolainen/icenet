# Test uproot
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
#from icenet.tools import iceroot
#from icenet.tools import io

## Write MVA-scores out

#basepath   = f'{cwd}/output/deploy' + '/' + f"modeltag-[{args['modeltag']}]"
#outpath    = aux.makedir(basepath + '/' + filename.rsplit('/', 1)[0])
#outputfile = basepath + '/' + filename.replace('.root', '-icenet.root')


def cartesian_product(*arrays):
    """
    N-dimensional generalized cartesian product between arrays
    
    Args:
        *arrays: a list of arrays
    
    Example:
        cartesian_product(*[values['m'], values['ctau']])
    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)


### Set the conditional variables
values = {'m':    np.round(np.linspace(2,   25, 11), 1),
          'ctau': np.round(np.linspace(10, 500, 11), 1),
          'xiO':  np.round(np.array([1.0]), 1),
          'xiL':  np.round(np.array([1.0]), 1)}


for var in values.keys():
    outs = cartesian_product(*[values['m'], values['ctau'], values['xiO'], values['xiL']])

print(outs)

scores     = {'m_5.0_ctau_100.0': np.random.randn(10000), 'm_5.0_ctau_500.0': np.random.rand(10000)}
outputfile = 'dev_test.root'


with uproot.recreate(outputfile, compression=None) as file:
    print(__name__ + f'.process_data: Saving root output to "{outputfile}"')
    file[f"tree1"] = {f"xgb0": scores}


