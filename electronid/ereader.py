# Electron ID data reader and cuts
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk


import os
import uproot
import numpy as np

from matplotlib import pyplot as plt

# Own
from icenet.tools import io
from icenet.tools import aux
from icenet.tools import iceplots


# Function implements basic selections (cuts)
#
#
def func_cuts(X, VARS):

  ind = np.ones((X.shape[0]), dtype = bool)
  ind = np.logical_and(ind, X[:, VARS.index('is_egamma')] == False) # generator level origin is electron
  ind = np.logical_and(ind, X[:, VARS.index('gsf_pt')] > 0)         #

  return ind


# Compute (eta,pt) reweighting coefficients
#
#
def compute_reweights(data, args):

  print(__name__ + '.compute_reweights: Computing re-weighting coefficients ...')

  PT           = data.trn.x[:,data.VARS.index('trk_pt')]
  ETA          = data.trn.x[:,data.VARS.index('trk_eta')]
  
  pt_binedges  = np.linspace(args['reweight_param']['bins_pt'][0],
                             args['reweight_param']['bins_pt'][1],
                             args['reweight_param']['bins_pt'][2])

  eta_binedges = np.linspace(args['reweight_param']['bins_eta'][0],
                             args['reweight_param']['bins_eta'][1],
                             args['reweight_param']['bins_eta'][2])

  # Full 2D  
  trn_weights = aux.reweightcoeff2D(PT, ETA, data.trn.y, pt_binedges, eta_binedges,
    shape_reference = args['reweight_param']['mode'], max_reg = args['reweight_param']['max_reg'])


  ### Plot some kinematic variables
  targetdir = './figs/{}/train/1D_kinematic/'.format(args['config'])
  os.makedirs(targetdir, exist_ok = True)

  TARGETS = ['trk_pt', 'trk_eta', 'trk_phi', 'trk_p']
  for k in TARGETS:
    iceplots.plotvar(x = data.trn.x[:, data.VARS.index(k)], y = data.trn.y, weights = trn_weights, var = k, NBINS = 70,
      targetdir = targetdir, title = 'training reweight reference: {}'.format(args['reweight_param']['mode']))

  return trn_weights



# Loads the root file
#
#
def load_root_file(root_path, class_id = []):

    ### From root trees
    print('\n')
    print( __name__ + '.load_root_file: Loading from file ' + root_path)
    file = uproot.open(root_path)
    events = file["ntuplizer"]["tree"]

    print(events.name)
    print(events.title)
    print(__name__ + '.load_root_file: events.numentries = {}'.format(events.numentries))

    ### First load all data
    VARS   = [x.decode() for x in events.keys()]
    X_dict = events.arrays(VARS, namedecode = "utf-8")
    Y      = events.array("is_e")

    ### Convert input to matrix
    X = np.array([X_dict[j] for j in VARS])
    X = np.transpose(X)

    # -----------------------------------------------------------------
    # @@ Selections done here @@

    ind = func_cuts(X, VARS)

    # -----------------------------------------------------------------

    N_before = X.shape[0]
    print(__name__ + ".load_root_file: Prior selections: {} events ".format(N_before))

    ### Select events
    X    = X[ind,:]
    Y    = Y[ind]

    N_after = X.shape[0]
    print(__name__ + ".load_root_file: Post  selections: {} events ({:.3f})".format(N_after, N_after / N_before) )
    
    # -----------------------------------------------------------------
    # @@ Data inputation done here @@

    # Data inputation
    z = [-999, -666, -10] # possible special values
    X = io.inpute_data(X, z, VARS)

    return X, Y, VARS
