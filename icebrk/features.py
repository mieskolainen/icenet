# Features in use in B/RK analyzer (protocode)
# 
# NOTE: use convention such that new features (added here) are always with underscore '_featurename' !
#
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk

import uproot
import numpy as np


# Top level features available in b-parking nanoAOD trees
mva_features = {
    
    # Electron 1 candidate
    'BToKEE_fit_l1_eta'  : None,             # Track eta
    'BToKEE_fit_l1_phi'  : None,             # Track phi

    # Electron 2 candidate
    'BToKEE_fit_l2_eta'  : None,             # Track eta
    'BToKEE_fit_l2_phi'  : None,             # Track phi

    # Kaon candidate
    'BToKEE_fit_k_eta'   : None,             # Track eta
    'BToKEE_fit_k_phi'   : None,             # Track phi

    # Triplet / Secondary vertex variables
    'BToKEE_svprob'      : None,             # Secondary vertex fit chi2 p-value
    'BToKEE_fit_cos2D'   : None,             # Cosine angle in the xy-plane between the B momentum and the separation between the B vertex and the beamspot
    'BToKEE_minDR'       : None,             # Minimum dR among 3 tracks
    'BToKEE_maxDR'       : None,             # Maximum dR among 3 tracks
    'BToKEE_charge'      : None,             # Charge
}


# Electron features indexed with [BToKEE_l1Idx], [BToKEE_l2Idx]
#
eid_features = {

    '_BToKEE_l1_unBiased' : None,            # Electron MVA output (Note: special value 20, then pfmvaId contains value, and vice versa)
    '_BToKEE_l1_pfmvaId'  : None,            # <--> (as with unBiased)

    '_BToKEE_l2_unBiased' : None,            # --|--
    '_BToKEE_l2_pfmvaId'  : None,            # --|--
}


# Electron features (ratios) indexed with [BToKEE_l1Idx] and [BToKEE_l2Idx]
e_r_features = {

    '_BToKEE_l1_dxy_sig'  : ('Electron_dxy', 'Electron_dxyErr', None),  # impact parameter (xy-plane) significance w.r.t first PV
    '_BToKEE_l2_dxy_sig'  : ('Electron_dxy', 'Electron_dxyErr', None)   # impact parameter (xy-plane) significance w.r.t first PV
}


# Kaon features indexed with [BToKEE_kIdx]
k_features = {
    '_BToKEE_k_DCA_sig'   : ('ProbeTracks_DCASig', None)                # kaon candidate impact parameter (xy-plane) significance w.r.t beamspot
}


# Ratio features
r_features = {
      
    # (key name, numerator, denominator)

    # Electron 1
    '_BToKEE_fit_l1_normpt' : ('BToKEE_fit_l1_pt',  'BToKEE_fit_mass',  None),  # pT(l1)/m(B)
    '_BToKEE_l1_iso04_rel'  : ('BToKEE_l1_iso04',   'BToKEE_fit_pt',    None),  # relative track isolation ([sum pT(track) ] / pT(B)) around B cone dR < 0.04

    # Electron 2
    '_BToKEE_fit_l2_normpt' : ('BToKEE_fit_l2_pt',  'BToKEE_fit_mass',  None),  # -|-
    '_BToKEE_l2_iso04_rel'  : ('BToKEE_l2_iso04',   'BToKEE_fit_pt',    None),  # -|-

    # Kaon
    '_BToKEE_fit_k_normpt'  : ('BToKEE_fit_k_pt',   'BToKEE_fit_mass',  None),  # -|-
    '_BToKEE_k_iso04_rel'   : ('BToKEE_k_iso04',    'BToKEE_fit_k_pt',  None),  # -|-

    # System / Secondary vertex
    '_BToKEE_fit_normpt'    : ('BToKEE_fit_pt',     'BToKEE_fit_mass',  None),  # pT(B)/m(B)
    '_BToKEE_l_xy_sig'      : ('BToKEE_l_xy',       'BToKEE_l_xy_unc',  None),  # Secondary vertex displacement significance
}


# Difference features
d_features = {
    '_BToKEE_dz'            : ('BToKEE_vtx_z',      'PV_z',             None)   # Triplet associated secondary vertex z-axis delta w.r.t. to the first PV
}


# Add all features together
#
#
all_features = dict()
all_features.update(mva_features)
all_features.update(eid_features)
all_features.update(e_r_features)
all_features.update(k_features)
all_features.update(r_features)
all_features.update(d_features)


def getdimension():
    """ Count the number of features per input triplet.
    """
    return len(mva_features) + len(eid_features) + len(e_r_features) + len(k_features) + len(r_features) + len(d_features)


def construct_new_branches(d):
    """Construct new feature branches.

    Args:
        d:
    Returns:

    """

    EPS = 1e-12 # division by zero protection

    # eid-features
    d['_BToKEE_l1_unBiased'] = d['Electron_unBiased'][d['BToKEE_l1Idx']]
    d['_BToKEE_l1_pfmvaId']  = d['Electron_pfmvaId'][d['BToKEE_l1Idx']]

    d['_BToKEE_l2_unBiased'] = d['Electron_unBiased'][d['BToKEE_l2Idx']]
    d['_BToKEE_l2_pfmvaId']  = d['Electron_pfmvaId'][d['BToKEE_l2Idx']]

    # e-r-features
    for name in e_r_features.keys():
        idkey = 'BToKEE_l1Idx' if ('l1' in name) else 'BToKEE_l2Idx'
        d[name] = d[e_r_features[name][0] ][d[idkey]] / (d[e_r_features[name][1] ][d[idkey]] + EPS)
    # k-features
    for name in k_features.keys():
        idkey = 'BToKEE_kIdx'
        d[name] = d[k_features[name][0] ][ d[idkey] ]

    # r-features
    for name in r_features.keys():
        d[name] = d[r_features[name][0]] / (d[r_features[name][1]] + EPS)
    # d-features
    for name in d_features.keys():
        d[name] = np.abs( d[d_features[name][0]] - d[d_features[name][1]] )


# Generate variable names
# parameter N generates them blockwise multiple times
# (for the multitriplet scheme)
#
def generate_feature_names(N=1):
    
    names = []
    for key in all_features.keys():
        for i in range(N):
            names.append(f'{key}[{i}]')
    return names
