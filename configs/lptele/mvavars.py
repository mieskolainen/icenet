# Generator level variables
KINEMATIC_GEN_VARS = [
]

# For plots, diagnostics etc.
KINEMATIC_VARS = [
  'trk_pt',
  'trk_eta',
  'trk_phi',
]

# Use these to test that the (re-weight trained) classifier
# is independent of these
UNIT_TEST_ID = [
  'trk_pt',
  'trk_eta'
]

CMSSW_MVA_SCALAR_VARS = [
#  'gsf_bdtout1',
#  'eid_rho',
#  'eid_ele_pt',
#  'eid_sc_eta',
#  'eid_shape_full5x5_sigmaIetaIeta',
#  'eid_shape_full5x5_sigmaIphiIphi',
#  'eid_shape_full5x5_circularity',
#  'eid_shape_full5x5_r9',
#  'eid_sc_etaWidth',
#  'eid_sc_phiWidth',
#  'eid_shape_full5x5_HoverE',
#  'eid_trk_nhits',
#  'eid_trk_chi2red',
#  'eid_gsf_chi2red',
#  'eid_brem_frac',
#  'eid_gsf_nhits',
#  'eid_match_SC_EoverP',#
#  'eid_match_eclu_EoverP',#
#  'eid_match_SC_dEta',
#  'eid_match_SC_dPhi',
#  'eid_match_seed_dEta',
#  'eid_sc_E',
#  'eid_trk_p',
  'ele_mva_value_depth15',#
  'trk_pt',
  'trk_eta',
]
