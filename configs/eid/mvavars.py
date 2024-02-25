# Generator level variables
KINEMATIC_GEN_VARS = [
]

# For plots, diagnostics etc.
KINEMATIC_VARS = [
  'trk_pt',
  'trk_eta',
  'trk_phi',
  'ele_mva_value_depth15'
]

# Use these to test that the (re-weight trained) classifier
# is independent of these
UNIT_TEST_ID = [
  'trk_pt',
  'trk_eta'
]

CMSSW_MVA_SCALAR_VARS = [
  'gsf_bdtout1',
  'eid_rho',
  'eid_ele_pt',
  'eid_sc_eta',
  'eid_shape_full5x5_sigmaIetaIeta',
  'eid_shape_full5x5_sigmaIphiIphi',
  'eid_shape_full5x5_circularity',
  'eid_shape_full5x5_r9',
  'eid_sc_etaWidth',
  'eid_sc_phiWidth',
  'eid_shape_full5x5_HoverE',
  'eid_trk_nhits',
  'eid_trk_chi2red',
  'eid_gsf_chi2red',
  'eid_brem_frac',
  'eid_gsf_nhits',
  'eid_match_SC_dEta',
  'eid_match_SC_dPhi',
  'eid_match_seed_dEta',
  'eid_sc_E',
  'eid_trk_p',
]

CMSSW_MVA_IMAGE_VARS = [
 'image_gsf_ref_eta',
 'image_gsf_ref_phi',
 'image_gsf_ref_R',
 'image_gsf_ref_p',
 'image_gsf_ref_pt',
 'image_gen_inner_eta',
 'image_gen_inner_phi',
 'image_gen_inner_R',
 'image_gen_inner_p',
 'image_gen_inner_pt',
 'image_gsf_inner_eta',
 'image_gsf_inner_phi',
 'image_gsf_inner_R',
 'image_gsf_inner_p',
 'image_gsf_inner_pt',
 'image_gsf_charge',
 'image_gsf_proj_eta',
 'image_gsf_proj_phi',
 'image_gsf_proj_R',
 'image_gsf_proj_p',
 'image_gsf_atcalo_eta',
 'image_gsf_atcalo_phi',
 'image_gsf_atcalo_R',
 'image_gsf_atcalo_p',
 'image_clu_n',
 'image_clu_eta',
 'image_clu_phi',
 'image_clu_e',
 'image_clu_nhit',
 'image_pf_n',
 'image_pf_eta',
 'image_pf_phi',
 'image_pf_p',
 'image_pf_pdgid',
 'image_pf_matched',
 'image_pf_lost'
]

CMSSW_MVA_SCALAR_VARS_ORIG = [
  'gsf_bdtout1',
  'eid_rho',
  'eid_ele_pt',
  'eid_sc_eta',
  'eid_shape_full5x5_sigmaIetaIeta',
  'eid_shape_full5x5_sigmaIphiIphi',
  'eid_shape_full5x5_circularity',
  'eid_shape_full5x5_r9',
  'eid_sc_etaWidth',
  'eid_sc_phiWidth',
  'eid_shape_full5x5_HoverE',
  'eid_trk_nhits',
  'eid_trk_chi2red',
  'eid_gsf_chi2red',
  'eid_brem_frac',
  'eid_gsf_nhits',
  'eid_match_SC_EoverP',
  'eid_match_eclu_EoverP',
  'eid_match_SC_dEta',
  'eid_match_SC_dPhi',
  'eid_match_seed_dEta',
  'eid_sc_E',
  'eid_trk_p',
]
