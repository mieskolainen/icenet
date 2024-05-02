# Generator level variables
KINEMATIC_GEN_VARS = [
    'gen_pt',
    'gen_eta',
    'gen_phi',
]

# Main kinematic variables
KINEMATIC_VARS = [
    'trk_pt',
    'trk_eta',
    'trk_phi',
    'gsf_pt',
    'gsf_eta',
    'gsf_phi',
    'ele_pt',
    'ele_eta',
    'ele_phi',
]

# Additional variables 
ADDITIONAL_VARS = [
    'is_e','is_mc','is_egamma','is_aod', # Labels
    'tag_pt','tag_eta',                  # Tag muon kine
    'has_trk','has_gsf','has_ele',       # Object flags
    'rho',                               # Proxy for pileup
    'ele_mva_value',                     # Existing 2019Aug07 BDT score
    'ele_mva_value_depth10',             # Existing 2020Sept15 BDT score 
]

# Use these to test that the (re-weight trained) classifier
# is independent of these
UNIT_TEST_ID = [
    'gsf_pt',
    'gsf_eta',
    'rho',
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

CMSSW_MVA_SCALAR_VARS_2019Aug07 = [
    # KF track
    '2019Aug07_trk_p',
    '2019Aug07_trk_nhits',
    '2019Aug07_trk_chi2red',
    # GSF track
    '2019Aug07_gsf_nhits',
    '2019Aug07_gsf_chi2red',
    # SC
    '2019Aug07_sc_E',
    '2019Aug07_sc_eta',
    '2019Aug07_sc_etaWidth',
    '2019Aug07_sc_phiWidth',
    # Track-cluster matching
    '2019Aug07_match_seed_dEta',
    '2019Aug07_match_eclu_EoverP',#
    '2019Aug07_match_SC_EoverP',#
    '2019Aug07_match_SC_dEta',
    '2019Aug07_match_SC_dPhi',
    # Shower shape vars
    '2019Aug07_shape_full5x5_sigmaIetaIeta',
    '2019Aug07_shape_full5x5_sigmaIphiIphi',
    '2019Aug07_shape_full5x5_HoverE',
    '2019Aug07_shape_full5x5_r9',
    '2019Aug07_shape_full5x5_circularity',
    # Misc
    '2019Aug07_rho',
    '2019Aug07_brem_frac',
    '2019Aug07_ele_pt',
    # Unbiased BDT from ElectronSeed
    '2019Aug07_gsf_bdtout1',
]

CMSSW_MVA_SCALAR_VARS_2020Sept15 = [
    # KF track
    '2020Sept15_trk_p',
    '2020Sept15_trk_nhits',
    '2020Sept15_trk_chi2red',
    '2020Sept15_trk_dr',#
    # GSF track
    '2020Sept15_gsf_nhits',
    '2020Sept15_gsf_chi2red',
    '2020Sept15_gsf_mode_p',
    '2020Sept15_gsf_dr',#
    # SC
    '2020Sept15_sc_E',
    '2020Sept15_sc_eta',
    '2020Sept15_sc_etaWidth',
    '2020Sept15_sc_phiWidth',
    '2020Sept15_sc_Nclus',#
    # Track-cluster matching
    '2020Sept15_match_seed_dEta',
    '2020Sept15_match_eclu_EoverP',
    '2020Sept15_match_SC_EoverP',
    '2020Sept15_match_SC_dEta',
    '2020Sept15_match_SC_dPhi',
    # Shower shape vars
    '2020Sept15_shape_full5x5_r9',
    '2020Sept15_shape_full5x5_HoverE',
    # Leading cluster
    '2020Sept15_sc_clus1_nxtal',
    '2020Sept15_sc_clus1_E',
    '2020Sept15_sc_clus1_E_ov_p',
    '2020Sept15_sc_clus1_deta',
    '2020Sept15_sc_clus1_dphi',
    # Sub-leading cluster
    '2020Sept15_sc_clus2_E',
    '2020Sept15_sc_clus2_E_ov_p',
    '2020Sept15_sc_clus2_dphi',
    '2020Sept15_sc_clus2_deta',
    # Misc
    '2020Sept15_rho',
    '2020Sept15_brem_frac',
    '2020Sept15_core_shFracHits',#
    # Unbiased BDT from ElectronSeed
    '2020Sept15_gsf_bdtout1',
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
    'image_pf_n',
    'image_pf_eta',
    'image_pf_phi',
    'image_pf_p',
    'image_pf_pdgid',
    'image_pf_matched',
    'image_pf_lost'
]

CMSSW_MVA_GRAPH_VARS = [
    'image_clu_eta',
    'image_clu_phi',
    'image_clu_e',
    'image_pf_eta',
    'image_pf_phi',
    'image_pf_p'
]

####################
####################
####################

# KINEMATIC_VARS are required for processing
KINEMATIC_VARS += ADDITIONAL_VARS

# CMSSW_MVA_SCALAR_VARS are features used by BDT models
CMSSW_MVA_SCALAR_VARS = []
#CMSSW_MVA_SCALAR_VARS += CMSSW_MVA_SCALAR_VARS_ORIG
CMSSW_MVA_SCALAR_VARS += CMSSW_MVA_SCALAR_VARS_2019Aug07
CMSSW_MVA_SCALAR_VARS += CMSSW_MVA_SCALAR_VARS_2020Sept15
CMSSW_MVA_SCALAR_VARS = list(set(CMSSW_MVA_SCALAR_VARS))

# LOAD_VARS are read out from ROOT files, 
LOAD_VARS  = []
#LOAD_VARS += KINEMATIC_GEN_VARS
LOAD_VARS += KINEMATIC_VARS
LOAD_VARS += CMSSW_MVA_SCALAR_VARS
#LOAD_VARS += CMSSW_MVA_IMAGE_VARS
#LOAD_VARS += CMSSW_MVA_GRAPH_VARS
LOAD_VARS = list(set(LOAD_VARS))
