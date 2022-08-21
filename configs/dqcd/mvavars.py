# Variables we read out from the root files (regular expressions supported here)
#LOAD_VARS = ['.+hlt.?', '.?gen.?']
#LOAD_VARS = ['.*'] # all


"""
KINEMATIC_GEN_VARS = [
  'gen_pt',
  'gen_eta',
  'gen_phi',
  'gen_energy'
]
"""

TRIGGER_VARS = [
  'HLT_Mu9_IP6_part0',
  'HLT_Mu9_IP6_part1',
  'HLT_Mu9_IP6_part2',
  'HLT_Mu9_IP6_part3',
  'HLT_Mu9_IP6_part4'
]

KINEMATIC_VARS = [
  # "Dummy" variables
  'ChsMET_phi',
  'ChsMET_pt',
  'ChsMET_sumEt'
]

MVA_SCALAR_VARS = [
  'nJet',
  'nMuon',
  'nsv',

  'MET_pt',
  'MET_phi'
]

# -----------------------------------------
# Conditional (parametric) model variables
MODEL_VARS = [
  '__model_m',
  '__model_ctau'
]
MVA_SCALAR_VARS += MODEL_VARS
# -----------------------------------------

# Charged particle flow
MVA_CPF_VARS = [
  
  'cpf_px',
  'cpf_py',
  'cpf_pz',

  'cpf_trackSip2dVal',
  'cpf_trackSip2dSig',

  'cpf_trackSip3dVal',
  'cpf_trackSip3dSig',
  
  'cpf_matchedSV',
  'cpf_jetIdx'
]

#MVA_CPF_VARS = ['cpf_.*']

# Neutral particle flow
MVA_NPF_VARS = [
  
  'npf_px',
  'npf_py',
  'npf_pz',

  'npf_jetIdx'
]

#MVA_NPF_VARS = ['npf_.*']

MVA_PF_VARS = MVA_CPF_VARS + MVA_NPF_VARS

# -----------------------------------------

# Jet variables
MVA_JET_VARS = [
  'Jet_pt',
  'Jet_eta',
  'Jet_phi',
  #'Jet_mass',

  'Jet_chEmEF',
  'Jet_chHEF',
  'Jet_neEmEF',
  'Jet_neHEF',

  'Jet_muEF',
  'Jet_muonSubtrFactor',

  'Jet_chFPV0EF',
  #'Jet_chFPV1EF',
  #'Jet_chFPV2EF',
  #'Jet_chFPV3EF',

  #'Jet_btagCMVA',
  #'Jet_btagCSVV2',
  #'Jet_btagDeepB',
  #'Jet_btagDeepC',
  #'Jet_btagDeepFlavB',
  #'Jet_btagDeepFlavC',
  'Jet_hadronFlavour',

  'Jet_nMuons'
]

#MVA_JET_VARS = ['Jet_.*']

# Muon variables
MVA_MUON_VARS = [
  'Muon_eta',
  'Muon_phi',
  'Muon_pt',

  'Muon_ptErr',
  'Muon_dxy',
  'Muon_dxyErr',
  
  'Muon_dz',
  'Muon_dzErr',
  
  'Muon_ip3d'
  'Muon_sip3d',
  'Muon_charge',
  
  #'Muon_mvaTTH',
  'Muon_tightId',
  'Muon_pfRelIso03_all',
  'Muon_miniPFRelIso_all'
]

#MVA_MUON_VARS = ['Muon_.*']

# Secondary vertex:
# - 'SV_' is the nanoAOD CMS-standard collection
# - 'sv_' is the custom jet-matched collection
# - 'svAdapted_' is the "customized" version of sv
#
MVA_SV_VARS = [
  'sv_ptrel',
  #'sv_mass',
  'sv_deta',
  'sv_dphi',
  'sv_dxy',
  'sv_dxysig',
  'sv_d3d',
  'sv_d3dsig',
  'sv_deltaR',
  'sv_costhetasvpv',
  'sv_chi2',
  'sv_ntracks',
  'sv_ndof'
]

#MVA_SV_VARS = ['sv_.*']


MVA_JAGGED_VARS = MVA_JET_VARS + MVA_MUON_VARS + MVA_SV_VARS + MVA_PF_VARS
# -----------------------------------------

PLOT_VARS = [
  'nJet',
  'nMuon',
  'nsv',
  
  'ChsMET_phi',
  'ChsMET_pt'
  #'sv_mass',
  #'sv_dxy',
  #'sv_dxysig',
  #'sv_dphi',
  #'sv_deta'
]

LOAD_VARS = [
  #'Jet_chEmEF',
  #'Jet_chHEF',
  #'Jet_eta',
  #'Jet_neEmEF',
  #'Jet_neHEF',
  #'Jet_phi',
  #'Jet_pt',
  #'Jet_nMuons',

  #'Muon_eta',
  #'Muon_phi',
  #'Muon_pt',

  #'sv_deta',
  #'sv_dphi',
  #'sv_mass',
  #'sv_chi2',
  #'sv_dxy',
  #'sv_dxysig',
  #'sv_d3d',
  #'sv_d3dsig',
  #'sv_ntracks',
  #'sv_ndof'
]

LOAD_VARS += KINEMATIC_VARS
LOAD_VARS += MVA_SCALAR_VARS
LOAD_VARS += MVA_JAGGED_VARS
LOAD_VARS += MVA_PF_VARS

LOAD_VARS += PLOT_VARS
LOAD_VARS += TRIGGER_VARS

print(LOAD_VARS)
