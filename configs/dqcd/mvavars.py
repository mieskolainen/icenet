# HLT-MVA variables. Use here only variables available in real data.


# Variables we read out from the root files (regular expressions supported here)
#LOAD_VARS = ['.+hlt.?', '.?gen.?']
#LOAD_VARS = ['.'] # all


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
  'nsv'
]

# -----------------------------------------
# Conditional (parametric) model variables
MODEL_VARS = [
  '__model_m',
  '__model_ctau'
]
MVA_SCALAR_VARS += MODEL_VARS
# -----------------------------------------

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

MVA_NPF_VARS = [
  
  'npf_px',
  'npf_py',
  'npf_pz',

  'npf_jetIdx'
]

MVA_PF_VARS = MVA_CPF_VARS + MVA_NPF_VARS

# -----------------------------------------

MVA_JAGGED_VARS = [
  
  'Jet_eta',
  'Jet_phi',
  'Jet_pt',
  'Jet_mass',

  'Jet_chEmEF',
  'Jet_chHEF',
  'Jet_neEmEF',
  'Jet_neHEF',
  'Jet_nMuons',

  'Muon_eta',
  'Muon_phi',
  'Muon_pt',
  'Muon_dxy',
  'Muon_dz',
  'Muon_charge',

  'sv_deta',
  'sv_dphi',
  'sv_dxy',
  'sv_dxysig',
  'sv_deltaR',
  'sv_d3d',
  'sv_d3dsig',
  'sv_chi2',
  'sv_ntracks',
  'sv_ndof'
]

# -----------------------------------------
MVA_JAGGED_VARS += MVA_PF_VARS
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
