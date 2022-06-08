# HLT-MVA variables. Use here only variables available in real data.


# Variables we read out from the root files (regular expressions supported here)
#LOAD_VARS = ['.+hlt.?', '.?gen.?']
#LOAD_VARS = ['.'] # all


"""
KINEMATIC_GEN_ID = [
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

KINEMATIC_ID = [
  # "Dummy" variables
  'ChsMET_phi',
  'ChsMET_pt',
  'ChsMET_sumEt'
]

MVA_SCALAR_ID = [
  'nJet',
  'nMuon',
  'nsv'
]

MVA_JAGGED_ID = []
"""
MVA_JAGGED_ID = [
  
  'Jet_chEmEF',
  'Jet_chHEF',
  'Jet_eta',
  'Jet_neEmEF',
  'Jet_neHEF',
  'Jet_phi',
  'Jet_pt',
  'Jet_nMuons',

  'Muon_eta'
  'Muon_phi',
  'Muon_pt',

  'sv_dxy',
  'sv_dxysig',
  'sv_dphi',
  'sv_deta',
  'sv_deltaR',
  'sv_chi2',
  'sv_d3d',
  'sv_d3dsig',
  'sv_ntracks',
  'sv_ndof'
]
"""


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

LOAD_VARS += KINEMATIC_ID
LOAD_VARS += MVA_SCALAR_ID
LOAD_VARS += MVA_JAGGED_ID

LOAD_VARS += PLOT_VARS
LOAD_VARS += TRIGGER_VARS

print(LOAD_VARS)
