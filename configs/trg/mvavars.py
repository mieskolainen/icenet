
KINEMATIC_GEN_VARS = [
  'gen_pt',
  'gen_eta',
  'gen_phi',
  'gen_energy'
]

KINEMATIC_VARS = [
  'x_hlt_pt',
  'x_hlt_eta',
  'x_hlt_trkValidHits'
]

# Use here only variables available in real data.
MVA_SCALAR_VARS = [
  'x_hlt_pms2',
  'x_hlt_invEInvP',
  'x_hlt_trkDEtaSeed',
  'x_hlt_trkDPhi',
  'x_hlt_trkChi2',
  'x_hlt_trkValidHits',
  'x_hlt_trkNrLayerIT'
]

# Mutual information regularization targets
MI_VARS = [
  'x_hlt_trkValidHits'
]

# Variables we use with name replacement, need to be found in both MC and DATA
NEW_VARS = [
  'x_hlt_eta',
  'x_hlt_pt',
  
  'x_hlt_pms2',
  'x_hlt_invEInvP',
  'x_hlt_trkDEtaSeed',
  'x_hlt_trkDPhi',
  'x_hlt_trkChi2',
  'x_hlt_trkValidHits',
  'x_hlt_trkNrLayerIT'
]

PLOT_VARS = [
  'gen_e1_pt',
  'gen_e1_eta',
  'gen_e2_pt',
  'gen_e2_eta',
  'gen_pt',
  'gen_eta'
]

# Variables we read out from the root files (regular expressions supported here)
#LOAD_VARS = ['.+hlt.?', '.?gen.?']
LOAD_VARS = ['.*'] # all
