# HLT-MVA variables. Use here only variables available in real data.


KINEMATIC_GEN_ID = [
  'gen_pt',
  'gen_eta',
  'gen_phi',
  'gen_energy'
]

KINEMATIC_ID = [
  'x_hlt_pt',
  'x_hlt_eta'
]

MVA_ID = [
  'x_hlt_pms2',
  'x_hlt_invEInvP',
  'x_hlt_trkDEtaSeed',
  'x_hlt_trkDPhi',
  'x_hlt_trkChi2',
  'x_hlt_trkValidHits',
  'x_hlt_trkNrLayerIT'
]

# Variables we are interested in, need to be found in both MC and DATA
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
