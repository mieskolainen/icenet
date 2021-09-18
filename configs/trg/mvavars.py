# HLT-MVA variables. Use here only variables available in real data.


KINEMATIC_GEN_ID = [
  'gen_pt',
  'gen_eta',
  'gen_phi',
  'gen_energy'
]

KINEMATIC_ID = [
  'e1_hlt_pt',
  'e1_hlt_eta',
  
  'l1_pt',
  'l1_eta',
  'l1_phi',
  'l1_dr',
  'l1_dphi',
  'l1_deta'

  'hlt_pt',
  'hlt_eta',
  'hlt_phi',
  'hlt_energy',
  'hlt_dr',
  'hlt_dphi',
  'hlt_deta'
]

MVA_ID = [
  'e1_hlt_pms2',
  'e1_hlt_invEInvP',
  'e1_hlt_trkDEtaSeed',
  'e1_hlt_trkDPhi',
  'e1_hlt_trkChi2',
  'e1_hlt_trkValidHits',
  'e1_hlt_trkNrLayerIT'
]

# Variables we are interested in, need to be found in both MC and DATA
NEW_VARS = [
    'e1_hlt_eta',
    'e1_hlt_pt',
    
    'e1_hlt_pms2',
    'e1_hlt_invEInvP',
    'e1_hlt_trkDEtaSeed',
    'e1_hlt_trkDPhi',
    'e1_hlt_trkChi2',
    'e1_hlt_trkValidHits',
    'e1_hlt_trkNrLayerIT'
]
