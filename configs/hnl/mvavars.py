
KINEMATIC_GEN_VARS = [

]

# Variables to plot along
KINEMATIC_VARS = [
  'tagger_score',
  'hnlJet_nominal_llpdnnx_ratio_LLP_Q',
  'hnlJet_nominal_llpdnnx_ratio_LLP_QMU',
  'hnlJet_nominal_llpdnnx_ratio_LLP_QE',
  'event_weight'
]

ALL_MVA_VARS = [

  # lepton 1
  'leadingLeptons_pt',
  'leadingLeptons_eta',
  'leadingLeptons_isElectron',

  # lepton 2 
  'subleadingLeptons_pt',
  'subleadingLeptons_eta',
  'subleadingLeptons_isElectron',

  # (l1,l2) pair
  'dilepton_charge',
  'dilepton_dPhimin',
  'dilepton_dRmin',
  'dilepton_mass',

  # (l1,l2) pair -- jet
  'nominal_m_llj',
  'nominal_dR_l2j',

  # Event global
  'nominal_ht',
  'nominal_met',
  'nominal_mtw',
  'nominal_dPhi_met_l1',

  # Event shape
  'nominal_eventShape_isotropy',
  'nominal_eventShape_circularity',
  'nominal_eventShape_sphericity',
  'nominal_eventShape_aplanarity',
  'nominal_eventShape_C',

  # HNL jet
  'hnlJet_nominal_pt',
  'hnlJet_nominal_eta',
  
  # HNL tagger
  'tagger_score'
]


TECHNO_VARS = [
  'label',
  'label_type',
  'genweight',
  'event_weight',
  'tagger_score'
]


# Use here only variables available in real data
MVA_SCALAR_VARS = ALL_MVA_VARS

# Mutual information regularization targets
MI_VARS = [
  'tagger_score'
]

#PLOT_VARS = [
#]

# Variables we read out from the root files (regular expressions supported here)
#LOAD_VARS = ['.+hlt.?', '.?gen.?']
LOAD_VARS = ['.*'] # all
