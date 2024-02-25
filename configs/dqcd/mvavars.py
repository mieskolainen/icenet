# DQCD root file input

# ---------------------------------------------------------
# Conditional (parametric) signal model variables

MODEL_VARS = [
  'MODEL_mpi',
  'MODEL_mA',
  'MODEL_ctau'
]

# ---------------------------------------------------------
# For plots, diagnostics ...

KINEMATIC_GEN_VARS = [
  # Conditional theory MC point parameters
  # Read from .yaml (input) steering cards and constructed under .iceroot
  'GEN_mpi',
  'GEN_mA',
  'GEN_ctau',
  
  # MC only NanoAOD
  'GenJet_pt',
  'GenJet_eta',
  'GenJet_phi',
  'GenJet_mass'
]

KINEMATIC_VARS = [
  
  # Data & MC NanoAOD
  'nJet',
  'nMuon',
  'nsv',

  'ChsMET_phi',
  'ChsMET_pt' ,
  'ChsMET_sumEt'
]

# ---------------------------------------------------------
# Trigger flag bit variables

TRIGGER_VARS = [
  'HLT_Mu9_IP6_part0',
  'HLT_Mu9_IP6_part1',
  'HLT_Mu9_IP6_part2',
  'HLT_Mu9_IP6_part3',
  'HLT_Mu9_IP6_part4'
]

# ---------------------------------------------------------
# Pure scalar variables (non-nested)

MVA_SCALAR_VARS = [
  'nJet',
  'nMuon',
  'nsv',

  'MET_pt',
  'MET_phi'
]

# ---------------------------------------------------------
# Charged particle flow
#
# 'cpf_' is the custom (nanotron) jet-matched collection
#
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

# ---------------------------------------------------------
# Neutral particle flow
# 
# 'npf_' is the custom (nanotron) jet-matched collection
#
MVA_NPF_VARS = [
  'npf_px',
  'npf_py',
  'npf_pz',

  'npf_jetIdx'
]

#MVA_NPF_VARS = ['npf_.*']


# ---------------------------------------------------------
# Jets
#  'Jet_' is the standard nanoAOD collection
#
MVA_JET_VARS = [
  'Jet_pt',
  'Jet_eta',
  'Jet_phi',
  'Jet_mass',

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

  'Jet_nMuons',
  'Jet_nElectrons',
  'Jet_nConstituents',

  #'Jet_btagCMVA',
  #'Jet_btagCSVV2',
  'Jet_btagDeepB',
  #'Jet_btagDeepC',
  #'Jet_btagDeepFlavB',
  #'Jet_btagDeepFlavC',
  'Jet_qgl',
  'Jet_puIdDisc',
  
  'Jet_muonIdx1',
  'Jet_muonIdx2' 
]

#MVA_JET_VARS = ['Jet_.*']

# ---------------------------------------------------------
# Muons
#  'Muon_' is the standard nanoAOD collection
#  'muon_' is the jet matched custom-collection 
#
MVA_MUON_VARS = [
  'Muon_eta',
  'Muon_phi',
  'Muon_pt',
  'Muon_ptErr',

  'Muon_dxy',         # impact parameter: dxy (with sign) wrt PV (cm)
  'Muon_dxyErr',
  
  'Muon_dz',          # impact parameter: dz  (with sign) wrt PV (cm)
  'Muon_dzErr',
  
  'Muon_ip3d',        # 3D impact parameter wrt first PV (cm)
  'Muon_sip3d',
  'Muon_charge',
  
  'Muon_tightId',
  'Muon_softMva',            # Soft MVA ID score
  'Muon_pfRelIso03_all',
  'Muon_miniPFRelIso_all',

  'Muon_jetIdx'
]

#MVA_MUON_VARS = ['Muon_.*']

# ---------------------------------------------------------
# Secondary vertex
#  'SV_'        is the standard nanoAOD collection
#  'sv_'        is the jet-matched custom collection
#  'svAdapted_' is the jet-matched custom collection with adapted SV-reco
#  'muonSV_'    is the custom muon SV collection

#  'SV_'        is the standard nanoAOD collection
#  'sv_'        is the jet-matched custom collection
#  'svAdapted_' is the jet-matched custom collection with adapted SV-reco

"""
MVA_SV_VARS = [
  'sv_ptrel',
  'sv_mass',
  'sv_deta',
  'sv_dphi',
  'sv_dxy',
  'sv_dxysig',
  'sv_d3d',
  'sv_d3dsig'
  'sv_deltaR',
  'sv_costhetasvpv',
  'sv_chi2',
  'sv_ntracks',
  'sv_ndof'
]
"""

MVA_MUONSV_VARS = [
  'muonSV_chi2',     # Reduced chi2, i.e. chi2 / ndof
  'muonSV_pAngle',   # Pointing angle: acos(p_SV * (SV - PV))

  'muonSV_dlen',     # 3D decay length (cm)
  'muonSV_dlenSig',  # 3D decay length significance
  'muonSV_dxy',      # 2D transverse decay length (cm)
  'muonSV_dxySig',   # 2D transverse decay length significance

  'muonSV_mu1pt',    # Muon (1) kinematics
  'muonSV_mu1eta',
  'muonSV_mu1phi',
  'muonSV_mu2pt',    # Muon (2) kinematics
  'muonSV_mu2eta',
  'muonSV_mu2phi',
  
  'muonSV_x',        # Sec. vertex position
  'muonSV_y',        # Sec. vertex position
  'muonSV_z'         # Sec. vertex position
]


MVA_SV_VARS = [
  'SV_pt',      # Transverse momentum
  'SV_eta',     # Pseudorapidity
  'SV_phi',     # Atzimuthal angle
  'SV_mass',    # Mass

  'SV_x',       # Sec. vertex position
  'SV_y',       # Sec. vertex position
  'SV_z',       # Sec. vertex position

  'SV_dxy',     # 2D transverse decay length (cm)
  'SV_dxySig',  # 2D transverse decay length significance
  'SV_dlen',    # 3D decay length (cm)
  'SV_dlenSig', # 3D decay length significance
  'SV_pAngle',  # Pointing angle: acos(p_SV * (SV - PV))

  'SV_chi2',    # Reduced chi2, i.e. chi2 / ndof
  'SV_ndof'     # Number of degrees of freedom
]

#MVA_SV_VARS = ['sv_.*']

# ---------------------------------------------------------
# Combine logical sets

KINEMATIC_VARS  += KINEMATIC_GEN_VARS
MVA_SCALAR_VARS += MODEL_VARS           # Treated on the same basis as scalar vars

MVA_PF_VARS      = MVA_CPF_VARS + MVA_NPF_VARS
MVA_JAGGED_VARS  = MVA_JET_VARS + MVA_MUON_VARS + MVA_MUONSV_VARS + MVA_SV_VARS # + MVA_PF_VARS

# ---------------------------------------------------------
# Variables we read out from the root files

LOAD_VARS = []

LOAD_VARS += KINEMATIC_VARS
LOAD_VARS += TRIGGER_VARS
LOAD_VARS += MVA_SCALAR_VARS
LOAD_VARS += MVA_JAGGED_VARS

# Mutual information regularization targets
MI_VARS = [
  #'muonSV_mass.*' # All
  'muonSV_mass_0'  # The leading one
]

print(LOAD_VARS)

# (regular expressions supported here)
#LOAD_VARS = ['.+hlt.?', '.?gen.?']
#LOAD_VARS = ['.*'] # all
