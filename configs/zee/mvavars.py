
KINEMATIC_VARS  = ['probe_eta', 'probe_pt', 'fixedGridRhoAll']

MVA_SCALAR_VARS = [
    'probe_sieie',
    'probe_sieip',
    'probe_s4',
    'probe_r9',
    'probe_pfChargedIsoWorstVtx',
    'probe_esEnergyOverRawE',
    'probe_esEffSigmaRR',
    'probe_ecalPFClusterIso',
    'probe_phiWidth',
    'probe_etaWidth',
    'probe_trkSumPtHollowConeDR03',
    'probe_trkSumPtSolidConeDR04',
    'probe_pfChargedIso',
]

MVA_SCALAR_VARS_ALL = MVA_SCALAR_VARS + KINEMATIC_VARS
LOAD_VARS           = KINEMATIC_VARS + MVA_SCALAR_VARS

#
MI_VARS = ['probe_eta', 'probe_pt']
