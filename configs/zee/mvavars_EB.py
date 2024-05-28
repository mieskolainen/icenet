# AUX variables + for other re-shuffling purposes
KINEMATIC_VARS  = ['probe_mvaID',
                   'probe_pfChargedIso',
                   'probe_ecalPFClusterIso',
                   'probe_trkSumPtHollowConeDR03',
                   'probe_trkSumPtSolidConeDR04']

# MVA input variables
MVA_SCALAR_VARS = [
    'probe_eta',
    'probe_pt',
    'fixedGridRhoAll',
    'probe_sieie',
    'probe_sieip',
    'probe_s4',
    'probe_r9',
    'probe_pfChargedIsoWorstVtx',
    'probe_ecalPFClusterIso',
    'probe_phiWidth',
    'probe_etaWidth',
    'probe_trkSumPtHollowConeDR03',
    'probe_trkSumPtSolidConeDR04',
    'probe_pfChargedIso'
]

# Technical for MI
MI_VARS = ['probe_eta', 'probe_pt', 'fixedGridRhoAll']

# These we load
LOAD_VARS = list(set(KINEMATIC_VARS + MVA_SCALAR_VARS + MI_VARS))
LOAD_VARS.sort() # Important, set() can have different order run-to-run !

