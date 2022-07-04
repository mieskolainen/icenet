# MVA variables. Use here only variables available in real data.


# Use these to test that the (re-weight trained) classifier
# is independent of these
KINEMATIC_VARS = [

]

MVA_SCALAR_VARS = [
  'candidate_energy',
  'candidate_px',
  'candidate_py',
  'candidate_pz',

  'candidate_charge',
  'candidate_pdgId'
]

"""
MVA_SCALAR_VARS = [

  'candidate_charge',
  'candidate_pdgId',
  #'candidate_id_probabilities',
  'candidate_time',
  'candidate_timeErr',
  'candidate_energy',
  'candidate_px',
  'candidate_py',
  'candidate_pz',
  'track_in_candidate',
  'tracksters_in_candidate'
]
"""