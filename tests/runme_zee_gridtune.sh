#!/bin/sh
# 
# Execute training and evaluation for the ZEE with a fine tuning GRID SEARCH
# 
# Automatic division of the parameter array points per Condor node. Keep the max
# runtime requirements in mind --> have the number of Condor jobs high enough
# so each job has only a few points to process.
# 
# This script expects environment variables GRID_ID and GRID_NODES
# from `icenet/setenv.sh`, which are set automatically, when one follows the
# instructions below.
# 
# A. When GRID_ID=-1 and GRID_NODES=1 --> Stage 1 training and cache file production.
# B. When GRID_ID=0(1,2,...) and GRID_NODES=N --> Training with different parameters.
# 
# Step 1. See files under tests/zee and modify relevant variables there (do not modify here)
# 
# Step 2. Add executable rights to all relevant .sh files with: chmod +x *.sh
# 
# Step 3. Submit Condor jobs with:
#         cd tests/zee
#         source submit.sh
# 
# m.mieskolainen@imperial.ac.uk, 2024
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Set default values (do not modify these, but set outside this script)

DEFAULT_DATAPATH="./actions-stash/input/icezee"
DEFAULT_CONFIG="tune0_EEm"
DEFAULT_MODELTAG="none"

DEFAULT_BETA_ARRAY=(0.0 0.1)
DEFAULT_SIGMA_ARRAY=(0.0 0.2)
DEFAULT_SWD_VAR="['.*']"
# -----------------------------------------------------------------------

# Check if is set, otherwise use the default

if [ -z "${DATAPATH+x}" ]; then
  DATAPATH=${DEFAULT_DATAPATH}
fi

if [ -z "${CONFIG+x}" ]; then
  CONFIG=${DEFAULT_CONFIG}
fi

if [ -z "${MODELTAG+x}" ]; then
  MODELTAG=${DEFAULT_MODELTAG}
fi

if [ -z "${BETA_ARRAY+x}" ]; then
  BETA_ARRAY=${DEFAULT_BETA_ARRAY}
fi

if [ -z "${SIGMA_ARRAY+x}" ]; then
  SIGMA_ARRAY=${DEFAULT_SIGMA_ARRAY}
fi

if [ -z "${SWD_VAR+x}" ]; then
  SWD_VAR=${DEFAULT_SWD_VAR}
fi

if [ ${maxevents+x} ]; then 
  MAX="--maxevents $maxevents"
else 
  MAX=""
fi


# -----------------------------------------------------------------------
# Initialization (Stage 1 training and pickle file creation)

# Ensure that GRID_ID and GRID_NODES are set to special values

if [[ $GRID_ID == -1 && $GRID_NODES == 1 ]]; then

  python analysis/zee.py --runmode genesis $MAX --modeltag ${MODELTAG} --config ${CONFIG}.yml --datapath $DATAPATH
  
  python analysis/zee.py --runmode train $MAX --modeltag ${MODELTAG} --config ${CONFIG}.yml --datapath $DATAPATH \
    --modeltag GRIDTUNE --run_id "INIT" --compute 0
  
  python analysis/zee.py --runmode eval $MAX --modeltag ${MODELTAG} --config ${CONFIG}.yml --datapath $DATAPATH \
    --modeltag GRIDTUNE --run_id "INIT" --compute 0
  
  return 0 # do not use exit
fi


# -----------------------------------------------------------------------
# Generic functions

# Function to generate all combinations of N arrays
generate_combinations() {
  local arr=("${!1}")   # First array
  shift
  if [ "$#" -eq 0 ]; then
    # Base case: return each element of the last array as a combination
    for elem in "${arr[@]}"; do
      echo "$elem"
    done
  else
    # Recursive case: get combinations of the remaining arrays
    local sub_combinations=($(generate_combinations "$@"))
    for elem in "${arr[@]}"; do
      for sub_comb in "${sub_combinations[@]}"; do
        echo "$elem,$sub_comb"
      done
    done
  fi
}

# Function to assign combinations
assign_combinations() {
  local total_combinations=$1
  local grid_id=$2
  local grid_nodes=$3
  local -n indices_ref=$4

  # Calculate total combinations and assign combinations per node
  local combinations_per_node=$(( (total_combinations + grid_nodes - 1) / grid_nodes )) # ceil(TOTAL_COMBINATIONS / GRID_NODES)

  # Calculate the start and end index for this GRID_ID
  local start_index=$(( grid_id * combinations_per_node ))
  local end_index=$(( start_index + combinations_per_node ))

  # Ensure we don't go out of bounds
  if [ "$end_index" -gt "$total_combinations" ]; then
    end_index=$total_combinations
  fi

  # Store the start and end indices in the reference array
  indices_ref=($start_index $end_index)
}

# -----------------------------------------------------------------------
# Combinatorics and indices

# Generate combinations
COMBINATIONS=($(generate_combinations BETA_ARRAY[@] SIGMA_ARRAY[@]))

# Assign combinations and store result in an array
assign_combinations "${#COMBINATIONS[@]}" $GRID_ID $GRID_NODES indices
START_INDEX=${indices[0]}
END_INDEX=${indices[1]}

# -----------------------------------------------------------------------
# Run through all the configurations

# Loop over combinations assigned to this GRID_ID
for (( i = START_INDEX; i < END_INDEX; i++ )); do

COMBINATION="${COMBINATIONS[$i]}"

# 1. Extract individual variable values from the combination string
IFS=',' read -r BETA SIGMA <<< "$COMBINATION"

# 2. Label the run
RUN_ID="point_${i}__beta_${BETA}__sigma_${SIGMA}"

# 3. Define tune command
SUPERTUNE="\
models.iceboost_swd.SWD_param.beta=${BETA} \
models.iceboost_swd.opt_param.noise_reg=${SIGMA} \
models.iceboost_swd.SWD_param.var=${SWD_VAR} \
"

# Print out
echo $RUN_ID
echo $SUPERTUNE
echo ""

# 4. Run
python analysis/zee.py --runmode genesis $MAX --modeltag ${MODELTAG} --config ${CONFIG}.yml --datapath $DATAPATH 

python analysis/zee.py --runmode train   $MAX --modeltag ${MODELTAG} --config ${CONFIG}.yml --datapath $DATAPATH \
  --modeltag GRIDTUNE --run_id $RUN_ID --supertune "${SUPERTUNE}" # Note " "

python analysis/zee.py --runmode eval    $MAX --modeltag ${MODELTAG} --config ${CONFIG}.yml --datapath $DATAPATH \
  --modeltag GRIDTUNE --run_id $RUN_ID --evaltag "minloss" --supertune "models.iceboost_swd.readmode=-1" 

python analysis/zee.py --runmode eval    $MAX --modeltag ${MODELTAG} --config ${CONFIG}.yml --datapath $DATAPATH \
  --modeltag GRIDTUNE --run_id $RUN_ID --evaltag "last"    --supertune "models.iceboost_swd.readmode=-2" 

done
