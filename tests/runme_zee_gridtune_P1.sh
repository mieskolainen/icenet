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
# Step 3. Submit Condor jobs
#         see the README.md under /tests/zee
# 
# m.mieskolainen@imperial.ac.uk, 2025
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Set default values (do not modify these, but set outside this script)

DEFAULT_DATAPATH="./actions-stash/input/icezee"
DEFAULT_CONFIG="tune0_EEm"
DEFAULT_MODELTAG="GRIDTUNE-P1"

DEFAULT_BETA_ARRAY=(0.0 0.1)
DEFAULT_SIGMA_ARRAY=(0.0 0.2)
DEFAULT_LR_ARRAY=(0.1)
DEFAULT_GAMMA_ARRAY=(1.5)
DEFAULT_MAXDEPTH_ARRAY=(13)
DEFAULT_LAMBDA_ARRAY=(2.0)
DEFAULT_ALPHA_ARRAY=(0.05)

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
if [ -z "${LR_ARRAY+x}" ]; then
  LR_ARRAY=${DEFAULT_LR_ARRAY}
fi
if [ -z "${GAMMA_ARRAY+x}" ]; then
  GAMMA_ARRAY=${DEFAULT_GAMMA_ARRAY}
fi
if [ -z "${MAXDEPTH_ARRAY+x}" ]; then
  MAXDEPTH_ARRAY=${DEFAULT_MAXDEPTH_ARRAY}
fi
if [ -z "${LAMBDA_ARRAY+x}" ]; then
  LAMBDA_ARRAY=${DEFAULT_LAMBDA_ARRAY}
fi
if [ -z "${ALPHA_ARRAY+x}" ]; then
  ALPHA_ARRAY=${DEFAULT_ALPHA_ARRAY}
fi


if [ -z "${SWD_VAR+x}" ]; then
  SWD_VAR=${DEFAULT_SWD_VAR}
fi

if [ ${maxevents+x} ]; then 
  MAX="--maxevents $maxevents"
else 
  MAX=""
fi

# Import
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/gridtune_utils.sh"

# -----------------------------------------------------------------------
# Initialization (Stage 1 training and pickle file creation)

# Ensure that GRID_ID and GRID_NODES are set to special values

if [[ $GRID_ID == -1 && $GRID_NODES == 1 ]]; then

  python analysis/zee.py --runmode genesis $MAX --modeltag ${MODELTAG} --config ${CONFIG}.yml --datapath $DATAPATH
  
  return 0 # do not use exit
fi

# -----------------------------------------------------------------------
# Combinatorics and indices

# Generate combinations
COMBINATIONS=($(generate_combinations BETA_ARRAY[@] SIGMA_ARRAY[@] LR_ARRAY[@] GAMMA_ARRAY[@] MAXDEPTH_ARRAY[@] LAMBDA_ARRAY[@] ALPHA_ARRAY[@]))

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
IFS=',' read -r LR GAMMA MAXDEPTH LAMBDA ALPHA <<< "$COMBINATION"

# 2. Label the run
RUN_ID="lr_${LR}__gamma_${GAMMA}__maxdepth_${MAXDEPTH}__lambda_${LAMBDA}__alpha_${ALPHA}"

# 3. Define tune command
SUPERTUNE="\
models.iceboost4D.model_param.learning_rate=${LR} \
models.iceboost4D.model_param.gamma=${GAMMA} \
models.iceboost4D.model_param.max_depth=${MAXDEPTH} \
models.iceboost4D.model_param.reg_lambda=${LAMBDA} \
models.iceboost4D.model_param.reg_alpha=${ALPHA}
"

# Print out
echo $RUN_ID
echo $SUPERTUNE
echo ""

# 4. Run
python analysis/zee.py --runmode train $MAX   --modeltag ${MODELTAG} --config ${CONFIG}.yml --datapath $DATAPATH \
  --run_id $RUN_ID --supertune "${SUPERTUNE}" --compute 0

python analysis/zee.py --runmode eval  $MAX   --modeltag ${MODELTAG} --config ${CONFIG}.yml --datapath $DATAPATH \
  --run_id $RUN_ID --compute 0

done
