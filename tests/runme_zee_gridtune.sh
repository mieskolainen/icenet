#!/bin/sh
# 
# Execute training and evaluation for the ZEE with a fine tuning GRID SEARCH
# 
# Automatic division of the parameter array points per Condor node. Keep the max
# runtime requirements in mind --> have the number of Condor jobs high enough
# so each job has only a few points to process.
# 
# This script expects environment variables GRID_ID and GRID_NODES
# from `setenv.sh`, which are set automatically, when one follows the
# instructions below.
# 
#
# Step 1. Create the following files in the homefolder
# 
# -----------------------------------------------------------------------
# `gpu_gridtune_task.job` condor array job steering file with 4 jobs
# -----------------------------------------------------------------------
# 
# executable = gpu_gridtune_task.sh
# arguments  = "$(PROCESS) 4"
# error      = gpu.$(CLUSTER).$(PROCESS).out
# output     = gpu.$(CLUSTER).$(PROCESS).output
# log        = gpu.$(CLUSTER).$(PROCESS).log
# request_gpus   = 1
# request_memory = 80G
# requirements   = TARGET.GPUs_DeviceName =?= "Tesla V100-PCIE-32GB"
# +MaxRuntime    = 86000
# queue 4
# 
# -----------------------------------------------------------------------
# `gpu_gridtune_task.sh` main executable file
# -----------------------------------------------------------------------
# 
# !/bin/sh
# 
# ** icenet/setenv.sh uses these **
# export HTC_PROCESS_ID=$1 
# export HTC_QUEUE_SIZE=$2
# 
# source <full_path_to_your_conda_init_script>/setconda.sh
# conda activate icenet
# 
# cd <full_path>/icenet
# source <full_path>/icenet/setenv.sh
#
# maxevents=300000
# DATAPATH=<full_path_to_your_data>
# CONFIG="tune0_EEm"
# 
# source <full_path>/icenet/tests/runme_zee_gridtune.sh
# 
# -----------------------------------------------------------------------
# 
# Step 2. Add executable rights with: chmod +x *.sh
# 
# Step 3. Submit with: condor_submit gpu_gridtune_task.job
# 
# 
# m.mieskolainen@imperial.ac.uk, 2024
# -----------------------------------------------------------------------

# Set default values

#DEFAULT_DATAPATH="/vols/cms/pfk18/phd/hgg/Jul23/NN21July/N/validations/outputs/Csplit_Jsamp/files"
DEFAULT_DATAPATH="./actions-stash/input/icezee"
DEFAULT_CONFIG="tune0_EB"


# Check if DATAPATH is set, otherwise use the default
if [ -z "${DATAPATH+x}" ]; then
    DATAPATH="$DEFAULT_DATAPATH"
fi

# Check if CONFIG is set, otherwise use the default
if [ -z "${CONFIG+x}" ]; then
    CONFIG="$DEFAULT_CONFIG"
fi

# Handle maxevents as before
if [ ${maxevents+x} ]; then 
    MAX="--maxevents $maxevents"
else 
    MAX=""
fi

# Now DATAPATH and CONFIG are guaranteed to have values
echo "DATAPATH is set to $DATAPATH"
echo "CONFIG is set to $CONFIG"

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
# Run setup

# Define arrays for N variables
BETA_ARRAY=(0.0 0.0025 0.005 0.01 0.02 0.04)
SIGMA_ARRAY=(0.0 0.025 0.05 0.1 0.2)

ARRAY_LIST=(BETA_ARRAY SIGMA_ARRAY)

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
models.iceboost_swd.SWD_param.var=['fixedGridRhoAll', 'probe_eta', 'probe_pt'] \
"

# Print out
echo $RUN_ID
echo $SUPERTUNE
echo ""

# 4. Run
python analysis/zee.py --runmode genesis $MAX --config ${CONFIG}.yml --datapath $DATAPATH 
python analysis/zee.py --runmode train   $MAX --config ${CONFIG}.yml --datapath $DATAPATH --modeltag GRIDTUNE --run_id $RUN_ID --supertune "${SUPERTUNE}" # Note " "
python analysis/zee.py --runmode eval    $MAX --config ${CONFIG}.yml --datapath $DATAPATH --modeltag GRIDTUNE --run_id $RUN_ID --evaltag "minloss" --supertune "models.iceboost_swd.readmode=-1" 
python analysis/zee.py --runmode eval    $MAX --config ${CONFIG}.yml --datapath $DATAPATH --modeltag GRIDTUNE --run_id $RUN_ID --evaltag "last"    --supertune "models.iceboost_swd.readmode=-2" 

done
