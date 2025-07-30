#!/bin/sh
#
# Generic functions
#
# m.mieskolainen@imperial.ac.uk, 2025

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
