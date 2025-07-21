#!/bin/sh
#
# Execute training and evaluation for the ZEE
#
# Run with: maxevents=10000; source tests/runme.sh
# 
# m.mieskolainen@imperial.ac.uk, 2025

#DATAPATH="/vols/cms/pfk18/icenet_files/processed_20Feb2025"
DATAPATH="./actions-stash/input/icezee"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

for CONFIG in "tune0_EEm" "tune0_EEp" "tune0_EB"
do
  # Step M1. Produce output (cache) files with all sign events
  # N.B. using tag '--compute 0' does only the steps we need here and skips other
  python analysis/zee.py --runmode genesis $MAX --config ${CONFIG}.yml --datapath $DATAPATH --hash_genesis M1-${CONFIG}                                         --run_id M1-run --supertune "drop_negative=False"
  python analysis/zee.py --runmode train   $MAX --config ${CONFIG}.yml --datapath $DATAPATH --hash_genesis M1-${CONFIG} --hash_post_genesis M1-${CONFIG}__train --run_id M1-run --compute 0
  python analysis/zee.py --runmode eval    $MAX --config ${CONFIG}.yml --datapath $DATAPATH --hash_genesis M1-${CONFIG} --hash_post_genesis M1-${CONFIG}__eval  --run_id M1-run --compute 0

  # Step M2. Default training with only positive sign events
  python analysis/zee.py --runmode genesis $MAX --config ${CONFIG}.yml --datapath $DATAPATH --hash_genesis M2-${CONFIG}                                         --run_id M2-run
  python analysis/zee.py --runmode train   $MAX --config ${CONFIG}.yml --datapath $DATAPATH --hash_genesis M2-${CONFIG} --hash_post_genesis M2-${CONFIG}__train --run_id M2-run
  python analysis/zee.py --runmode eval    $MAX --config ${CONFIG}.yml --datapath $DATAPATH --hash_genesis M2-${CONFIG} --hash_post_genesis M2-${CONFIG}__eval  --run_id M2-run
  
  # Step M3. Evaluation with all sign events
  python analysis/zee.py --runmode eval $MAX --config ${CONFIG}.yml --datapath $DATAPATH --hash_genesis M1-${CONFIG} --hash_post_genesis M1-${CONFIG}__eval --run_id M2-run --evaltag M3__mixed-eval
done

# MANUAL SPECIFIC CACHE FILE SELECTION
# ** by carefully inspecting outputs under /output and /checkpoint **

# Example: These correspond to Step 1. above
#HASH="KA14gKO82bSegMkRSDE5anR10PzJn3mD6Yk8B9Qg5PE="
#POST_HASH="KA14gKO82bSegMkRSDE5anR10PzJn3mD6Yk8B9Qg5PE=__jnjX7Oa_HEMM4WH4J1otCnCFoRl1TDIn_6Gx7mh4sz8="

# Example: This corresponds to Step 2. training above
#RUN_ID="2024-08-08_20-27-57_lxcgpu00"
