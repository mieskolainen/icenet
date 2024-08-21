#!/bin/sh
#
# Execute training and evaluation for the ZEE
#
# Run with: source runme.sh

#DATAPATH="/vols/cms/pfk18/phd/hgg/Jul23/NN21July/N/validations/outputs/Csplit_Jsamp/files"
DATAPATH="./travis-stash/input/icezee"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

CONFIG="tune0_EEp.yml"

# Step 1. Produce output (cache) files with all sign events
python analysis/zee.py --runmode "genesis" $MAX --config $CONFIG --datapath $DATAPATH --supertune "drop_negative=False"
python analysis/zee.py --runmode "train"   $MAX --config $CONFIG --datapath $DATAPATH --supertune "drop_negative=False" --compute 0
python analysis/zee.py --runmode "eval"    $MAX --config $CONFIG --datapath $DATAPATH --supertune "drop_negative=False" --compute 0

# Step 2. Default training with only positive sign events
python analysis/zee.py --runmode "genesis" $MAX --config $CONFIG --datapath $DATAPATH
python analysis/zee.py --runmode "train"   $MAX --config $CONFIG --datapath $DATAPATH
python analysis/zee.py --runmode "eval"    $MAX --config $CONFIG --datapath $DATAPATH

# ** Change these manually by carefully inspecting outputs under /output and /checkpoint **

# These correspond to Step 1. above
HASH="KA14gKO82bSegMkRSDE5anR10PzJn3mD6Yk8B9Qg5PE="
POST_HASH="KA14gKO82bSegMkRSDE5anR10PzJn3mD6Yk8B9Qg5PE=__jnjX7Oa_HEMM4WH4J1otCnCFoRl1TDIn_6Gx7mh4sz8="

# This corresponds to Step 2. training above
RUN_ID="2024-08-08_20-27-57_lxcgpu00"

# Step 3. Evaluation with all sign events
python analysis/zee.py --runmode "eval" $MAX --config $CONFIG --datapath $DATAPATH --hash_genesis $HASH --hash_post_genesis $POST_HASH --run_id $RUN_ID --evaltag "mixed-eval"

