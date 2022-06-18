#!/bin/sh
#
# Execute "deep batched" training and evaluation for electron ID
#
# Run with: source runme.sh

CONFIG="tune0"

DATAPATH="./travis-stash/input/iceid"
#DATAPATH="/vols/cms/icenet/data/2020Oct16"
#DATAPATH="/home/user/imperial_new_trees/2020Oct16"

mkdir ./figs/eid/$CONFIG -p # for output ascii dump

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# Use * or other glob wildcards for filenames
python ./analysis/eid_deep_train.py $MAX --config $CONFIG --datapath $DATAPATH --datasets "output_*.root"
#python ./analysis/eid_eval.py $MAX --config $CONFIG --datapath $DATAPATH --datasets "output_0.root"

