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

# Use * or other glob wildcards for filenames

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

python ./analysis/eid_deep_train.py $MAX --config $CONFIG --datapath $DATAPATH --datasets "output_*.root" # output_{0,1}
#python ./analysis/eid_eval.py $MAX --config $CONFIG --datapath $DATAPATH --datasets "output_0.root"

