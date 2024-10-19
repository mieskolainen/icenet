#!/bin/sh
#
# Execute "deep batched" training and evaluation for electron ID
#
# Run with: maxevents=10000; source tests/runme.sh

CONFIG="tune0.yml"
DATAPATH="./actions-stash/input/iceid"
#DATAPATH="/vols/cms/icenet/data/2020Oct16"

mkdir "figs/eid/config__$CONFIG" -p # for output ascii dump

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# Use * or other glob wildcards for filenames
python ./analysis/eid_deep_train.py --runmode all $MAX --config $CONFIG --datapath $DATAPATH --datasets "output_*.root"
#python ./analysis/eid_eval.py --runmode eval $MAX --config $CONFIG --datapath $DATAPATH --datasets "output_0.root"

