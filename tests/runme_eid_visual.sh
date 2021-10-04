#!/bin/sh
#
# Execute training and evaluation for electron ID
#
# Run with: source runme.sh

CONFIG="tune0"

DATAPATH="./travis-stash/input/iceid"

# Use * or other glob wildcards for filenames

python ./analysis/eid_visual_tensors.py --config $CONFIG --datapath $DATAPATH --datasets "output_*" #,1 #,2,3,4,5,6
