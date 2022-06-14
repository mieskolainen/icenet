#!/bin/sh
#
# Execute "image matrix" visualization for electron ID
#
# Run with: source runme.sh

CONFIG="tune0"
DATAPATH="./travis-stash/input/iceid"

# Use * or other glob wildcards for filenames

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

python analysis/eid_visual_tensors.py $MAX --config $CONFIG --datapath $DATAPATH --datasets "output_*.root"
