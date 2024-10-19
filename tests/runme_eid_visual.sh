#!/bin/sh
#
# Execute "image matrix" visualization for electron ID
#
# Run with: maxevents=10000; source tests/runme.sh

CONFIG="tune0.yml"
DATAPATH="./actions-stash/input/iceid"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# Use * or other glob wildcards for filenames
python analysis/eid_visual_tensors.py $MAX --config $CONFIG --datapath $DATAPATH --datasets "output_*.root"
