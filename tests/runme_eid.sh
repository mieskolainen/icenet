#!/bin/sh
# 
# Execute training and evaluation for the electron ID
# 
# Run with: maxevents=10000; source tests/runme.sh

CONFIG="tune0.yml"

#DATAPATH="/vols/cms/icenet/data/2020Oct16"
#MAX="--maxevents 30000000"
#DATASETS="output_[0-4].root" # Use e.g. _[0-20] or _*

DATAPATH="./actions-stash/input/iceid"
DATASETS="output_*.root"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# Use * or other glob wildcards for filenames
python analysis/eid.py --runmode genesis $MAX --config $CONFIG --datapath $DATAPATH --datasets $DATASETS
python analysis/eid.py --runmode train   $MAX --config $CONFIG --datapath $DATAPATH --datasets $DATASETS
python analysis/eid.py --runmode eval    $MAX --config $CONFIG --datapath $DATAPATH --datasets $DATASETS
