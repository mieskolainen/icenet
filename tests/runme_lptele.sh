#!/bin/sh
# 
# Execute training and evaluation for the electron ID
# 
# Run with: source runme.sh

CONFIG="tune0.yml"

#DATAPATH="./travis-stash/input/lptele"
DATAPATH="./data/lptele"
#DATAPATH="/vols/cms/icenet/data/2020Oct16"
#DATAPATH="/home/user/imperial_new_trees/2020Oct16"

DATASETS="output_*.root"

#mkdir -p "figs/lptele/config-[$CONFIG]" # for output ascii dump

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# Use * or other glob wildcards for filenames
python analysis/lowptele.py --runmode genesis $MAX --config $CONFIG --datapath $DATAPATH --datasets $DATASETS
python analysis/lowptele.py --runmode train   $MAX --config $CONFIG --datapath $DATAPATH --datasets $DATASETS
python analysis/lowptele.py --runmode eval    $MAX --config $CONFIG --datapath $DATAPATH --datasets $DATASETS
