#!/bin/sh
# 
# Execute training and evaluation for the electron ID
# 
# Run with: source runme.sh

CONFIG="tune0"
DATAPATH="./travis-stash/input/iceid"
#DATAPATH="/vols/cms/icenet/data/2020Oct16"
#DATAPATH="/home/user/imperial_new_trees/2020Oct16"

mkdir "figs/eid/config_[$CONFIG]" -p # for output ascii dump

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# Use * or other glob wildcards for filenames
python analysis/eid.py --runmode genesis $MAX --config tune0.yml --datapath $DATAPATH --datasets "output_*.root"
python analysis/eid.py --runmode train   $MAX --config tune0.yml --datapath $DATAPATH --datasets "output_*.root"
python analysis/eid.py --runmode eval    $MAX --config tune0.yml --datapath $DATAPATH --datasets "output_*.root"
