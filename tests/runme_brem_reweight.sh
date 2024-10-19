#!/bin/sh
# 
# Execute training and evaluation for the electron ID
# 
# Run with: maxevents=10000; source tests/runme.sh

CONFIG="tune0_reweight.yml"
#DATAPATH="/vols/cms/bainbrid/ntuples/icenet/"
DATAPATH="./"
MCMAP="map_mc_test.yml" # uses actions-stash/input/icebrem
#MCMAP="map_mc.yml" # local-stash
#MCMAP="map_mc_large.yml" # large-stash
#DATASETS="output_*.root"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

python analysis/brem.py --runmode genesis --config $CONFIG --datapath $DATAPATH --inputmap $MCMAP $MAX
python analysis/brem.py --runmode train   --config $CONFIG --datapath $DATAPATH --inputmap $MCMAP $MAX
python analysis/brem.py --runmode eval    --config $CONFIG --datapath $DATAPATH --inputmap $MCMAP $MAX
