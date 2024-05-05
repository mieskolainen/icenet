#!/bin/sh
# 
# Execute training and evaluation for the electron ID
# 
# Run with: maxevents=10000; source runme.sh

CONFIG="tune0.yml"
DATAPATH="./travis-stash/input/icebrem"
MCMAP="mc_map.yml"
#DATASETS="output_*.root"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

python analysis/brem.py --runmode genesis --config $CONFIG --datapath $DATAPATH --inputmap $MCMAP $MAX
python analysis/brem.py --runmode train   --config $CONFIG --datapath $DATAPATH --inputmap $MCMAP $MAX
python analysis/brem.py --runmode eval    --config $CONFIG --datapath $DATAPATH --inputmap $MCMAP $MAX
