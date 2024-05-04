#!/bin/sh
# 
# Execute training and evaluation for the electron ID
# 
# Run with: source runme.sh

CONFIG="tune0.yml"
DATAPATH="./travis-stash/input/icebrem"
MCMAP="mc_map.yml"
#DATASETS="output_*.root"
MAX=10000000

#python analysis/brem.py --runmode genesis --config $CONFIG --datapath $DATAPATH --datasets $DATASETS --maxevents $MAX
#python analysis/brem.py --runmode train   --config $CONFIG --datapath $DATAPATH --datasets $DATASETS --maxevents $MAX
#python analysis/brem.py --runmode eval    --config $CONFIG --datapath $DATAPATH --datasets $DATASETS --maxevents $MAX

python analysis/brem.py --runmode genesis --config $CONFIG --datapath $DATAPATH --inputmap $MCMAP --maxevents $MAX
python analysis/brem.py --runmode train   --config $CONFIG --datapath $DATAPATH --inputmap $MCMAP --maxevents $MAX
python analysis/brem.py --runmode eval    --config $CONFIG --datapath $DATAPATH --inputmap $MCMAP --maxevents $MAX
