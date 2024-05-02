#!/bin/sh
# 
# Execute training and evaluation for the electron ID
# 
# Run with: source runme.sh

CONFIG="tune_reweight.yml"
DATAPATH="./travis-stash/input/lptele"
MCMAP="mc_map.yml"
#DATASETS="output_mc*.root"
MAX=10000000

#python analysis/lowptele.py --runmode genesis --config $CONFIG --datapath $DATAPATH --datasets $DATASETS --maxevents $MAX
#python analysis/lowptele.py --runmode train   --config $CONFIG --datapath $DATAPATH --datasets $DATASETS --maxevents $MAX
#python analysis/lowptele.py --runmode eval    --config $CONFIG --datapath $DATAPATH --datasets $DATASETS --maxevents $MAX

python analysis/lowptele.py --runmode genesis --config $CONFIG --datapath $DATAPATH --inputmap $MCMAP --maxevents $MAX 
python analysis/lowptele.py --runmode train   --config $CONFIG --datapath $DATAPATH --inputmap $MCMAP --maxevents $MAX 
#python analysis/lowptele.py --runmode eval    --config $CONFIG --datapath $DATAPATH --inputmap $MCMAP --maxevents $MAX 
