#!/bin/sh
# 
# Execute training and evaluation for the electron ID
# 
# Run with: source runme.sh

CONFIG="tune0.yml"
DATAPATH="./data/lptele"
MCMAP="mc_map.yml"
MAX=10000000
#DATASETS="output_*.root"

#python analysis/lowptele.py --runmode genesis --config $CONFIG --datapath $DATAPATH --datasets $DATASETS --maxevents $MAX
#python analysis/lowptele.py --runmode train   --config $CONFIG --datapath $DATAPATH --datasets $DATASETS --maxevents $MAX
#python analysis/lowptele.py --runmode eval    --config $CONFIG --datapath $DATAPATH --datasets $DATASETS --maxevents $MAX

python analysis/lowptele.py --runmode genesis --config $CONFIG --datapath $DATAPATH --inputmap $MCMAP --maxevents $MAX
python analysis/lowptele.py --runmode train   --config $CONFIG --datapath $DATAPATH --inputmap $MCMAP --maxevents $MAX
python analysis/lowptele.py --runmode eval    --config $CONFIG --datapath $DATAPATH --inputmap $MCMAP --maxevents $MAX
