#!/bin/sh
#
# Execute training and evaluation for electron ID
#
# Run with: source runme.sh

CONFIG="tune0"
DATAPATH="/home/user/cernbox/HLT_electrons"


# Use * or other glob wildcards for filenames

python ./analysis/trg_train.py --config $CONFIG --datapath $DATAPATH --datasets "none" #,1 #,2,3,4,5,6
python ./analysis/trg_eval.py  --config $CONFIG --datapath $DATAPATH --datasets "none" #,1 #,2,3,4,5,6
