#!/bin/sh
#
# Execute training and evaluation for electron ID
#
# Run with: source runme.sh

CONFIG="tune0"
#DATAPATH="/home/user/imperial_trees"
DATAPATH="/home/user/imperial_new_trees"

python eid_train.py --config $CONFIG --datapath $DATAPATH --datasets 0 #,1 #,2,3,4,5,6
python eid_eval.py  --config $CONFIG --datapath $DATAPATH --datasets 0 #,1 #,2,3,4,5,6
