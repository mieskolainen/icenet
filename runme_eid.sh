#!/bin/sh
#
# Execute training and evaluation for electron ID
#
# Run with: source runme.sh

CONFIG="tune0_dev"

DATAPATH="/home/user/imperial_trees/2019sep13"
#DATAPATH="/home/user/imperial_new_trees"
#DATAPATH="/home/user/imperial_old_trees"
#DATAPATH="/vols/cms/icenet/data"

python ./analysis/eid_train.py --config $CONFIG --datapath $DATAPATH --datasets 0 #,1 #,2,3,4,5,6
python ./analysis/eid_eval.py  --config $CONFIG --datapath $DATAPATH --datasets 0 #,1 #,2,3,4,5,6
#python ./analysis/eid_visual_tensors.py --config $CONFIG --datapath $DATAPATH --datasets 0 #,1 #,2,3,4,5,6
