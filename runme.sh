#!/bin/sh
#
# Execute training and evaluation
#
# Run with: source runme.sh

CONFIG="tune0"
DATAPATH="/home/user/imperial_trees"

python train.py --config $CONFIG --datapath $DATAPATH --datasets 0,1
python eval.py  --config $CONFIG --datapath $DATAPATH --datasets 0,1

