#!/bin/sh
#
# Execute training and evaluation
#
# Run with: source runme.sh

CONFIG="tune0"

#DATAPATH="/home/user/imperial_trees/georges_trees"
DATAPATH="/home/user/imperial_trees/bparking_April"

# Training
python brk_train.py --config $CONFIG --datapath $DATAPATH --datasets 0

# Calculation
python brk_calc.py  --config $CONFIG --datapath $DATAPATH --datasets 0

# Statistics
python brk_print.py  --config $CONFIG --datapath $DATAPATH --datasets 0

#python brk_print.py --config $CONFIG --datapath $DATAPATH --datasets 0 #,1,2,3 #,4,5,6,7
