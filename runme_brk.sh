#!/bin/sh
#
# Execute training and evaluation
#
# Run with: source runme.sh

CONFIG="tune0"

#DATAPATH="/home/user/imperial_trees/georges_trees"
DATAPATH="/home/user/imperial_trees/bparking_April"

# Training
python ./analysis/brk_train.py --config $CONFIG --datapath $DATAPATH --datasets 0

# Calculation
python ./analysis/brk_calc.py  --config $CONFIG --datapath $DATAPATH --datasets 0

# Statistics
python ./analysis/brk_print.py  --config $CONFIG --datapath $DATAPATH --datasets 0

#python ./analysis/brk_print.py --config $CONFIG --datapath $DATAPATH --datasets 0 #,1,2,3 #,4,5,6,7
