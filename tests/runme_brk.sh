#!/bin/sh
#
# Execute training and evaluation for B-parking
#
# Run with: source runme.sh

CONFIG="tune0"
DATAPATH="./travis-stash/input/icebrk"

# Training, Calculation, Statistics
python analysis/brk_train.py --config $CONFIG --datapath $DATAPATH --datasets 0
python analysis/brk_calc.py  --config $CONFIG --datapath $DATAPATH --datasets 0
python analysis/brk_print.py  --config $CONFIG --datapath $DATAPATH --datasets 0
