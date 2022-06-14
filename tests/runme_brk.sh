#!/bin/sh
#
# Execute training and evaluation for B-parking
#
# Run with: source runme.sh

CONFIG="tune0"
DATAPATH="./travis-stash/input/icebrk"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# Training, Calculation, Statistics
python analysis/brk_train.py  $MAX --config $CONFIG --datapath $DATAPATH --datasets 0
python analysis/brk_calc.py   $MAX --config $CONFIG --datapath $DATAPATH --datasets 0
python analysis/brk_print.py  $MAX --config $CONFIG --datapath $DATAPATH --datasets 0
