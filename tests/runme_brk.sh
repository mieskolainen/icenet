#!/bin/sh
#
# Execute training and evaluation for B-parking
#
# Run with: source runme.sh

CONFIG="tune0"
DATAPATH="./travis-stash/input/icebrk"

if [ -z ${MAXEVENTS+x} ]; then MAX="--maxevents $MAXEVENTS"; else MAX=""; fi

# Training, Calculation, Statistics
python analysis/brk_train.py  $MAX --config $CONFIG --datapath $DATAPATH --datasets 0
python analysis/brk_calc.py   $MAX --config $CONFIG --datapath $DATAPATH --datasets 0
python analysis/brk_print.py  $MAX --config $CONFIG --datapath $DATAPATH --datasets 0
