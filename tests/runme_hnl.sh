#!/bin/sh
#
# Execute training and evaluation for the HNL
#
# Run with: maxevents=10000; source tests/runme.sh

CONFIG="tune0.yml"
DATAPATH="./actions-stash/input/icehnl"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# tee redirect output to both a file and to screen
python analysis/hnl.py --runmode genesis $MAX --config $CONFIG --datapath $DATAPATH #| tee "./figs/hnl/$CONFIG/train_output.txt"
python analysis/hnl.py --runmode train   $MAX --config $CONFIG --datapath $DATAPATH #| tee "./figs/hnl/$CONFIG/train_output.txt"
python analysis/hnl.py --runmode eval    $MAX --config $CONFIG --datapath $DATAPATH #| tee "./figs/hnl/$CONFIG/eval_output.txt"
