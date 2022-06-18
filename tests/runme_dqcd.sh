#!/bin/sh
#
# Execute training and evaluation for the DQCD analysis
#
# Run with: source runme.sh

CONFIG="tune0"

DATAPATH="/home/user/travis-stash/input/icedqcd"

mkdir ./figs/dqcd/$CONFIG -p # for output ascii dump

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# Use * or other glob wildcards for filenames
# tee redirect output to both a file and to screen
python analysis/dqcd_train.py  $MAX --config $CONFIG --datapath $DATAPATH --datasets "none" | tee "./figs/dqcd/$CONFIG/train_output.txt"
python analysis/dqcd_eval.py   $MAX --config $CONFIG --datapath $DATAPATH --datasets "none" | tee "./figs/dqcd/$CONFIG/eval_output.txt"
