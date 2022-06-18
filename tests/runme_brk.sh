#!/bin/sh
#
# Execute training and evaluation for B-parking
#
# Run with: source runme.sh

CONFIG="tune0"

DATAPATH="./travis-stash/input/icebrk"

mkdir ./figs/brk/$CONFIG -p # for output ascii dump

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# Use * or other glob wildcards for filenames

# Training, Calculation, Analysis
python analysis/brk_train.py  $MAX --config $CONFIG --datapath $DATAPATH --datasets 'BParkNANO_mc_relaxed_Kee_0.root' | tee "./figs/brk/$CONFIG/train_output.txt"
python analysis/brk_calc.py   $MAX --config $CONFIG --datapath $DATAPATH --datasets 'BParkNANO_mc_relaxed_Kee_0.root' | tee "./figs/brk/$CONFIG/calc_output.txt"
python analysis/brk_print.py  $MAX --config $CONFIG --datapath $DATAPATH --datasets 'BParkNANO_mc_relaxed_Kee_0.root' | tee "./figs/brk/$CONFIG/print_output.txt"
