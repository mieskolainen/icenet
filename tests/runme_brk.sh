#!/bin/sh
#
# Execute training and evaluation for B-parking
#
# Run with: maxevents=10000; source tests/runme.sh

CONFIG="tune0.yml"
DATAPATH="./actions-stash/input/icebrk"

mkdir "figs/brk/config__$CONFIG" -p # for output ascii dump

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# Use * or other glob wildcards for filenames

# Training, Calculation, Analysis
python analysis/brk_train.py   --runmode all $MAX --config $CONFIG --datapath $DATAPATH --datasets 'BParkNANO_mc_relaxed_Kee_0.root' #| tee "./figs/brk/$CONFIG/train_output.txt"
python analysis/brk_calc.py    --runmode all $MAX --config $CONFIG --datapath $DATAPATH --datasets 'BParkNANO_mc_relaxed_Kee_0.root' #| tee "./figs/brk/$CONFIG/calc_output.txt"
python analysis/brk_analyze.py --runmode all $MAX --config $CONFIG --datapath $DATAPATH --datasets 'BParkNANO_mc_relaxed_Kee_0.root' #| tee "./figs/brk/$CONFIG/print_output.txt"
