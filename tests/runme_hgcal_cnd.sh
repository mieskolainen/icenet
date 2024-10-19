#!/bin/sh
# 
# Execute training and evaluation for HGCAL
# 
# Run with: maxevents=10000; source tests/runme.sh

CONFIG="tune0_cnd.yml"

#DATAPATH="./actions-stash/input/hgcal"
DATAPATH="/home/user/actions-stash/input/icehgcal"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# Use * or other glob wildcards for filenames
python analysis/hgcal_cnd.py --runmode genesis $MAX --config $CONFIG --datapath $DATAPATH #| tee "./figs/hgcal/$CONFIG/train_output.txt"
python analysis/hgcal_cnd.py --runmode train   $MAX --config $CONFIG --datapath $DATAPATH #| tee "./figs/hgcal/$CONFIG/train_output.txt"
python analysis/hgcal_cnd.py --runmode eval    $MAX --config $CONFIG --datapath $DATAPATH #| tee "./figs/hgcal/$CONFIG/eval_output.txt"
