#!/bin/sh
# 
# Execute training and evaluation for HGCAL
# 
# Run with: source runme.sh

CONFIG="tune0_cnd"

#DATAPATH="./travis-stash/input/hgcal"
DATAPATH="/home/user/travis-stash/input/icehgcal"

mkdir ./figs/hgcal_cnd/$CONFIG -p # for output ascii dump

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# Use * or other glob wildcards for filenames
python analysis/hgcal_cnd_train.py $MAX --config $CONFIG --datapath $DATAPATH --datasets "none" #| tee "./figs/hgcal/$CONFIG/train_output.txt"
python analysis/hgcal_cnd_eval.py  $MAX --config $CONFIG --datapath $DATAPATH --datasets "none" #| tee "./figs/hgcal/$CONFIG/eval_output.txt"
