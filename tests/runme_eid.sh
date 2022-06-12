#!/bin/sh
# 
# Execute training and evaluation for the electron ID
# 
# Run with: source runme.sh

CONFIG="tune0"

DATAPATH="./travis-stash/input/iceid"
#DATAPATH="/vols/cms/icenet/data/2020Oct16"
#DATAPATH="/home/user/imperial_new_trees/2020Oct16"

mkdir ./figs/eid/$CONFIG -p # for output ascii dump


# Use * or other glob wildcards for filenames
python analysis/eid_train.py --config $CONFIG --datapath $DATAPATH --datasets "output_*.root" #| tee "./figs/eid/$CONFIG/train_output.txt" #,1 #,2,3,4,5,6
python analysis/eid_eval.py  --config $CONFIG --datapath $DATAPATH --datasets "output_*.root" #| tee "./figs/eid/$CONFIG/eval_output.txt" #,1 #,2,3,4,5,6

