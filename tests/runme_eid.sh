#!/bin/sh
# 
# Execute training and evaluation for the electron ID
# 
# Run with: source runme.sh

CONFIG="tune0"

DATAPATH="./travis-stash/input/iceid"


# Use * or other glob wildcards for filenames

mkdir ./figs/eid/$CONFIG -p # for output ascii dump

python ./analysis/eid_train.py --config $CONFIG --datapath $DATAPATH --datasets "output_0" | tee "./figs/eid/$CONFIG/train_output.txt" #,1 #,2,3,4,5,6
python ./analysis/eid_eval.py  --config $CONFIG --datapath $DATAPATH --datasets "output_0" | tee "./figs/eid/$CONFIG/eval_output.txt" #,1 #,2,3,4,5,6
#python ./analysis/eid_visual_tensors.py --config $CONFIG --datapath $DATAPATH --datasets "output_0" #,1 #,2,3,4,5,6
