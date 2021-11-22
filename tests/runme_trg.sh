#!/bin/sh
#
# Execute training and evaluation for the electron HLT trigger
#
# Run with: source runme.sh

CONFIG="tune0"

DATAPATH="./travis-stash/input/icetrg"
DATAPATH="/vols/cms/mmieskol/HLT_electron_data/22112021"
#DATAPATH="/home/user/HLT_electron_data/22112021"


# Use * or other glob wildcards for filenames
mkdir ./figs/trg/$CONFIG -p # for output ascii dump

# tee redirect output to both a file and to screen
python ./analysis/trg_train.py --config $CONFIG --datapath $DATAPATH --datasets "none" | tee "./figs/trg/$CONFIG/train_output.txt"
#python ./analysis/trg_eval.py  --config $CONFIG --datapath $DATAPATH --datasets "none" | tee "./figs/trg/$CONFIG/eval_output.txt"


