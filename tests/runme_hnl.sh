#!/bin/sh
#
# Execute training and evaluation for the HNL
#
# Run with: source runme.sh

CONFIG="tune0.yml"
DATAPATH="/home/user/travis-stash/input/icehnl"
#DATAPATH="/vols/cms/mmieskol/HLT_electron_data/22112021"
#DATAPATH="/home/user/HLT_electron_data/22112021"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# Use * or other glob wildcards for filenames
mkdir "figs/hnl/config_[$CONFIG]" -p # for output ascii dump

# tee redirect output to both a file and to screen
python analysis/hnl.py --runmode "genesis" $MAX --config $CONFIG --datapath $DATAPATH --datasets "none" #| tee "./figs/hnl/$CONFIG/train_output.txt"
python analysis/hnl.py --runmode "train"   $MAX --config $CONFIG --datapath $DATAPATH --datasets "none" #| tee "./figs/hnl/$CONFIG/train_output.txt"
python analysis/hnl.py --runmode "eval"    $MAX --config $CONFIG --datapath $DATAPATH --datasets "none" #| tee "./figs/hnl/$CONFIG/eval_output.txt"
