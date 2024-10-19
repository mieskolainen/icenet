#!/bin/sh
# 
# Execute training and evaluation for HGCAL
# 
# Run with: maxevents=10000; source tests/runme.sh

CONFIG="tune0_trk.yml"

#DATAPATH="./actions-stash/input/hgcal/close_by_double_pion"
DATAPATH="/home/user/actions-stash/input/icehgcal/close_by_double_pion"
TAG='close_by_double_pion'

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# Use * or other glob wildcards for filenames
python analysis/hgcal_trk.py --runmode genesis $MAX --config $CONFIG --datapath $DATAPATH --datasets "ntuples*.root" #| tee "./figs/hgcal/$CONFIG/train_output.txt"
python analysis/hgcal_trk.py --runmode train   $MAX --config $CONFIG --datapath $DATAPATH --datasets "ntuples*.root" #| tee "./figs/hgcal/$CONFIG/train_output.txt"
python analysis/hgcal_trk.py --runmode eval    $MAX --config $CONFIG --datapath $DATAPATH --datasets "ntuples*.root" #| tee "./figs/hgcal/$CONFIG/eval_output.txt"
