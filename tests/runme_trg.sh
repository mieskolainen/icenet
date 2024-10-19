#!/bin/sh
#
# Execute training and evaluation for the electron HLT trigger
#
# Run with: maxevents=10000; source tests/runme.sh

CONFIG="tune0.yml"
DATAPATH="./actions-stash/input/icetrg"
#DATAPATH="/vols/cms/mmieskol/HLT_electron_data/22112021"
#DATAPATH="/home/user/HLT_electron_data/22112021"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# tee redirect output to both a file and to screen
python analysis/trg.py --runmode "genesis" $MAX --config $CONFIG --datapath $DATAPATH
python analysis/trg.py --runmode "train"   $MAX --config $CONFIG --datapath $DATAPATH
python analysis/trg.py --runmode "eval"    $MAX --config $CONFIG --datapath $DATAPATH
