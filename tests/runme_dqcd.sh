#!/bin/sh
#
# Execute training and evaluation for the DQCD analysis
#
# Run with: source runme.sh

CONFIG="tune0.yml"
DATAPATH="/home/user/travis-stash/input/icedqcd"

mkdir "figs/dqcd/config_[$CONFIG]" -p # for output ascii dump

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# Use * or other glob wildcards for filenames
# tee redirect output to both a file and to screen
python analysis/dqcd.py --runmode genesis $MAX --inputmap mc_input.yml --config $CONFIG --datapath $DATAPATH
python analysis/dqcd.py --runmode train   $MAX --inputmap mc_input.yml --config $CONFIG --datapath $DATAPATH --use_conditional 0
python analysis/dqcd.py --runmode eval    $MAX --inputmap mc_input.yml --config $CONFIG --datapath $DATAPATH --use_conditional 0
