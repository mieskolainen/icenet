#!/bin/sh
#
# Execute training and evaluation for the DQCD analysis
#
# Run with: source runme.sh

CONFIG="tune0.yml"
DATAPATH="/vols/cms/mc3909/"

mkdir ./figs/dqcd/$CONFIG -p # for output ascii dump

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# Use * or other glob wildcards for filenames
# tee redirect output to both a file and to screen
python analysis/dqcd.py --runmode genesis $MAX --inputfiles mc_input_ic.yml --config $CONFIG --datapath $DATAPATH
python analysis/dqcd.py --runmode train   $MAX --inputfiles mc_input_ic.yml --config $CONFIG --datapath $DATAPATH
python analysis/dqcd.py --runmode eval    $MAX --inputfiles mc_input_ic.yml --config $CONFIG --datapath $DATAPATH
