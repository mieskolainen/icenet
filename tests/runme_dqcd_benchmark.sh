#!/bin/sh
#
# Execute different training and evaluation runs for DQCD
#
# Run with: source runme.sh

# --------------------
## Process input

DATAPATH="/home/user/travis-stash/input/icedqcd"
CONFIG="tune0.yml"
CMD="python analysis/dqcd.py"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

$CMD $MAX --runmode genesis --config $CONFIG --inputmap mc_input.yml   --datapath $DATAPATH
$CMD $MAX --runmode genesis --config $CONFIG --inputmap mc_input_1.yml --datapath $DATAPATH
$CMD $MAX --runmode genesis --config $CONFIG --inputmap mc_input_2.yml --datapath $DATAPATH


# --------------------
## Train models

$CMD $MAX --runmode train --config $CONFIG --inputmap mc_input.yml   --modeltag pointANY --use_conditional 1
$CMD $MAX --runmode train --config $CONFIG --inputmap mc_input_1.yml --modeltag point1
$CMD $MAX --runmode train --config $CONFIG --inputmap mc_input_2.yml --modeltag point2


# --------------------
## Evaluate models

# Conditional model
$CMD $MAX --runmode eval --config $CONFIG --inputmap mc_input.yml   --modeltag pointANY --use_conditional 1
$CMD $MAX --runmode eval --config $CONFIG --inputmap mc_input_1.yml --modeltag pointANY --use_conditional 1
$CMD $MAX --runmode eval --config $CONFIG --inputmap mc_input_2.yml --modeltag pointANY --use_conditional 1


# Single points matched model
$CMD $MAX --runmode eval --config $CONFIG --inputmap mc_input_1.yml --modeltag point1
$CMD $MAX --runmode eval --config $CONFIG --inputmap mc_input_2.yml --modeltag point2


# Single points unmatched model
$CMD $MAX --runmode eval --config $CONFIG --inputmap mc_input_1.yml --modeltag point2
$CMD $MAX --runmode eval --config $CONFIG --inputmap mc_input_2.yml --modeltag point1

