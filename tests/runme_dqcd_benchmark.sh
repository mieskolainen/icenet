#!/bin/sh
#
# Execute different training and evaluation runs for DQCD
#
# Run with: source runme.sh

# --------------------
## Process input

DATAPATH="/home/user/travis-stash/input/icedqcd"
CMD="python analysis/dqcd.py"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

$CMD $MAX --runmode genesis --config tune0.yml --inputmap mc_input.yml   --datapath $DATAPATH
$CMD $MAX --runmode genesis --config tune0.yml --inputmap mc_input_1.yml --datapath $DATAPATH
$CMD $MAX --runmode genesis --config tune0.yml --inputmap mc_input_2.yml --datapath $DATAPATH


# --------------------
## Train models

$CMD $MAX --runmode train --config tune0.yml --inputmap mc_input.yml   --modeltag pointANY --use_conditional 1
$CMD $MAX --runmode train --config tune0.yml --inputmap mc_input_1.yml --modeltag point1
$CMD $MAX --runmode train --config tune0.yml --inputmap mc_input_2.yml --modeltag point2


# --------------------
## Evaluate models

$CMD $MAX --runmode eval --config tune0.yml --inputmap mc_input.yml   --modeltag pointANY --use_conditional 1
$CMD $MAX --runmode eval --config tune0.yml --inputmap mc_input_1.yml --modeltag pointANY --use_conditional 1
$CMD $MAX --runmode eval --config tune0.yml --inputmap mc_input_2.yml --modeltag pointANY --use_conditional 1


# Single points matched
$CMD $MAX --runmode eval --config tune0.yml --inputmap mc_input_1.yml --modeltag point1
$CMD $MAX --runmode eval --config tune0.yml --inputmap mc_input_2.yml --modeltag point2


# Single points unmatched
$CMD $MAX --runmode eval --config tune0.yml --inputmap mc_input_1.yml --modeltag point2
$CMD $MAX --runmode eval --config tune0.yml --inputmap mc_input_2.yml --modeltag point1
