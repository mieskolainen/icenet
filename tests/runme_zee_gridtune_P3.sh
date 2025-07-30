#!/bin/sh
# 
# Execute final training with 'Mixed Setup'
#
# m.mieskolainen@imperial.ac.uk, 2025

if [ -z "$DATAPATH" ]; then
    echo "Error: DATAPATH environment variable is not set."
    echo "Please set it, e.g.: export DATAPATH=/vols/cms/files"
    return
fi

if [ -z "$CONFIG" ]; then
    echo "Error: CONFIG environment variable is not set."
    echo "Please set it, e.g.: export CONFIG=tune0_EB"
    return
fi

if [ -z "$MODELTAG" ]; then
    echo "Error: MODELTAG environment variable is not set."
    echo "Please set it, e.g.: export MODELTAG=GRIDTUNE-P3"
    return
fi

if [ -z "$maxevents" ]; then
    echo "Error: maxevents environment variable is not set."
    echo "Please set it, e.g.: export maxevents=1000000000"
    return
fi

MAX="--maxevents $maxevents"

# Step M1. Produce output (cache) files with all sign events
# N.B. using tag '--compute 0' does only the steps we need here and skips other
python analysis/zee.py --runmode genesis $MAX --modeltag ${MODELTAG} --config ${CONFIG}.yml --datapath $DATAPATH --hash_genesis M1-${CONFIG}                                         --run_id M1-final --supertune "drop_negative=False"
python analysis/zee.py --runmode train   $MAX --modeltag ${MODELTAG} --config ${CONFIG}.yml --datapath $DATAPATH --hash_genesis M1-${CONFIG} --hash_post_genesis M1-${CONFIG}__train --run_id M1-final --compute 0
python analysis/zee.py --runmode eval    $MAX --modeltag ${MODELTAG} --config ${CONFIG}.yml --datapath $DATAPATH --hash_genesis M1-${CONFIG} --hash_post_genesis M1-${CONFIG}__eval  --run_id M1-final --compute 0

# Step M2. Default training with only positive sign events
python analysis/zee.py --runmode genesis $MAX --modeltag ${MODELTAG} --config ${CONFIG}.yml --datapath $DATAPATH --hash_genesis M2-${CONFIG}                                         --run_id M2-final
python analysis/zee.py --runmode train   $MAX --modeltag ${MODELTAG} --config ${CONFIG}.yml --datapath $DATAPATH --hash_genesis M2-${CONFIG} --hash_post_genesis M2-${CONFIG}__train --run_id M2-final
python analysis/zee.py --runmode eval    $MAX --modeltag ${MODELTAG} --config ${CONFIG}.yml --datapath $DATAPATH --hash_genesis M2-${CONFIG} --hash_post_genesis M2-${CONFIG}__eval  --run_id M2-final

# Step M3. Evaluation with all sign events
python analysis/zee.py --runmode eval    $MAX --modeltag ${MODELTAG} --config ${CONFIG}.yml --datapath $DATAPATH --hash_genesis M1-${CONFIG} --hash_post_genesis M1-${CONFIG}__eval --run_id M2-final --evaltag M3__mixed-eval
