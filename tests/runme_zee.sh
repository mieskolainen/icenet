#!/bin/sh
#
# Execute training and evaluation for the ZEE
#
# Run with: source runme.sh

CONFIG="tune0.yml"
DATAPATH="/vols/cms/pfk18/phd/hgg/Jul23/NN21July/N/validations/outputs/Csplit_Jsamp"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# tee redirect output to both a file and to screen
python analysis/zee.py --runmode "genesis" $MAX --config $CONFIG --datapath $DATAPATH
#python analysis/zee.py --fastplot 1 --runmode "train"   $MAX --config $CONFIG --datapath $DATAPATH
python analysis/zee.py --fastplot 1 --runmode "eval"    $MAX --config $CONFIG --datapath $DATAPATH
