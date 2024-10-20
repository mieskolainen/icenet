#!/bin/sh
#
# Execute training and evaluation for the ZEE
#
# Run with: maxevents=10000; source tests/runme.sh

#DATAPATH="/vols/cms/pfk18/phd/hgg/Jul23/NN21July/N/validations/outputs/Csplit_Jsamp/files"
DATAPATH="./actions-stash/input/icezee"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

for CONFIG in "tune0_EEm" "tune0_EEp" # "tune0_EB" 
do
  python analysis/zee.py --runmode "genesis" $MAX --config "${CONFIG}.yml" --datapath $DATAPATH
  python analysis/zee.py --runmode "train"   $MAX --config "${CONFIG}.yml" --datapath $DATAPATH
  python analysis/zee.py --runmode "eval"    $MAX --config "${CONFIG}.yml" --datapath $DATAPATH --evaltag "mytest"
done
