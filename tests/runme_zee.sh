#!/bin/sh
#
# Execute training and evaluation for the ZEE
#
# Run with: source runme.sh

#DATAPATH="/vols/cms/pfk18/phd/hgg/Jul23/NN21July/N/validations/outputs/Csplit_Jsamp/files"
DATAPATH="./travis-stash/input/icezee"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

for CONFIG in "tune0_EEp.yml" # "tune0_EB.yml" "tune0_EEm.yml" 
do
  python analysis/zee.py --runmode "genesis" $MAX --config $CONFIG --datapath $DATAPATH
  python analysis/zee.py --runmode "train"   $MAX --config $CONFIG --datapath $DATAPATH
  python analysis/zee.py --runmode "eval"    $MAX --config $CONFIG --datapath $DATAPATH --evaltag "mytest"
done
