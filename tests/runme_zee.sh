#!/bin/sh
#
# Execute training and evaluation for the ZEE
#
# Run with: source runme.sh

#DATAPATH="/vols/cms/pfk18/phd/hgg/Jul23/NN21July/N/validations/outputs/Csplit_Jsamp/files"
DATAPATH="./travis-stash/input/icezee"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

for CONFIG in "tune0_EEp" # "tune0_EB" "tune0_EEm" 
do
  python analysis/zee.py --runmode "genesis" $MAX --config "${CONFIG}.yml" --datapath $DATAPATH --hash_genesis "minimal_${CONFIG}"
  python analysis/zee.py --runmode "train"   $MAX --config "${CONFIG}.yml" --datapath $DATAPATH --hash_genesis "minimal_${CONFIG}" --hash_post_genesis "minimal_${CONFIG}__train"
  python analysis/zee.py --runmode "eval"    $MAX --config "${CONFIG}.yml" --datapath $DATAPATH --hash_genesis "minimal_${CONFIG}" --hash_post_genesis "minimal_${CONFIG}__eval" --evaltag "mytest"
done
