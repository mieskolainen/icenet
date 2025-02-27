#!/bin/sh
#
# Execute training and evaluation for the ZEE
#
# Run with: maxevents=10000; source tests/runme.sh

#DATAPATH="/vols/cms/pfk18/icenet_files/processed_29_nov_24"
DATAPATH="./actions-stash/input/icezee"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

for CONFIG in "tune0_EEm" "tune0_EEp" # "tune0_EB" 
do
  python analysis/zee.py --runmode genesis $MAX --config ${CONFIG}.yml --datapath $DATAPATH
  python analysis/zee.py --runmode train   $MAX --config ${CONFIG}.yml --datapath $DATAPATH
  python analysis/zee.py --runmode eval    $MAX --config ${CONFIG}.yml --datapath $DATAPATH --evaltag mytest
done

#--supertune "models.iceboost_swd.model_param.objective='custom:binary_cross_entropy:hessian:iterative:0.9'"
#--supertune "models.iceboost_swd.SWD_param.var=['fixedGridRhoAll', 'probe_eta', 'probe_pt']"
#--supertune "models.iceboost_swd.readmode=-2"
