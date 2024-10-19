#!/bin/sh
# 
# Execute training and evaluation for the electron ID
# 
# Run with: maxevents=10000; source tests/runme.sh

ICEPATH=/home/hep/rjb3/work/icenet
cd $ICEPATH
echo "$(pwd)"
#echo "superclean ..."; rm -f -r $ICEPATH/output/*; rm -f -r $ICEPATH/figs/*; rm -f -r $ICEPATH/checkpoint/*; rm -f -r $ICEPATH/tmp/*
source $ICEPATH/setconda.sh
conda activate icenet
source $ICEPATH/setenv.sh

CONFIG="tune0.yml"
DATAPATH="/vols/cms/bainbrid/ntuples/icenet/"
#MCMAP="map_mc_test.yml" # uses actions-stash/input/icebrem
#MCMAP="map_mc.yml" # local-stash
MCMAP="map_mc_large.yml" # large-stash
#DATASETS="output_*.root"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

python analysis/brem.py --runmode genesis --config $CONFIG --datapath $DATAPATH --inputmap $MCMAP $MAX
python analysis/brem.py --runmode train   --config $CONFIG --datapath $DATAPATH --inputmap $MCMAP $MAX
python analysis/brem.py --runmode eval    --config $CONFIG --datapath $DATAPATH --inputmap $MCMAP $MAX
