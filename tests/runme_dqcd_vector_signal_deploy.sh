#!/bin/sh
#
# Execute distributed deployment for the DQCD analysis
# Run with: source runme.sh

CONFIG="tune0.yml"
#DATAPATH="/home/user/travis-stash/input/icedqcd"
DATAPATH="/vols/cms/mc3909"

# Grid (distributed) processing (set per node via export GRID_ID=0; export GRID_NODES=10)
#GRID_ID=0
#GRID_NODES=1

python analysis/dqcd_deploy.py --use_conditional 1 --inputmap 'include/vector_deploy.yml' --modeltag vector_all --grid_id $GRID_ID --grid_nodes $GRID_NODES --config $CONFIG --datapath $DATAPATH
