#!/bin/sh
#
# Execute distributed deployment for the DQCD analysis
# Run with: source runme.sh

# Remember to execute first: runme_dqcd_vector_init_yaml.sh (only once, and just once)

CONFIG="tune0.yml"
DATAPATH="/vols/cms/mc3909"

source setenv.sh

python analysis/dqcd_deploy.py --use_conditional 1 --inputmap 'include/vector_deploy.yml' --modeltag vector_all --grid_id $GRID_ID --grid_nodes $GRID_NODES --config $CONFIG --datapath $DATAPATH
