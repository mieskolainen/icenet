#!/bin/sh
#
# Execute distributed deployment for the DQCD analysis
# Run with: source runme.sh

# Remember to execute first: runme_dqcd_vector_init_yaml.sh (only once, and just once)

source $HOME/setconda.sh
conda activate icenet

ICEPATH="/vols/cms/mmieskol/icenet"
cd $ICEPATH
echo "$(pwd)"
source $ICEPATH/setenv.sh

CONFIG="tune0.yml"
DATAPATH="/vols/cms/mc3909"

python analysis/dqcd_deploy.py --use_conditional 1 --inputmap 'include/data-D_deploy.yml' --modeltag vector_all --grid_id $GRID_ID --grid_nodes $GRID_NODES --config $CONFIG --datapath $DATAPATH
