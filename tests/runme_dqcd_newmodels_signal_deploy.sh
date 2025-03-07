#!/bin/sh
#
# Execute distributed deployment for the DQCD analysis
#
# Run with: source tests/runme.sh

# Remember to execute first: runme_dqcd_newmodels_init_yaml.sh (only once, and just once)

source $HOME/setconda.sh
conda activate icenet

ICEPATH="/vols/cms/mmieskol/icenet"
cd $ICEPATH
echo "$(pwd)"
source $ICEPATH/setenv.sh

CONFIG="tune0_new.yml"
DATAPATH="/vols/cms/khl216"
CONDITIONAL=1

python analysis/dqcd_deploy.py --runmode deploy --use_conditional $CONDITIONAL --inputmap 'include/scenarioA_deploy.yml' --modeltag scenarioA_all --grid_id $GRID_ID --grid_nodes $GRID_NODES --config $CONFIG --datapath $DATAPATH
