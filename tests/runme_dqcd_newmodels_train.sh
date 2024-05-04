#!/bin/sh
#
# Execute training and evaluation for the DQCD analysis
#
# Remember to execute first: runme_dqcd_newmodels_init_yaml.sh (only once, and just once)

source $HOME/setconda.sh
conda activate icenet

ICEPATH="/vols/cms/mmieskol/icenet"
cd $ICEPATH
echo "$(pwd)"
source $ICEPATH/setenv.sh

CONFIG="tune0_new.yml"
DATAPATH="/vols/cms/khl216"

CONDITIONAL=0
MAX=5000000    # Tune according to maximum CPU RAM available

python analysis/dqcd.py --runmode genesis  --maxevents $MAX --inputmap mc_map__scenarioA_all.yml --config $CONFIG --datapath $DATAPATH
python analysis/dqcd.py --runmode train    --maxevents $MAX --inputmap mc_map__scenarioA_all.yml --modeltag scenarioA_all --config $CONFIG --datapath $DATAPATH --use_conditional $CONDITIONAL
python analysis/dqcd.py --runmode eval     --maxevents $MAX --inputmap mc_map__scenarioA_all.yml --modeltag scenarioA_all --config $CONFIG --datapath $DATAPATH --use_conditional $CONDITIONAL
python analysis/dqcd.py --runmode optimize --maxevents $MAX --inputmap mc_map__scenarioA_all.yml --modeltag scenarioA_all --config $CONFIG --datapath $DATAPATH --use_conditional $CONDITIONAL
