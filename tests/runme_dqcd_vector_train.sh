#!/bin/sh
#
# Execute training and evaluation for the DQCD analysis
#
# Remember to execute first: runme_dqcd_vector_init_yaml.sh (only once, and just once)

CONFIG="tune0.yml"
DATAPATH="/vols/cms/mc3909"

CONDITIONAL=1
MAX=30000 # Tune according to maximum CPU RAM available

source setenv.sh

python analysis/dqcd.py --runmode genesis  --maxevents $MAX --inputmap mc_map__vector_all.yml --modeltag vector_all --config $CONFIG --datapath $DATAPATH
python analysis/dqcd.py --runmode train    --maxevents $MAX --inputmap mc_map__vector_all.yml --modeltag vector_all --config $CONFIG --datapath $DATAPATH --use_conditional $CONDITIONAL
python analysis/dqcd.py --runmode eval     --maxevents $MAX --inputmap mc_map__vector_all.yml --modeltag vector_all --config $CONFIG --datapath $DATAPATH --use_conditional $CONDITIONAL
python analysis/dqcd.py --runmode optimize --maxevents $MAX --inputmap mc_map__vector_all.yml --modeltag vector_all --config $CONFIG --datapath $DATAPATH --use_conditional $CONDITIONAL
