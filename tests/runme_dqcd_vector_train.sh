#!/bin/sh
#
# Execute training and evaluation for the DQCD analysis
#
# Remember to execute first: runme_dqcd_vector_init_yaml.sh (only once, and just once)

source $HOME/setconda.sh
conda activate icenet

ICEPATH="/vols/cms/mmieskol/icenet"
cd $ICEPATH
echo "$(pwd)"
source $ICEPATH/setenv.sh

CONFIG="tune0-dxy.yml"
DATAPATH="/vols/cms/mc3909"

CONDITIONAL=1
MAX=1000000 # Tune according to maximum CPU RAM available

for CONFIG in "tune0-dxy.yml" "tune0-dlen.yml" "tune0-chi2.yml"
do
    rm $ICEPATH/output/dqcd/processed_data_*

    python analysis/dqcd.py --runmode genesis  --maxevents $MAX --inputmap mc_map__vector_all.yml --modeltag vector_all --config $CONFIG --datapath $DATAPATH
    python analysis/dqcd.py --runmode train    --maxevents $MAX --inputmap mc_map__vector_all.yml --modeltag vector_all --config $CONFIG --datapath $DATAPATH --use_conditional $CONDITIONAL
    python analysis/dqcd.py --runmode eval     --maxevents $MAX --inputmap mc_map__vector_all.yml --modeltag vector_all --config $CONFIG --datapath $DATAPATH --use_conditional $CONDITIONAL
    python analysis/dqcd.py --runmode optimize --maxevents $MAX --inputmap mc_map__vector_all.yml --modeltag vector_all --config $CONFIG --datapath $DATAPATH --use_conditional $CONDITIONAL
done
