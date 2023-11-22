#!/bin/sh
#
# Execute training and evaluation for the DQCD analysis
#
# Remember to execute first: runme_dqcd_vector_init_yaml.sh (only once, and just once)

#source $HOME/setconda.sh
__conda_setup="$('/vols/cms/khl216/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/vols/cms/khl216/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/vols/cms/khl216/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/vols/cms/khl216/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate icenet

ICEPATH="/vols/cms/khl216/icenet_new_models/icenet"
cd $ICEPATH
echo "$(pwd)"
source $ICEPATH/setenv.sh

CONFIG="tune0-dxy.yml"
DATAPATH="/vols/cms/khl216"

CONDITIONAL=0
MAX=1000000 # Tune according to maximum CPU RAM available

for CONFIG in "tune0-dxy.yml" "tune0-dlen.yml" "tune0-chi2.yml"
do
    rm $ICEPATH/output/dqcd/processed_data_*

    python analysis/dqcd.py --runmode genesis  --maxevents $MAX --inputmap mc_map__scenarioA_all.yml --modeltag scenarioA_all --config $CONFIG --datapath $DATAPATH
    python analysis/dqcd.py --runmode train    --maxevents $MAX --inputmap mc_map__scenarioA_all.yml --modeltag scenarioA_all --config $CONFIG --datapath $DATAPATH --use_conditional $CONDITIONAL
    python analysis/dqcd.py --runmode eval     --maxevents $MAX --inputmap mc_map__scenarioA_all.yml --modeltag scenarioA_all --config $CONFIG --datapath $DATAPATH --use_conditional $CONDITIONAL
    python analysis/dqcd.py --runmode optimize --maxevents $MAX --inputmap mc_map__scenarioA_all.yml --modeltag scenarioA_all --config $CONFIG --datapath $DATAPATH --use_conditional $CONDITIONAL
done
