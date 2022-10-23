#!/bin/sh
#
# Execute training and evaluation for the DQCD analysis
#
# Run with: source runme.sh

CONFIG="tune0.yml"
#DATAPATH="/home/user/travis-stash/input/icedqcd"
DATAPATH="/vols/cms/mc3909"

CONDITIONAL=1

#mkdir "figs/dqcd/config-[$CONFIG]" -p # for output ascii dump

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# Set system memory limits
ulimit -s unlimited
ulimit -l unlimited
ulimit -v unlimited

# Use * or other glob wildcards for filenames
# tee redirect output to both a file and to screen

# Generate steering YAML for QCD
python configs/dqcd/include/ymlgen.py --process 'QCD'        --filerange '[0-10]'

# Darkphoton
python configs/dqcd/include/ymlgen.py --process 'darkphoton' --filerange '[150-2000]'

python analysis/dqcd.py --runmode genesis  $MAX --inputmap mc_map__darkphoton_all.yml --modeltag darkphoton_all --config $CONFIG --datapath $DATAPATH
python analysis/dqcd.py --runmode train    $MAX --inputmap mc_map__darkphoton_all.yml --modeltag darkphoton_all --config $CONFIG --datapath $DATAPATH --use_conditional $CONDITIONAL
python analysis/dqcd.py --runmode eval     $MAX --inputmap mc_map__darkphoton_all.yml --modeltag darkphoton_all --config $CONFIG --datapath $DATAPATH --use_conditional $CONDITIONAL
python analysis/dqcd.py --runmode optimize $MAX --inputmap mc_map__darkphoton_all.yml --modeltag darkphoton_all --config $CONFIG --datapath $DATAPATH --use_conditional $CONDITIONAL
