#!/bin/sh
#
# Execute training and evaluation for the DQCD analysis
#
# Run with: source runme.sh

CONFIG="tune0.yml"
#DATAPATH="/home/user/travis-stash/input/icedqcd"
DATAPATH="/vols/cms/mc3909"

# Set system memory limits
ulimit -s unlimited
ulimit -l unlimited
ulimit -v unlimited

# Use * or other glob wildcards for filenames
# tee redirect output to both a file and to screen

#python configs/dqcd/include/ymlgen.py --process 'QCD' --filerange '[11-100000]' --outputfile configs/dqcd/include/QCD_deploy.yml
#python analysis/dqcd_deploy.py --use_conditional 1 --inputmap 'include/QCD_deploy.yml' --modeltag vector_all --config $CONFIG --datapath $DATAPATH

#python configs/dqcd/include/ymlgen.py --process 'vector' --filerange '[0-149]' --outputfile configs/dqcd/include/vector_deploy.yml
#python analysis/dqcd_deploy.py --use_conditional 1 --inputmap 'include/vector_deploy.yml' --modeltag vector_all --config $CONFIG --datapath $DATAPATH

python configs/dqcd/include/ymlgen.py --process 'data' --filerange '[0-100000]' --outputfile configs/dqcd/include/data_deploy.yml
python analysis/dqcd_deploy.py --use_conditional 1 --inputmap 'include/data_deploy.yml' --modeltag vector_all --config $CONFIG --datapath $DATAPATH
