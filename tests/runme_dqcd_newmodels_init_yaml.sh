#!/bin/sh
#
# Generate dynamic YAML files
#
# Use * or other glob wildcards for filenames
#
# Run with: source runme.sh

#source $HOME/setconda.sh
conda activate icenet

ICEPATH="/vols/cms/khl216/icenet_new_models/icenet"
cd $ICEPATH
echo "$(pwd)"
source $ICEPATH/setenv.sh

# Training
python configs/dqcd/include/ymlgen.py --process 'QCD'    --filerange '[0-50]'      --outputfile configs/dqcd/include/QCD_newmodels.yml
python configs/dqcd/include/ymlgen.py --process 'scenarioA' --filerange '[0-20]'      --outputfile configs/dqcd/include/scenarioA.yml

# Deployment
python configs/dqcd/include/ymlgen.py --process 'QCD'    --filerange '[51-100000]' --outputfile configs/dqcd/include/QCD_newmodels_deploy.yml
python configs/dqcd/include/ymlgen.py --process 'scenarioA' --filerange '[21-100000]' --outputfile configs/dqcd/include/scenarioA_deploy.yml



