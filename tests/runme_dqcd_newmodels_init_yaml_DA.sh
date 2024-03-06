#!/bin/sh
#
# Generate dynamic YAML files
#
# Use * or other glob wildcards for filenames
#
# Run with: source runme.sh

source $HOME/setconda.sh
conda activate icenet

ICEPATH="/vols/cms/mmieskol/icenet"
cd $ICEPATH
echo "$(pwd)"
source $ICEPATH/setenv.sh

# Training
python configs/dqcd/include/ymlgen.py --process 'QCD'       --filerange '[0-50]' --outputfile configs/dqcd/include/QCD_newmodels.yml
python configs/dqcd/include/ymlgen.py --process 'scenarioA' --filerange '[0-20]' --outputfile configs/dqcd/include/scenarioA.yml

# Training (Domain Adaptation)
python configs/dqcd/include/ymlgen.py --process 'data-D'    --filerange '[0-10]'  --outputfile configs/dqcd/include/data_DA.yml

# Deployment
python configs/dqcd/include/ymlgen.py --process 'QCD'       --filerange '[51-100000]' --outputfile configs/dqcd/include/QCD_newmodels_deploy.yml
python configs/dqcd/include/ymlgen.py --process 'scenarioA' --filerange '[21-100000]' --outputfile configs/dqcd/include/scenarioA_deploy.yml
