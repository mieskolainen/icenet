#!/bin/sh
#
# Generate dynamic YAML files
#
# Use * or other glob wildcards for filenames
#
# Run with: source filename.sh

#source $HOME/setconda.sh
conda activate icenet

ICEPATH="/vols/cms/mmieskol/icenet"
cd $ICEPATH
echo "$(pwd)"
source $ICEPATH/setenv.sh

# Training
python configs/dqcd/include/ymlgen.py --paramera 'new' --process 'QCD'       --filerange '[0-50]'      --outputfile configs/dqcd/include/QCD_new.yml
python configs/dqcd/include/ymlgen.py --paramera 'new' --process 'scenarioA' --filerange '[0-20]'      --outputfile configs/dqcd/include/scenarioA.yml

# Deployment
python configs/dqcd/include/ymlgen.py --paramera 'new' --process 'QCD'       --filerange '[51-100000]' --outputfile configs/dqcd/include/QCD_new_deploy.yml
python configs/dqcd/include/ymlgen.py --paramera 'new' --process 'scenarioA' --filerange '[21-100000]' --outputfile configs/dqcd/include/scenarioA_deploy.yml

