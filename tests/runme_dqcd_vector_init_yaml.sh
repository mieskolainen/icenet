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
python configs/dqcd/include/ymlgen.py --paramera 'old' --process 'QCD'    --filerange '[0-50]'      --outputfile configs/dqcd/include/QCD.yml
python configs/dqcd/include/ymlgen.py --paramera 'old' --process 'vector' --filerange '[0-10]'      --outputfile configs/dqcd/include/vector.yml

# Deployment
python configs/dqcd/include/ymlgen.py --paramera 'old' --process 'QCD'    --filerange '[51-100000]' --outputfile configs/dqcd/include/QCD_deploy.yml
python configs/dqcd/include/ymlgen.py --paramera 'old' --process 'vector' --filerange '[11-100000]' --outputfile configs/dqcd/include/vector_deploy.yml
python configs/dqcd/include/ymlgen.py --paramera 'old' --process 'data-B' --filerange '*'           --outputfile configs/dqcd/include/data-B_deploy.yml
python configs/dqcd/include/ymlgen.py --paramera 'old' --process 'data-C' --filerange '*'           --outputfile configs/dqcd/include/data-C_deploy.yml
python configs/dqcd/include/ymlgen.py --paramera 'old' --process 'data-D' --filerange '*'           --outputfile configs/dqcd/include/data-D_deploy.yml


