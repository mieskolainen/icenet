#!/bin/sh
#
# Generate dynamic YAML files
#
# Use * or other glob wildcards for filenames
#
# Run with: source runme.sh

# Training
python configs/dqcd/include/ymlgen.py --process 'QCD'    --filerange '[0-10]'
python configs/dqcd/include/ymlgen.py --process 'vector' --filerange '[0-5]'

# Deployment
python configs/dqcd/include/ymlgen.py --process 'QCD'    --filerange '[11-100000]' --outputfile configs/dqcd/include/QCD_deploy.yml
python configs/dqcd/include/ymlgen.py --process 'vector' --filerange '[6-100000]'  --outputfile configs/dqcd/include/vector_deploy.yml
python configs/dqcd/include/ymlgen.py --process 'data'   --filerange '*'           --outputfile configs/dqcd/include/data_deploy.yml
