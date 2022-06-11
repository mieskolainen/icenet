#!/bin/sh
#
# Execute "deep batched" training and evaluation for electron ID
#
# Run with: source runme.sh

CONFIG="tune0"

DATAPATH="./travis-stash/input/iceid"

# Use * or other glob wildcards for filenames

python ./analysis/eid_deep_train.py --config $CONFIG --datapath $DATAPATH --datasets "output_[0-99].root" # output_{0,1}
python ./analysis/eid_eval.py --config $CONFIG --datapath $DATAPATH --datasets "output_0.root"

