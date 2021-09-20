#!/bin/sh
#
# Execute training and evaluation for electron ID
#
# Run with: source runme.sh

CONFIG="tune0"
DATAPATH="/home/user/cernbox/HLT_electrons"


# Use * or other glob wildcards for filenames
mkdir ./figs/trg/$CONFIG -p # for output ascii dump

python ./analysis/trg_train.py --config $CONFIG --datapath $DATAPATH --datasets "none" > ./figs/trg/$CONFIG/train_output.txt
python ./analysis/trg_eval.py  --config $CONFIG --datapath $DATAPATH --datasets "none" > ./figs/trg/$CONFIG/eval_output.txt

