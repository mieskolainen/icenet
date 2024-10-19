#!/bin/sh
#
# Execute different training and evaluation runs for DQCD
#
# Run with: source tests/runme.sh

source $HOME/setconda.sh
conda activate icenet

ICEPATH="/vols/cms/mmieskol/icenet"
cd $ICEPATH
echo "$(pwd)"
source $ICEPATH/setenv.sh

# --------------------
## Process input

CONFIG="tune0.yml"

DATAPATH="/home/user/travis-stash/input/icedqcd"
#DATAPATH="/vols/cms/khl216""

CMD="python analysis/dqcd.py"

if [ ${maxevents+x} ]; then MAX="--maxevents $maxevents"; else MAX=""; fi

# Generate steering YAML for QCD
python configs/dqcd/include/ymlgen.py --process 'QCD'        --filerange '[0-10]'

# Vector
python configs/dqcd/include/ymlgen.py --process 'vector'     --filerange '[150-2000]'


$CMD $MAX --runmode genesis --config $CONFIG --inputmap mc_map__vector_all.yml                       --datapath $DATAPATH
$CMD $MAX --runmode genesis --config $CONFIG --inputmap mc_map__vector_m_10_ctau_10_xiO_1_xiL_1.yml  --datapath $DATAPATH
$CMD $MAX --runmode genesis --config $CONFIG --inputmap mc_map__vector_m_10_ctau_100_xiO_1_xiL_1.yml --datapath $DATAPATH


# --------------------
## Train models

$CMD $MAX --runmode train --config $CONFIG --inputmap mc_map__vector_all.yml                       --modeltag vector_all --use_conditional 1
$CMD $MAX --runmode train --config $CONFIG --inputmap mc_map__vector_m_10_ctau_10_xiO_1_xiL_1.yml  --modeltag vector_m_10_ctau_10_xiO_1_xiL_1
$CMD $MAX --runmode train --config $CONFIG --inputmap mc_map__vector_m_10_ctau_100_xiO_1_xiL_1.yml --modeltag vector_m_10_ctau_100_xiO_1_xiL_1


# --------------------
## Evaluate models

## Conditional model
$CMD $MAX --runmode eval --config $CONFIG --inputmap mc_map__vector_all.yml                       --modeltag vector_all --use_conditional 1
$CMD $MAX --runmode eval --config $CONFIG --inputmap mc_map__vector_m_10_ctau_10_xiO_1_xiL_1.yml  --modeltag vector_all --use_conditional 1
$CMD $MAX --runmode eval --config $CONFIG --inputmap mc_map__vector_m_10_ctau_100_xiO_1_xiL_1.yml --modeltag vector_all --use_conditional 1


## Single points matched model
$CMD $MAX --runmode eval --config $CONFIG --inputmap mc_map__vector_m_10_ctau_10_xiO_1_xiL_1.yml  --modeltag vector_m_10_ctau_10_xiO_1_xiL_1
$CMD $MAX --runmode eval --config $CONFIG --inputmap mc_map__vector_m_10_ctau_100_xiO_1_xiL_1.yml --modeltag vector_m_10_ctau_100_xiO_1_xiL_1


## Single points unmatched model
$CMD $MAX --runmode eval --config $CONFIG --inputmap mc_map__vector_m_10_ctau_10_xiO_1_xiL_1.yml  --modeltag vector_m_10_ctau_100_xiO_1_xiL_1
$CMD $MAX --runmode eval --config $CONFIG --inputmap mc_map__vector_m_10_ctau_100_xiO_1_xiL_1.yml --modeltag vector_m_10_ctau_10_xiO_1_xiL_1

