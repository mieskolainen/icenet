#!/bin/sh
#
# GPU grid tuning task

echo "Grid tuning job started"
pwd

ICEPATH="/vols/cms/mmieskol/icenet"

# ** icenet/setenv.sh uses these **
export HTC_PROCESS_ID=$1
export HTC_QUEUE_SIZE=$2
export HTC_CLUSTER_ID=$3

# Init conda
source /home/hep/mmieskol/setconda.sh
conda activate icenet

# Init icenet
mkdir $ICEPATH/tmp -p
cd $ICEPATH
source $ICEPATH/setenv.sh

# Execute
DATAPATH="/vols/cms/pfk18/phd/hgg/Jul23/NN21July/N/validations/outputs/Csplit_Jsamp/files"
CONFIG="tune0_EB"
maxevents=100000
source /vols/cms/mmieskol/icenet/tests/runme_zee_gridtune.sh

# Create the done file when the job completes
donefile="${ICEPATH}/tmp/icenet_${HTC_CLUSTER_ID}_${HTC_PROCESS_ID}.done"
touch $donefile

echo "Task done, created file: ${donefile}"
