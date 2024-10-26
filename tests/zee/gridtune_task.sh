#!/bin/sh
#
# (modify this file for your setup)
#
# GPU grid tuning task 
#
# m.mieskolainen@imperial.ac.uk, 2024

echo "Grid tuning job started"
pwd

# ** icenet/setenv.sh uses these **
export HTC_PROCESS_ID=$1
export HTC_QUEUE_SIZE=$2
export HTC_CLUSTER_ID=$3

# -------------------------------
# MODIFY THESE !

SETCONDA="/home/hep/mmieskol/setconda.sh"
ICEPATH="/vols/cms/mmieskol/icenet"

DATAPATH="/vols/cms/pfk18/phd/hgg/Jul23/NN21July/N/validations/outputs/Csplit_Jsamp/files"
CONFIG="tune0_EB"
maxevents=500000

BETA_ARRAY=(0.0 0.0025 0.005 0.01 0.02 0.04)
SIGMA_ARRAY=(0.0 0.025 0.05 0.1 0.2)

#SWD_VAR="[.*]" # all
SWD_VAR="['fixedGridRhoAll', 'probe_eta', 'probe_pt']"
# -------------------------------

# Init conda
source $SETCONDA
conda activate icenet

# Init icenet
mkdir $ICEPATH/tmp -p
cd $ICEPATH
source $ICEPATH/setenv.sh

# Execute
source $ICEPATH/tests/runme_zee_gridtune.sh

# Create the done file when the job completes
donefile="${ICEPATH}/tmp/icenet_${HTC_CLUSTER_ID}_${HTC_PROCESS_ID}.done"
touch $donefile

echo "Task done, created file: ${donefile}"
