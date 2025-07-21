#!/bin/sh
#
# (Do not modify this file!)
#
# m.mieskolainen@imperial.ac.uk, 2025

echo "Grid tuning job started"
pwd

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
