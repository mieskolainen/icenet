#!/bin/sh
#
# (Do not modify this file!)
#
# m.mieskolainen@imperial.ac.uk, 2025

if [ -z "$TUNESCRIPT" ]; then
    echo "Error: TUNESCRIPT environment variable is not set."
    echo "Please set it, e.g.: export TUNESCRIPT=runme_zee_gridtune_S1.sh"
    return 0
fi

echo "Grid tuning job started with TUNESCRIPT=${TUNESCRIPT}"
pwd

# Init conda
source $SETCONDA
conda activate icenet

# Init icenet
mkdir $ICEPATH/tmp -p
cd $ICEPATH
source $ICEPATH/setenv.sh

# Execute
source $ICEPATH/tests/$TUNESCRIPT

# Create the done file when the job completes
donefile="${ICEPATH}/tmp/icenet_${HTC_CLUSTER_ID}_${HTC_PROCESS_ID}.done"
touch $donefile

echo "Task done, created file: ${donefile}"
