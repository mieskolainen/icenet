#!/bin/sh
#

# Set library paths
# (use after activating the conda environment)

if [ ${ICENET_ENV} ]
then
    echo '** ICENET environment already set. Run with: ICENET_ENV=; source setenv.sh to reset **'

else
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

    echo $LD_LIBRARY_PATH

    echo "Before:"
    ulimit -a

    # Set system memory limits
    ulimit -s unlimited  # stack
    ulimit -u 65536      # num of processes (for Sun Grid Engine use)
    #ulimit -v unlimited # virtual memory

    echo ""
    echo "After:"
    ulimit -a
    echo ""

    export GRID_ID=0
    export GRID_NODES=1

    # Sun Grid Engine array job environment variables
    if [ ${SGE_TASK_ID} ]
    then
        export GRID_ID=$(expr ${SGE_TASK_ID} - 1)
        export GRID_NODES=${SGE_TASK_LAST}
    fi

    CWD=`pwd`
    TUNE_RESULT_DIR="$CWD/tmp/ray/GRID_ID_${GRID_ID}"

    echo "GRID_ID=${GRID_ID}"
    echo "GRID_NODES=${GRID_NODES}"
    echo "TUNE_RESULT_DIR=${TUNE_RESULT_DIR}"
    echo ""
    
    export ICENET_ENV=True

    echo '** ICENET: New environment variables set **'
fi
