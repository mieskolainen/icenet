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

    ## Sun Grid Engine array job environment variables
    if [ ${SGE_TASK_ID} ]
    then
        echo "Detected Sun Grid Engine (SGE) grid job"
        export GRID_ID=$(expr ${SGE_TASK_ID} - 1) # - 1
        export GRID_NODES=${SGE_TASK_LAST}
    fi

    ## For HTCondor, create two environment variables
    # in the main steering job and read in those in my_script.sh
    # and define corresponding HTC_PROCESS_ID and HTC_QUEUE_SIZE environment
    # variables.
    
    # ----------------------------------------------
    # Example HTCondor 'array.job' file with 10 jobs
    # ----------------------------------------------
    # 
    # executable  = my_script.sh
    # arguments   = "$(PROCESS) 10"
    # output      = my_script.$(CLUSTER).$(PROCESS).out
    # error       = my_script.$(CLUSTER).$(PROCESS).error
    # log         = my_script.$(CLUSTER).$(PROCESS).log
    # +MaxRuntime = 6500
    # queue 10
    # ----------------------------------------------

    # ----------------------------------------------
    # Example 'my_script.sh' file
    # ----------------------------------------------
    # 
    # export HTC_PROCESS_ID=$1
    # export HTC_QUEUE_SIZE=$2
    # 
    # ... init conda ...
    # 
    # conda activate icenet
    # source setenv.sh
    # 
    # ... more logic ...
    # ----------------------------------------------
    
    if [ ${HTC_PROCESS_ID} ]
    then
        echo "Detected HTCondor (HTC) grid job"
        export GRID_ID=${HTC_PROCESS_ID}
        export GRID_NODES=${HTC_QUEUE_SIZE}
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
