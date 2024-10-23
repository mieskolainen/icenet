#!/bin/bash

# Condor submission with first init job and then
# an array job once that is finished
#
# Emulating DAGMan without using it.
#
# Run with: source submit.sh (execute the command in the same folder)
#
# m.mieskolainen@imperial.ac.uk, 2024

mkdir logs -p

ICEPATH="/vols/cms/mmieskol/icenet"

TASK_SCRIPT="gridtune_task.sh"
INIT_JOB="gridtune_init.job"
ARRAY_JOB="gridtune_array.job"
PERIOD=15

# Submit the first job
echo "Submitting init job"
FIRST_JOB_ID=$(condor_submit $INIT_JOB | awk '/submitted to cluster/ {print int($6)}')
echo " "
cat $INIT_JOB

# Check if job submission was successful
if [[ -z "$FIRST_JOB_ID" ]]; then
    echo "Error: Failed to submit the first job"
    exit 1
fi

sleep 5

echo "First job with ID = ${FIRST_JOB_ID}"
echo "Waiting for the first job to finish"

# Initialize start time for cumulative waiting
start_time=$(date +%s)

while true; do
    # Check if the job is still in the queue
    job_status=$(condor_q $FIRST_JOB_ID -format "%d" JobStatus 2>/dev/null)

    # If condor_q returns nothing, check condor_history
    if [ -z "$job_status" ]; then
        # Job is no longer in the queue, check the history
        job_status=$(condor_history $FIRST_JOB_ID -limit 1 -format "%d" JobStatus 2>/dev/null)
        
        # Exit the loop if the job has completed
        if [ "$job_status" -eq "4" ]; then
            echo "Job completed successfully."
            break
        else
            echo "Job is no longer running but didn't finish as expected -- exit"
            exit 0
        fi
    fi

    # Calculate the cumulative time spent waiting
    current_time=$(date +%s)
    elapsed_time=$((current_time - start_time))
    elapsed_minutes=$((elapsed_time / 60))
    elapsed_seconds=$((elapsed_time % 60))

    # Otherwise, job is still in the queue, and we can wait
    echo "First job is still running (status: $job_status) | Elapsed: ${elapsed_minutes} min and ${elapsed_seconds} sec"
    sleep $PERIOD
done

# Submit the array job
echo ""
echo "Submitting array job"
condor_submit $ARRAY_JOB
echo " "
cat $ARRAY_JOB
