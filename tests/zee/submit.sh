#!/bin/bash
#
# (Do not modify this file)
# 
# Condor submission with first init job and then
# an array job once that is finished. Emulating DAGMan without using it.
#
# This file expects variable `TASK_SCRIPT` set externally. See README.md.
# 
# m.mieskolainen@imperial.ac.uk, 2024

mkdir logs -p

# Fixed
INIT_JOB="gridtune_init.job"
ARRAY_JOB="gridtune_array.job"
PERIOD=15

# Replace the line and save to the temporary file
TEMP_FILE="temp_$(basename "${INIT_JOB}")"
sed "s|^executable\s*=\s*task\.sh|executable = ${TASK_SCRIPT}|" $INIT_JOB > $TEMP_FILE

# Submit the first job
echo "Submitting init job"
FIRST_JOB_ID=$(condor_submit $TEMP_FILE | awk '/submitted to cluster/ {print int($6)}')
echo " "
cat $TEMP_FILE
rm $TEMP_FILE

# Check if job submission was successful
if [[ -z "$FIRST_JOB_ID" ]]; then
    echo "Error: Failed to submit the first job"
    exit 1
fi

sleep 3

echo ""
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

# Create temp file
TEMP_FILE="temp_$(basename "${ARRAY_JOB}")"
sed "s|^executable\s*=\s*task\.sh|executable = ${TASK_SCRIPT}|" $ARRAY_JOB > $TEMP_FILE

condor_submit $TEMP_FILE
echo " "
cat $TEMP_FILE
rm $TEMP_FILE
