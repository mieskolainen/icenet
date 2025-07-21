#!/bin/bash
#
# (Do not modify this file!)
# 
# Condor submission with first init job and then
# an array job once that is finished. Emulating DAGMan without using it.
#
# This file expects variable `TASK_SCRIPT` set externally. See README.md.
# 
# m.mieskolainen@imperial.ac.uk, 2025

mkdir logs -p

# === Check for required environment variable ===
if [ -z "$TASK_SCRIPT" ]; then
    echo "Error: TASK_SCRIPT environment variable is not set."
    echo "Please set it, e.g.: export TASK_SCRIPT=gridtune_task_EB.sh"
    return
fi

# Fixed
INIT_JOB="gridtune_init.job"
ARRAY_JOB="gridtune_array.job"
PERIOD=15

# === Extract clean identifier from TASK_SCRIPT ===
TASK_BASENAME=$(basename "${TASK_SCRIPT}" .sh)
TASK_NAME="${TASK_BASENAME//[^a-zA-Z0-9_]/_}"  # Sanitize

# === Create unique temporary job files ===
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

TEMP_FILE_INIT="tmp__${TASK_NAME}_${TIMESTAMP}_${INIT_JOB}"
TEMP_FILE_ARRAY="tmp__${TASK_NAME}_${TIMESTAMP}_${ARRAY_JOB}"

sed "s|^executable\s*=\s*task\.sh|executable = ${TASK_SCRIPT}|" "$INIT_JOB" > "$TEMP_FILE_INIT"
sed "s|^executable\s*=\s*task\.sh|executable = ${TASK_SCRIPT}|" "$ARRAY_JOB" > "$TEMP_FILE_ARRAY"

chmod +x *.sh

# === Submit the first init job === 
echo "Submitting init job"
FIRST_JOB_ID=$(condor_submit $TEMP_FILE_INIT | awk '/submitted to cluster/ {print int($6)}')
echo " "
cat $TEMP_FILE_INIT

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

# === Submit the array job === 
echo ""
echo "Submitting array job"

condor_submit $TEMP_FILE_ARRAY
echo " "
cat $TEMP_FILE_ARRAY
