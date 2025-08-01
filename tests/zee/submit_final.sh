#!/bin/bash
#
# (Do not modify this file!)
# 
# Condor submission
#
# This file expects variable `CONFIG` and `TASK_SCRIPT` set externally. See README.md.
# 
# m.mieskolainen@imperial.ac.uk, 2025

mkdir -p logs

# === Check for required environment variable ===
if [ -z "$TASK_SCRIPT" ]; then
    echo "Error: TASK_SCRIPT environment variable is not set."
    echo "Please set it, e.g.: export TASK_SCRIPT=gridtune_task_EB.sh"
    return 0
fi

if [ -z "$CONFIG" ]; then
    echo "Error: CONFIG environment variable is not set."
    echo "Please set it, e.g.: export CONFIG=tune0_EB"
    return 0
fi

if [ -z "$maxevents" ]; then
    echo "Error: maxevents environment variable is not set."
    echo "Please set it, e.g.: export maxevents=500000"
    return 0
fi

# === Fixed Parameters ===
INIT_JOB="gridtune_init.job"

# === Extract clean identifier from TASK_SCRIPT ===
TASK_BASENAME=$(basename "${TASK_SCRIPT}" .sh)
TASK_NAME="${TASK_BASENAME//[^a-zA-Z0-9_]/_}"  # Sanitize

# === Create unique temporary job files ===
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

TEMP_FILE="tmp__${TASK_NAME}_${TIMESTAMP}_$(basename "${INIT_JOB}")"

# Replace executable line
sed "s|^executable\s*=\s*task\.sh|executable = ${TASK_SCRIPT}|" "$INIT_JOB" \
| sed "/^environment\s*=/d" \
| sed "1ienvironment = \"CONFIG=${CONFIG} maxevents=${maxevents}\"" \
> "$TEMP_FILE"

chmod +x *.sh

# === Print out ===
echo "Created job: $TEMP_FILE"
echo " "
cat $TEMP_FILE
echo " "

# === Submit ===
condor_submit "$TEMP_FILE"

