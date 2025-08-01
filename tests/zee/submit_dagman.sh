#!/bin/bash
#
# (Do not modify this file!)
# 
# DAGMan based submission
#
# This file expects variable `TASK_SCRIPT` set externally. See README.md.
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
ARRAY_JOB="gridtune_array.job"

# === Extract clean identifier from TASK_SCRIPT ===
TASK_BASENAME=$(basename "${TASK_SCRIPT}" .sh)
TASK_NAME="${TASK_BASENAME//[^a-zA-Z0-9_]/_}"  # Sanitize

# === Create unique temporary job files ===
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

TEMP_FILE_INIT="tmp__${TASK_NAME}_${TIMESTAMP}_$(basename "${INIT_JOB}")"
TEMP_FILE_ARRAY="tmp__${TASK_NAME}_${TIMESTAMP}_$(basename "${ARRAY_JOB}")"

# Replace executable line
sed "s|^executable\s*=\s*task\.sh|executable = ${TASK_SCRIPT}|" "$INIT_JOB" \
| sed "/^environment\s*=/d" \
| sed "1ienvironment = \"CONFIG=${CONFIG} maxevents=${maxevents}\"" \
> "$TEMP_FILE_INIT"

sed "s|^executable\s*=\s*task\.sh|executable = ${TASK_SCRIPT}|" "$ARRAY_JOB" \
| sed "/^environment\s*=/d" \
| sed "1ienvironment = \"CONFIG=${CONFIG} maxevents=${maxevents}\"" \
> "$TEMP_FILE_ARRAY"

chmod +x *.sh

# === Print out ===
echo "Created INIT job: $TEMP_FILE_INIT"
echo " "
cat $TEMP_FILE_INIT
echo " "

echo "Created ARRAY job: $TEMP_FILE_ARRAY"
echo " "
cat $TEMP_FILE_ARRAY
echo " "

# === Generate unique DAG file ===
DAG_FILE="tmp__${TASK_NAME}_${TIMESTAMP}.dag"

cat <<EOF > "$DAG_FILE"
# Filename: $DAG_FILE
JOB A ${TEMP_FILE_INIT}
JOB B ${TEMP_FILE_ARRAY}

# Make B depend on A finishing successfully
PARENT A CHILD B
EOF

chmod +x *.dag

# === Submit the DAG ===
echo "Submitting DAG: $DAG_FILE"
condor_submit_dag "$DAG_FILE"
