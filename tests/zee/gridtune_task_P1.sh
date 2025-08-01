#!/bin/sh
# 
# (Modify this file!)
# 
# m.mieskolainen@imperial.ac.uk, 2025

SETCONDA="/home/hep/mmieskol/setconda.sh"
ICEPATH="/vols/cms/mmieskol/icenet"

DATAPATH="/vols/cms/pfk18/icenet_files/processed_20Feb2025"
MODELTAG="GRIDTUNE-P1"

# === Check for required environment variable ===
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

# ---------------------------------------
# Hyperparameters (N-dim cartesian product)

LR_ARRAY=(0.05 0.1)
GAMMA_ARRAY=(0.0 0.5 1.5)
MAXDEPTH_ARRAY=(6 8 12)
LAMBDA_ARRAY=(1.0 2.0)
ALPHA_ARRAY=(0.0 0.05)

# ---------------------------------------

# ** icenet/setenv.sh uses these **
export HTC_PROCESS_ID=$1
export HTC_QUEUE_SIZE=$2
export HTC_CLUSTER_ID=$3

TUNESCRIPT=runme_zee_gridtune_P1.sh
source $ICEPATH/tests/zee/helper.sh
