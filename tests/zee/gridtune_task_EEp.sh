#!/bin/sh
#
# (Modify this file!)
#
# m.mieskolainen@imperial.ac.uk, 2024

SETCONDA="/home/hep/mmieskol/setconda.sh"
ICEPATH="/vols/cms/mmieskol/icenet"

DATAPATH="/vols/cms/pfk18/icenet_files/processed_29_nov_24"
CONFIG="tune0_EEp"
MODELTAG="GRIDTUNE"
maxevents=500000

BETA_ARRAY=(0.0 0.0025 0.005 0.01 0.02 0.04)
SIGMA_ARRAY=(0.0 0.025 0.05 0.1 0.2)
LR_ARRAY=(0.1)
GAMMA_ARRAY=(1.5)
MAXDEPTH_ARRAY=(13)
LAMBDA_ARRAY=(2.0)
ALPHA_ARRAY=(0.05)

SWD_VAR="['.*']" # all
#SWD_VAR="['fixedGridRhoAll', 'probe_eta', 'probe_pt']"

# ---------------------------------------

# ** icenet/setenv.sh uses these **
export HTC_PROCESS_ID=$1
export HTC_QUEUE_SIZE=$2
export HTC_CLUSTER_ID=$3

source $ICEPATH/tests/zee/helper.sh
