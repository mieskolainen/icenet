# Instructions for the hyperparameter grid scan using Condor for `icezee`

m.mieskolainen@imperial.ac.uk, 2025

Launch by executing the commands in the folder `tests/zee`:

```
cd tests/zee
chmod +x *.sh
```

## Method 1: DAGMan Condor Launch

Run these one-by-one on the main server node (no need for screen)

```
TASK_SCRIPT="gridtune_task_EEm.sh"; source submit_dagman.sh
TASK_SCRIPT="gridtune_task_EEp.sh"; source submit_dagman.sh
TASK_SCRIPT="gridtune_task_EB.sh";  source submit_dagman.sh
```

## Method 2: Manual Condor Launch

Run these one-by-one using `screen` sessions on the main server node

```
TASK_SCRIPT="gridtune_task_EEm.sh"; source submit_manual.sh
TASK_SCRIPT="gridtune_task_EEp.sh"; source submit_manual.sh
TASK_SCRIPT="gridtune_task_EB.sh";  source submit_manual.sh
```

## Clean-up

After all jobs have executed, one can remove temporary files

```
rm tmp__*
```
