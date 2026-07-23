# Documentation for the reweighting algorithm presented in the CMS-MLG-25-002 paper
 
Each subdetector requires its own dedicated model training, HPC submission and finetuning.

The variables included in Stage 1 and Stage 2 are defined in  
https://github.com/mieskolainen/icenet/blob/master/configs/zee/mvavars_EB.py 
for detector barrel and 
https://github.com/mieskolainen/icenet/blob/master/configs/zee/mvavars.py
for detector endcaps.

The training and evaluation parameters are defined in 
https://github.com/mieskolainen/icenet/blob/master/configs/zee/models.yml
for all available models in icenet, irrespective of whether they were trained or not, whilst
https://github.com/mieskolainen/icenet/blob/master/configs/zee/active_models.yml
lists the models that were trained.

https://github.com/mieskolainen/icenet/blob/master/configs/zee/plots.yml
lists the plotting parameters, including the values chosen for evaluating with a specific temperature, or temperature ranges.

https://github.com/mieskolainen/icenet/blob/master/configs/zee/stage1rw.yml
defines the Stage 1 reweighting, which was the `iceboost4D` architecture for the 4 Stage 1 variables.

https://github.com/mieskolainen/icenet/blob/master/configs/zee/tune0_EB.yml
https://github.com/mieskolainen/icenet/blob/master/configs/zee/tune0_EEm.yml
https://github.com/mieskolainen/icenet/blob/master/configs/zee/tune0_EEp.yml
list subdetector-specific run parameters such as rng seeds, whether negative weights were dropped and numerical precision bounds.

To submit new training jobs, instructions for job submission (both for manual submission and condor dagman submission) are in https://github.com/mieskolainen/icenet/blob/master/tests/zee/README.md
Each training job requires running 4 commands of the following form
```
CONFIG="tune0_EEm"
maxevents=500000
TASK_SCRIPT="gridtune_task_P1.sh"
source submit_dagman.sh
```
to specify the detector region, maximum training event number and the gridtune script depending on job stage, all as bash arguments. The job is then started by sourcing the dagman or manual bash script mentioned in the instructions linked above.

In practice for the paper presented, models were trained in 3 phases. As discussed in the paper, Stage 1 models were trained in a positive-weights-only mode and separately in a full-events mode. This was phase 1. Phase 2 trained Stage 2 model on positively weighted events only, pre-reweighted with the S1 model on only positively events. Phase 3 is a final evaluation mode, where all events including negative events were piped through  the S1 trained on all events, and the S2 that was trained on positive-only. This check thus took the full events into account. 
Note that in Phase 3 `maxevents=1000000000` is an arbitrarily large number > the event count in the input sample, which guarantees that all events were evaluated and included in the final metrics, and also separates the Phase 3 evaluation step from the prior training phases that used a smaller event count during training.

https://github.com/mieskolainen/icenet/blob/master/tests/zee/gridtune_init.job
defines the condor job run requirements such as GPU type, run time and memory requirements for the Stage 1 reweighter, whilst 
https://github.com/mieskolainen/icenet/blob/master/tests/zee/gridtune_array.job 
defines these for the Stage 2 reweighters.

https://github.com/mieskolainen/icenet/blob/master/tests/zee/gridtune_task_P1.sh
https://github.com/mieskolainen/icenet/blob/master/tests/zee/gridtune_task_P2.sh
https://github.com/mieskolainen/icenet/blob/master/tests/zee/gridtune_task_P3.sh
define the hyperparameters of the finetuning jobs. For this paper different values for only the following variables  
* learning rate
* noise regularisation
* maximum tree depth and
* SWD strength
  
were iterated over for S2, while for S1 only

* learning rate
* L1 regularisation
* L2 regularisation
* maximum tree depth and
* Gamma

were iterated over.

These `gridtune_task.sh` files set the hyperparameter lists and job IDs, then call https://github.com/mieskolainen/icenet/blob/master/tests/zee/helper.sh
which activates the icenet environment and then runs 
https://github.com/mieskolainen/icenet/blob/master/tests/runme_zee_gridtune_P1.sh
https://github.com/mieskolainen/icenet/blob/master/tests/runme_zee_gridtune_P2.sh
https://github.com/mieskolainen/icenet/blob/master/tests/runme_zee_gridtune_P3.sh
that calculate the specific hyperparameter combinations for each finetuning job. 

Each job run by these `gridtune.sh` bash files is of the form
```
python analysis/zee.py --runmode train $MAX   --modeltag ${MODELTAG} --config ${CONFIG}.yml --datapath $DATAPATH \
  --run_id $RUN_ID --supertune "${SUPERTUNE}" --compute 0

python analysis/zee.py --runmode eval  $MAX   --modeltag ${MODELTAG} --config ${CONFIG}.yml --datapath $DATAPATH \
  --run_id $RUN_ID --compute 0
```
where `SUPERTUNE` sets the chosen hyperparameter values for a given finetuning job, `--compute 0` skips some metrics calculation and certain file saving not needed during training, and `--modeltag` and `--run_id` define run tags for clear output targets when saving results.


If finetuning is not needed, and only jobs with pre-defined hyperparameters is to be run, running the following file is all that is needed:
```

maxevents=500000
source runme_zee.sh
```
where `runme_zee.sh` is in  https://github.com/mieskolainen/icenet/blob/master/tests/runme_zee.sh


