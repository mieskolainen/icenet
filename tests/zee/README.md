# Instructions for hyperparameter grid scan for `icezee` with Condor

m.mieskolainen@imperial.ac.uk, 2025

The overall goal is to train an *amortized conditional reweighter* AI-model using classifiers as density ratio estimators. For more information, see `configs/zee/NOTES.md`.

## First steps

```
cd tests/zee
chmod +x *.sh
```

Modify `gridtune_task_P{1,2,3}.sh` according to your environment and icenet installation folder. Execute grid launch commands in the folder `tests/zee`.

*NB. remember to repeat this, if you make e.g. several copies of the icenet installation under different paths for three different detector slices (EEm,EB,EEp).*

#### Which samples to tune against ?

For the Phase-1 and Phase-2 tuning jobs below, change `tst` field in the main steering `.yml` cards under `configs/zee` in order to use the validation sample dataframes:

```
genesis_runmode.mcfile['tst'] = ['MC_val_EEm*.parquet']
genesis_runmode.datafile['tst'] = ['MC_val_EEm*.parquet']
```

This way `--eval` stage metrics saved in tensorboard files are done against validation samples, and not the final evaluation sample.

## Method 1: DAGMan Condor Launch

Run these one-by-one on the main server node (no need for screen)

Repeat this for all `_EEm`, `_EEp`, and `_EB` by changing `CONFIG=...` line below.

### Phase 1:

First to clear cache, execute in the main folder
```
rm output/zee/* -f -r
```

Launch `S1-model` [kinematic reweighter] tuning jobs:
```
CONFIG="tune0_EEm"
maxevents=500000
TASK_SCRIPT="gridtune_task_P1.sh"
source submit_dagman.sh
```

Afterwords, find the best model using `icefit/iceboard.py` and `tensorboard`
applied to `/checkpoint/...`, which contains validation metrics (loss, AUC) as a function of epochs / boosts and cross-check visually reweighted kinematic distributions under `/figs/.../train` and `.../eval`.

Then update manually `models.yml` of the Stage-1 model accordingly, so the optimal parameter values will be used for the initialization of the next stage.

### Phase 2:

First to clear cache, execute in the main folder
```
rm output/zee/* -f -r
```

Launch `S2-model` [amortized conditional reweighter] tuning jobs:
```
CONFIG="tune0_EEm"
maxevents=500000
TASK_SCRIPT="gridtune_task_P2.sh"
source submit_dagman.sh
```

Then find the best model using `icefit/iceboard.py` and `tensorboard` applied to `/figs/...` and check visually full reweighting chain results under `/figs/.../eval`, then update manually `models.yml` of the Stage-2 model accordingly.

#### Which one is the best model ?

This multi-objective optimization depends on a combination of several factors:

- Full reweight chain goodness-of-fit -- chi2 of key observable histograms
- ESS (effective sample size) -- weight variance induced effective loss of events
- Preserving conditionality
- Differential behavior (visual)

The bias variance trade-off should be seen in 2D scatter plots in tensorboard. For example the pair

- `rESS/C0S12_C0S1/probe_eta` & `chi2/hybrid/C1_C0S12/mvaID`

should resolve a *Pareto frontier* with hyperparameter trials. The first metric here measures the reduction in effective sample size due to the S2-model (variance) [choose variables used in S1-stage] and the second one measures overall goodness-of-fit (primarily bias) against data [choose variables used in S2-stage].

The conditionality preservation can be monitored with

- `chi2/hybrid/C0_C0S2/{probe_eta, probe_pt, [other conditional]}`

This compares the original MC sample against the S2-model reweighted MC sample, without S1-model applied.

### Phase 3:

Run the final "mixed training" with two Stage-1 models incorporating pos+neg and pos-only input event MC weights and pos-only trained Stage-2, together with evaluation.

First to clear cache, execute in the main folder
```
rm output/zee/* -f -r
```

Then launch:
```
CONFIG="tune0_EEm"
maxevents=1000000000
TASK_SCRIPT="gridtune_task_P3.sh"
source submit_final.sh
```


## Method 2: Manual Condor Launch

As above, but use `submit_manual.sh` instead of `submit_dagman.sh` for Phase 1 and 2.

## Clean-up

After all jobs have executed, one can remove temporary files

```
rm tmp__*
```
