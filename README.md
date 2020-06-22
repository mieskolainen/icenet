# icenet
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

work in progress!

## Dependencies
```
Python 3.7+ & see requirements.txt [created with pipreqs]
```

## Conda virtual environment setup
```
conda create -y --name icenet python==3.7.3
conda install -y --name icenet -c conda-forge --file requirements.txt
conda activate icenet
...[do your work]...
conda deactivate

conda info --envs
conda list --name icenet
```

## Folder structure

```
-checkpoint   Saved models
-configs      Input configuration
-docs         Documentation
-figs         Output figures
-icebrk       B/R(K) functions
-icefit       Fitting functions
-iceid        Electron ID functions
-icenet       Deep learning classes & functions
-iceplot      Plotting tools
-tests        Test functions
-dev          Development code
```

## Electron ID training / evaluation
```
python eid_train.py --config tune0 --datapath <some path> --datasets <0,1,2,...>
python eid_eval.py  --config tune0 --datapath <some path> --datasets <0,1,2,...>
```

## B/R(K) training / calculation / print / fit
```
python brk_train.py --config tune0 --datapath <some path> --datasets <0,1,2,...>
python brk_calc.py  --config tune0 --datapath <some path> --datasets <0,1,2,...> --tag <xyz>
python brk_print.py --config tune0 --datapath <some path> --datasets <0,1,2,...> --tag <xyz>
python brk_fit.py   --config tune0 --tag <xyz>
```

## Algorithms/packages included
```
1. Factorized (dim-by-dim) likelihoods & ratios using histograms [numpy]
2. Gradient boosted decision trees [xgboost]
3. Multinomial Logistic Regression [pytorch]
4. MAXOUT multilayer feedforward network [pytorch]
5. Deep Normalizing Flow (BNAF) based likelihoods & ratios [pytorch]
6. Graph Convolution Network (GCN) [pytorch]
7. Permutation Equivariant Networks [pytorch]
...
```


Mikael Mieskolainen, 2020\
m.mieskolainen@imperial.ac.uk
