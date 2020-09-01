# icenet
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

01/09/2020 -- work in progress!

## Dependencies
```
Python 3.8+ & see requirements.txt [created with pipreqs]
```

## Conda virtual environment setup
```
conda create -y --name icenet python==3.8.5
conda install -y --name icenet -c conda-forge --file requirements.txt

conda activate icenet
* xgboost, pytorch, torch-geometric ... setup now inside the environment *

...[do your work]...
conda deactivate

conda info --envs
conda list --name icenet
```

## xgboost setup
```
# Pick CPU or GPU version

conda install -c conda-forge py-xgboost
conda install -c nvidia -c rapidsai py-xgboost
```

## pytorch and torchvision setup
```
# Pick CPU or GPU version (check CUDA version with nvidia-smi)

conda install pytorch==1.5.1 torchvision==0.6.1 cpuonly -c pytorch
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.2 -c pytorch
```

## torch-geometric setup
```
# Pick CPU or GPU version

export CUDA=cpu
export CUDA=cu102 (or cu92, cu101)

pip install torch-scatter==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-sparse==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-cluster==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-spline-conv==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.5.0.html
pip install torch-geometric
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
-output       HDF5, pickle outputs
-dev          Development code
```

## Electron ID classifier
```
python eid_train.py --config tune0 --datapath <some path> --datasets <0,1,2,...>
python eid_eval.py  --config tune0 --datapath <some path> --datasets <0,1,2,...>
```

## B/R(K) analyzer
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
6. Permutation Equivariant Networks [pytorch]
7. CNN Networks [pytorch]
8. Graph Neural Nets [pytorch-geometric]
...
```


Mikael Mieskolainen, 2020\
m.mieskolainen@imperial.ac.uk
