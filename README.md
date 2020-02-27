# icenet
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

work in progress

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


## Training
```
python train.py --config tune0 --datapath <some path> --datasets <0,1,2,...>
```

## Evaluation
```
python eval.py --config tune0 --datapath <some path> --datasets <0,1,2,...>
```

## Algorithms/packages included
```
1. Factorized (dim-by-dim) likelihood ratio using histograms [numpy]
2. Gradient boosted decision trees [xgboost]
3. Multinomial logistic regression [pytorch]
4. Maxout multilayer feedforward network (MAXOUT-MLP) [pytorch]
5. Deep Normalizing Flow (BNAF) based likelihood ratio [pytorch]
```


Mikael Mieskolainen, 2020\
m.mieskolainen@imperial.ac.uk
