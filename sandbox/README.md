# Sandbox

Simple standalone scripts for diagnostic checks, etc.

## eid

```
cd eid
. run.sh # runs train.py
```

A (now obsolete) script that trains a simple BDT for electron ID and
produces a figure comparing various ROCs (based on existing and new
BDTs).

```
cd eid
python reweight.py
```

A reweighting script that uses synthetic data. 

## brem

```
cd brem
. run.sh # runs reweight.py
```

A script that reweights samples by pT, eta, rho (a proxy for pileup),
etc (using a KNN to cluster) and then trains a BDT with the
un/weighted samples to demonstrate the reweighting method is able to
remove the discrimatory power of the aforementioned variables.
