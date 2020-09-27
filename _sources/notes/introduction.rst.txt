Introduction
=======================

The library structure is the following.

.. contents::
    :local:

Folder structure
-----------------------

.. code-block:: none

	-analysis     Main steering macros and scripts
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


Electron ID classifier
-----------------------
.. code-block:: none

	python ./analysis/eid_train.py --config tune0 --datapath <some path> --datasets <0,1,2,...>
	python ./analysis/eid_eval.py  --config tune0 --datapath <some path> --datasets <0,1,2,...>


B/R(K) analyzer
-----------------------
.. code-block:: none

	python ./analysis/brk_train.py --config tune0 --datapath <some path> --datasets <0,1,2,...>
	python ./analysis/brk_calc.py  --config tune0 --datapath <some path> --datasets <0,1,2,...> --tag <xyz>
	python ./analysis/brk_print.py --config tune0 --datapath <some path> --datasets <0,1,2,...> --tag <xyz>
	python ./analysis/brk_fit.py   --config tune0 --tag <xyz>


Algorithms/packages included
-----------------------------
.. code-block:: none

	1. Factorized (dim-by-dim) likelihoods & ratios using histograms [numpy]
	2. Gradient boosted decision trees [xgboost]
	3. Multinomial Logistic Regression [pytorch]
	4. MAXOUT multilayer feedforward network [pytorch]
	5. Deep Normalizing Flow (BNAF) based likelihoods & ratios [pytorch]
	6. Permutation Equivariant Networks [pytorch]
	7. CNN Networks [pytorch]
	8. Graph Neural Nets [pytorch-geometric]
	...

