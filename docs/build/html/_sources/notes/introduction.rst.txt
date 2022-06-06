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
	-icefit       Fitting and statistics functions
	-iceid        Electron ID functions
	-icenet       Deep learning classes & functions
	-icetrg       HLT trigger classifiers
	-iceplot      Plotting tools
	-tests        Test and steering scripts
	-output       HDF5, pickle outputs
	-dev          Development code


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
	9. ...


