Introduction
=======================

The library structure is as follows:

.. contents::
    :local:

Folders
-----------------------

.. code-block:: none

	-analysis     Main steering macros and scripts
	-checkpoint   Saved models
	-configs      Input configuration
	-docs         Documentation
	-figs         Output figures
	-icebrk       B/R(K) analysis functions
	-icedqcd      DQCD analysis functions
	-icefit       Fitting and statistics functions
	-icehgcal     HGCAL functions
	-icehnl       HNL analysis functions
	-iceid        Electron ID functions
	-icenet       Core deep learning functions & I/O
	-iceplot      Plotting tools
	-icetrg       HLT trigger functions
	-tests        Test and steering scripts
	-output       HDF5, pickle outputs
	-dev          Development code


Algorithms and models
-----------------------------
.. code-block:: none

	1.  Factorized (dim-by-dim) likelihoods & ratios using histograms [numpy]
	2.  Gradient boosted decision trees with custom autograd loss [xgboost+pytorch]
	3.  Multinomial Logistic Regression, Deep MLPs [pytorch]
	4.  MAXOUT multilayer feedforward network [pytorch]
	5.  Deep Normalizing Flow (BNAF) based likelihoods & ratios [pytorch]
	6.  Permutation Equivariant Networks (DeepSets) [pytorch]
	7.  CNN Networks [pytorch]
	8.  Graph Neural Nets (graph-, node-, edge-level inference) [pytorch-geometric]
	9.  Variational autoencoders [pytorch]
	10. Neural mutual information estimator [pytorch]
	11. ...


Advanced training methodologies
----------------------------------
.. code-block:: none
	
	1. Model distillation
	2. Conditional (parametric) classifiers
	3. Deep domain adaptation (via gradient reversal)
	4. Automated hyperparameter tuning (via raytune)
	5. Algorithmically [de]correlated (regulated) networks
	6. ...

