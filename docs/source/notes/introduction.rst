Introduction
=======================

The library structure is as follows

.. contents::
    :local:


Basic design principles
---------------------------

Core deep learning and I/O functions and classes are designed to be problem generic.
That is, they can be used without any specific strict workflow and can handle near arbitrary
inputs as suitable (parquet files, ROOT files ...).

Many high energy physics applications such as the signal-from-background discrimination problem
fit under certain "quasi-templated YAML-python-workflow" as manifested from the implemented applications.


YAML-configuration files
---------------------------

End-to-end deep learning applications are configured with YAML-files.
See source files for different applications under ``/configs``


Folder structure
-----------------------

Folders starting with a name ``ice`` denote modules, typically
either different core modules such as ``icenet`` or ``icefit``
or physics applications such as ``icedqcd``, which contain their problem
specific I/O functions.

.. code-block:: none

	-analysis     Main steering macros and scripts
	-checkpoint   Trained and saved AI-models
	-configs      YAML-input configuration
	-docs         Documentation
	-figs         Output figures
	-icebrem      Electron ID application
	-icebrk       B/R(K) analysis (proto) application [combinatorial classification]
	-icedqcd      DQCD analysis application [large scale new physics analysis, domain adaptation]
	-icefit       Core fitting and statistics [tag & probe ++]
	-icehgcal     HGCAL detector application [graph neural networks]
	-icehnl       HNL analysis application [neural mutual information with BDT and MLP]
	-iceid        Electron ID application
	-icenet       Core deep learning & I/O functions
	-iceplot      Core plotting tools
	-iceqsub      SGE submission steering functions
	-icetrg       HLT trigger application
	-icezee       High-dimensional reweighting application [advanced MLP models and regularization]
	-tests        Tests, continuous integration (CI) and bash-launch scripts
	-output       HDF5, pickle outputs
	-dev          Development code

The quasi-templated workflow mechanics is implemented in ``icenet/tools/process.py``.


AI-algorithms and models
-----------------------------

Various ML and AI-models are implemented and supported. From a fixed dimensional input models
such as boosted decision trees (BDT) via XGBoost enhanced with a custom torch autograd driven loss function,
aka ``ICEBOOST``, to more complex "Geometric Deep Learning" with graph neural networks using torch-geometric
as a low-level backend.

The library is ultimately agnostic regarding the underlying models, i.e.
new torch models or loss functions can be easily added and other computational libraries such as JAX can be used.

For adding new torch models, see source files under ``/icenet/deep``, especially ``train.py``, ``iceboost.py``, ``optimize.py`` and ``predict.py``.

Reasily available models such as

.. code-block:: none
	
	1.  ICEBOOST: Gradient boosted decision trees with a custom autograd loss [xgboost+pytorch]
	2.  Kolmogorov-Arnold representation theorem networks [pytorch]
	3.  Lipschitz continuous MLPs [pytorch]
	4.  Graph Neural Nets (graph-, node-, edge-level inference) [pytorch-geometric]
	5.  Deep Normalizing Flow (BNAF) based pdfs & likelihood ratios [pytorch]
	6.  Neural mutual information estimator (MINE) and non-linear distance correlation (DCORR) [pytorch]
	7.  MaxOUT multilayer feedforward network [pytorch]
	8.  Permutation Equivariant Networks (DeepSets) [pytorch]
	9.  CNN-Tensor networks [pytorch]
	10. Variational autoencoders [pytorch]
	11. Deep MLPs, logistic regression [pytorch]
	12. Simple estimators such as factorized (dim-by-dim) pdfs & likelihood ratios using histograms [numpy]
	13. ...


Advanced ML-training technology
----------------------------------

See source files under ``/icenet/deep``

.. code-block:: none
	
	1. Model distillation
	2. Conditional (theory) parametric classifiers
	3. Inverse CDF based dequantization of a lattice sampled conditional variables 
	4. Simple and deep domain adaptation (via gradient reversal)
	5. Automated hyperparameter tuning (via raytune)
	6. Algorithmically [de]correlated (regulated) BDTs and networks with MINE
	7. Logit temperature scaling diagnostics and optimization (model output calibration)
	8. ...


Automated selectors and combinatorics for distributions
-------------------------------------------------------------

The plotting machinery allows sophisticated filtering/cuts or "combinatorial" binning of various metrics, such as ROC-curves and other figures.
See steering-file examples under ``/configs/*/plots.yml``


Sun Grid Engine (SGE) / HTCondor execution
------------------------------------------------------

DQCD analysis deployment example:

.. code-block:: none

	source tests/runme_dqcd_vector_init_yaml.sh
	python iceqsub/iceqsub.py --job dqcd_vector_data-D

After inspecting the launch command, launch by adding `--run`. Make sure you have
execute rights (chmod +x) for the steering script under `/tests`.
