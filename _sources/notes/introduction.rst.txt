Introduction
=======================

The library structure is as follows

.. contents::
    :local:


Basic design principles
---------------------------

Core deep learning and I/O functions and classes are designed to be problem generic.
That is, they can be used without any specific strict workflow and can handle near arbitrary
inputs as suitable.

Many high energy physics applications such as a signal from background discrimination problem
fit under certain "quasi-templated YAML-python-workflow" as manifested from the implemented applications.


YAML-configuration files
---------------------------

End-to-end deep learning applications are configured with YAML-files.
See source files for different applications under ``/configs``


Oracle Grid execution
-----------------------

DQCD analysis deployment example:

.. code-block:: none

	source tests/runme_dqcd_vector_init_yaml.sh
	python iceqsub/iceqsub.py --job dqcd_vector_data-D

After inspecting the launch command, launch by adding `--run`. Make sure you have
execute rights (chmod +x) for the steering script under `/tests`.


Folder structure
-----------------------

Folders starting with a name ``ice`` denote modules, typically
either different core modules such as ``icenet`` or ``icefit``
or physics applications such as ``icedqcd``, which contain their problem
specific I/O functions.

.. code-block:: none

	-analysis     Main steering macros and scripts
	-checkpoint   Trained and saved machine learning models
	-configs      YAML-input configuration
	-docs         Documentation
	-figs         Output figures
	-icebrk       B/R(K) analysis application
	-icedqcd      DQCD analysis application
	-icefit       Core fitting and statistics
	-icehgcal     HGCAL detector application
	-icehnl       HNL analysis application
	-iceid        Electron ID application
	-icenet       Core deep learning & I/O functions
	-iceplot      Core plotting tools
	-icetrg       HLT trigger application
	-tests        Test and bash-launch scripts
	-output       HDF5, pickle outputs
	-dev          Development code


ML-algorithms and models
-----------------------------

Various ML-models are implemented and supported. From a fixed dimensional input models
such as boosted decision trees (BDT) via XGBoost enhanced with a custom torch autograd computed loss function,
aka ``ICEBOOST``, to more complex "Geometric Deep Learning" with graph neural networks using torch-geometric
as a low-level backend.

The library is typically agnostic regarding the underlying models, i.e.
new torch models or loss functions can be easily added and other computational libraries such as JAX can be used.


See source files under ``/icenet/deep``

.. code-block:: none
	
	1.  Factorized (dim-by-dim) likelihoods & ratios using histograms [numpy]
	2.  ICEBOOST: Gradient boosted decision trees with custom autograd loss [xgboost+pytorch]
	3.  Multinomial Logistic Regression, Deep MLPs [pytorch]
	4.  MaxOUT multilayer feedforward network [pytorch]
	5.  Deep Normalizing Flow (BNAF) based likelihoods & ratios [pytorch]
	6.  Permutation Equivariant Networks (DeepSets) [pytorch]
	7.  CNN-Tensor networks [pytorch]
	8.  Graph Neural Nets (graph-, node-, edge-level inference) [pytorch-geometric]
	9.  Variational autoencoders [pytorch]
	10. Neural mutual information estimator (MINE) [pytorch]
	11. ...


Advanced ML-training technology
----------------------------------
See source files under ``/icenet/deep``

.. code-block:: none
	
	1. Model distillation
	2. Conditional (parametric) classifiers
	3. Deep domain adaptation (via gradient reversal)
	4. Automated hyperparameter tuning (via raytune)
	5. Algorithmically [de]correlated (regulated) BDTs and networks with MINE
	6. ...


Automated selectors and combinatorics for distributions
-------------------------------------------------------

The plotting machinery allows sophisticated filtering/cuts or "combinatorial" binning of various metrics, such as ROC-curves and other figures. See steering-file examples under ``/configs/*/plots.yml``

