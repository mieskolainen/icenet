Installation
=======================

The framework installation goes as follows.

.. contents::
    :local:

Preliminaries: Conda installation
----------------------------------
.. code-block:: none

	wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh

Then add rights (chmod +x) and execute with './filename.sh'

Preliminaries: CUDA paths
---------------------------
.. code-block:: none

	source /vols/software/cuda/setup.sh 11.2.0

This is for IC machines, mutatis mutandis for other environments.
(or 10.2.2, see ls /vols/software/cuda)

Fully automated setup
----------------------------------
.. code-block:: none

	git clone https://github.com/mieskolainen/icenet && cd icenet
		
	conda create -y --name icenet python==3.8.5
	conda activate icenet
	
	# Pick GPU or CPU version (GPU version works for CPU too)	
	pip install -r requirements-gpu-linux.txt
	pip install -r requirements-cpu-linux.txt

Note: Lacking cudnn (or cudatoolkit) may give an error such as: 'ImportError: libcudnn.so.7'.
This can be installed inside the conda environment with

.. code-block:: none

	conda install -c anaconda cudnn

Note: If you experience ´No space left on device´ problem with pip, set the temporary path

.. code-block:: none
	
	mkdir $HOME/tmp
	export TMPDIR=$HOME/tmp

Note: If you experience ´Requirement already satisfied´ infinite loop with pip, try
removing e.g. ´tensorflow´ from requirements.txt, and install it separately with

.. code-block:: none
	
	pip install tensorflow


Then if something else fails, follow the instructions step-by-step below.


Conda virtual environment setup
--------------------------------
.. code-block:: none

	conda create -y --name icenet python==3.8.5
	conda activate icenet
	conda install -c conda-forge --file requirements.txt
	
	* xgboost, pytorch, torch-geometric ... setup now inside the environment *

	...[do your work]...
	
	conda deactivate

	conda info --envs
	conda list --name icenet


XGBoost setup
--------------
.. code-block:: none

	# Pick CPU or GPU version

	conda install -c conda-forge py-xgboost
	conda install -c nvidia -c rapidsai py-xgboost


Pytorch and torchvision setup
------------------------------

.. code-block:: none

	# Pick CPU or GPU version below
	# Check maximum CUDA version supported by your drivers with nvidia-smi
	
	conda install pytorch==1.7.1 torchvision==0.8.2 cpuonly -c pytorch
	conda install pytorch==1.7.1 torchvision==0.8.2 -c pytorch

Pytorch-geometric setup
--------------------------

.. code-block:: none
	
	# Pick CPU or GPU version below
	
	export CUDA=cpu
	export CUDA=cu102 # (or cu92, cu101, cu110)
	
	pip install torch-scatter==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.7.0.html
	pip install torch-sparse==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.7.0.html
	pip install torch-cluster==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.7.0.html
	pip install torch-spline-conv==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.7.0.html
	pip install torch-geometric


GPU-support commands
---------------------

Show the graphics card status

.. code-block:: none
	
	nvidia-smi	

Show CUDA-compiler tools status

.. code-block:: none
	
	nvcc --version	

Show Tensorflow and Pytorch GPU support in Python

.. code-block:: none
	
	import tensorflow
	from tensorflow.python.client import device_lib
	print(device_lib.list_local_devices())
	
	import torch
	torch.cuda.is_available()
	print(torch.cuda.get_device_name(0))

