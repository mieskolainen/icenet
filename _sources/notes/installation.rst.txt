Installation
=======================

The framework installation goes as follows.

.. contents::
    :local:

Preliminaries: Conda installation
-------------------
.. code-block:: none

	wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
      
Then add rights (chmod +x) and execute with './filename.sh'

Fully automated setup
----------------------------------
.. code-block:: none

	conda create -y --name icenet python==3.8.5
	conda activate icenet
	
	# Pick CPU or GPU version (GPU version works for CPU too)	
	pip install -r requirements-cpu-linux.txt
	pip install -r requirements-gpu-linux.txt

	# Install this if and only if not already in your system
	conda install -c anaconda cudnn

Installing cudnn will install also cudatoolkit. Lacking cudnn may give an error such as: 'ImportError: libcudnn.so.7: cannot open shared object file'

Note: If you experience ´No space left on device´ problem with pip, set the temporary path
with: export TMPDIR="somepath"

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
	
	conda install pytorch==1.6.0 torchvision==0.6.1 cpuonly -c pytorch
	conda install pytorch==1.6.0 torchvision==0.6.1 -c pytorch


Pytorch-geometric setup
--------------------------

.. code-block:: none
	
	# Pick CPU or GPU version below
	
	export CUDA=cpu
	export CUDA=cu102 # (or cu92, cu101)

	pip install torch-scatter==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.6.0.html
	pip install torch-sparse==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.6.0.html
	pip install torch-cluster==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.6.0.html
	pip install torch-spline-conv==latest+$CUDA -f https://pytorch-geometric.com/whl/torch-1.6.0.html
	pip install torch-geometric


