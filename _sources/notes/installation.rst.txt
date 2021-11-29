Installation
=======================

The framework installation goes as follows.

.. contents::
    :local:


Preliminaries: Conda installation
----------------------------------
.. code-block:: none

	wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh

Then execute with 'bash filename.sh'


Pre-installed CUDA paths (optional)
------------------------------------
.. code-block:: none

	source /vols/software/cuda/setup.sh 11.2.0

This can be used with IC machines, however, is not needed with conda driven setup.


Automated setup
----------------------------------
.. code-block:: none

	git clone https://github.com/mieskolainen/icenet && cd icenet
	
	# Create environment
	conda create -y --name icenet python==3.9.6
	conda activate icenet
	
	# Install cudatoolkit and cudnn (make sure no other installations overlap)
	conda install -c nvidia cudatoolkit==11.1.74 cudnn==8.0.4
	conda install -c conda-forge cudatoolkit-dev
	
	# Pick GPU or CPU version (GPU version works for CPU too)	
	pip install -r requirements-gpu-linux.txt
	pip install -r requirements-cpu-linux.txt


Note: If you experience ´No space left on device´ problem with pip, set the temporary path

.. code-block:: none
	
	mkdir $HOME/tmp
	export TMPDIR=$HOME/tmp


Note: If you experience ´OSError: libcusparse.so.11´ (or similar) with torch-geometric, set the system path

.. code-block:: none

	export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"


Note: If you experience ´Could not load dynamic library libcusolver.so.10´ with tensorflow, make a symbolic link

.. code-block:: none

	ln -s $CONDA_PREFIX/lib/libcusolver.so.11 $CONDA_PREFIX/lib/libcusolver.so.10

Note: $CONDA_PREFIX will be found after 'conda activate icenet'

Note: If you experience ´Requirement already satisfied´ infinite loop with pip, try
removing e.g. ´tensorflow´ from requirements.txt, and install it separately with

.. code-block:: none
	
	pip install tensorflow


Then if something else fails, study the instructions step-by-step below.


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


Grid engine GPU job submission
-------------------------------

Use the following command with IC machines

-- code-block:: none
    qsub -q gpu.q@lxcgpu* <other commands>


Conda virtual environment commands
-----------------------------------
.. code-block:: none

	conda create -y --name icenet python==3.9.6
	conda activate icenet

	...[install dependencies with pip, do your work]...
	
	conda deactivate

	conda info --envs
	conda list --name icenet
	
	# Remove environment completely
	conda env remove --name icenet

