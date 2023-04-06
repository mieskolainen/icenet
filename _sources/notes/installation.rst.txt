Installation
=======================

The framework installation goes as follows.

.. contents::
    :local:


Preliminaries: Conda installation
----------------------------------
.. code-block:: none

	wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh 

Then execute the installer with ``bash filename.sh``.


Pre-installed CUDA paths (optional, deprecated)
------------------------------------
.. code-block:: none

	source /vols/software/cuda/setup.sh 11.2.0

This can be used with IC machines in principle, however, is not needed with Conda driven setup.


Automated setup
----------------------------------

Remark: To avoid ``No space left on device`` problem with conda or pip, set the temporary path first, e.g.

.. code-block:: none
	
	mkdir $HOME/tmp
	export TMPDIR=$HOME/tmp

Execute

.. code-block:: none

	git clone git@github.com:mieskolainen/icenet.git && cd icenet
	
	# Create environment
	conda create -y --name icenet python==3.10.10
	conda activate icenet
	
	# Install cudatoolkit and cudnn (make sure no other installations overlap)
	conda install -c nvidia cudatoolkit==11.7.0
	conda install -c conda-forge cudnn==8.4.1.50 cudatoolkit-dev==11.7.0
	
	# Install dependencies with pip
	pip install -r requirements.txt
	
	(OR pip install -r requirements-cpu-only.txt e.g. for Github Actions)


Initialize the environment
----------------------------------

Always start with

.. code-block:: none

	conda activate icenet
	source setenv.sh


Possible problems
----------------------------------

Note: If you experience ``OSError: libcusparse.so.11`` (or similar) with torch-geometric, set the system path

.. code-block:: none

	export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"


Note: If you experience ``Could not load dynamic library libcusolver.so.10`` with tensorflow, make a symbolic link

.. code-block:: none

	ln -s $CONDA_PREFIX/lib/libcusolver.so.11 $CONDA_PREFIX/lib/libcusolver.so.10

Note: ``$CONDA_PREFIX`` will be found after ``conda activate icenet``

Note: If you experience "Requirement already satisfied" infinite loop with pip, try
removing e.g. ``tensorflow`` from requirements.txt, and install it separately with

.. code-block:: none
	
	pip install tensorflow


Then if something else fails, google with the error message.


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

.. code-block:: none

	qsub -q gpu.q@lxcgpu* <other commands>


Conda virtual environment commands
-----------------------------------
.. code-block:: none

	conda create -y --name icenet python==3.X.Y
	conda activate icenet

	...[install dependencies with pip, do your work]...
	
	conda deactivate

	conda info --envs
	conda list --name icenet
	
	# Remove environment completely
	conda env remove --name icenet

C-library versions
-----------------------------------

.. code-block:: none

	ldd --version
