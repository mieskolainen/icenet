Installation
=======================

The framework installation goes as follows.

.. contents::
    :local:


Preliminaries: Conda installation
----------------------------------
.. code-block::

	wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh

Then execute the installer with ``bash filename.sh`` and finally set ``.condarc`` as follows ``nano .condarc`` (home directory)

.. code-block::

    channel_priority: strict
    channels:
      - conda-forge
      - defaults


Pre-installed CUDA paths (OBSOLETE)
------------------------------------
.. code-block::

	source /vols/software/cuda/setup.sh 11.8.0

This can be used with IC machines in principle, however, is not tested.


Automated setup
----------------------------------

Remark: To avoid ``No space left on device`` problem with conda or pip, set the temporary path first

.. code-block::
	
	mkdir <PATH_WITH_SPACE>/tmp
	export TMPDIR=<PATH_WITH_SPACE>/tmp

Also after activating the icenet conda environment.

Execute

.. code-block::
	
	git clone git@github.com:mieskolainen/icenet.git && cd icenet
	
	# Create the environment
	conda env create -f environment.yml
	conda activate icenet
	
	# Install dependencies with pip
	python -m pip install -r requirements.txt
	
	(OR -r requirements-cpu-only.txt e.g. for Github Actions)

Note: The command ``python -m pip`` should use the pip installed under the conda environment.


Initialize the environment
----------------------------------

Always start with

.. code-block::

	conda activate icenet
	source setenv.sh


Possible problems
----------------------------------

Note: One may need to steer where ``pip`` installs the packages, for example

.. code-block::

	python -m pip install --target $CONDA_PREFIX <package>

Note: If you experience ``OSError: libcusparse.so.11`` (or similar) with torch-geometric, set the system path

.. code-block::

	export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

Note: If you experience ``Could not load dynamic library libcusolver.so.10`` with tensorflow, make a symbolic link

.. code-block::

	ln -s $CONDA_PREFIX/lib/libcusolver.so.11 $CONDA_PREFIX/lib/libcusolver.so.10

Note: ``$CONDA_PREFIX`` will be found after ``conda activate icenet``

Note: If you experience "Requirement already satisfied" infinite loop with pip, try
removing e.g. ``tensorflow`` from requirements.txt, and install it separately with

.. code-block::
	
	pip install tensorflow

Then if something else fails, google with the error message.


Show installation paths of binaries
--------------------------------------

.. code-block::
	
	which -a pip
	which -a python


GPU-support commands
---------------------

Show the graphics card status

.. code-block::
	
	nvidia-smi

Show CUDA-compiler tools status

.. code-block::
	
	nvcc --version

Show Tensorflow and Pytorch GPU support in Python

.. code-block::
	
	import tensorflow
	from tensorflow.python.client import device_lib
	print(device_lib.list_local_devices())
	
	import torch
	torch.cuda.is_available()
	print(torch.cuda.get_device_name(0))


HTCondor GPU job submission
-------------------------------

Use the following command with IC machines

.. code-block::

	condor_submit <job_description_file>
	condor_rm <job_id>
	condor_ssh_to_job <job_id> (DEBUG)
	condor_q
	condor_status --gpus

With an example job description file as

.. code-block::

	executable     = gpu_task.sh
	error          = gpu.$(CLUSTER).error
	output         = gpu.$(CLUSTER).output
	log            = gpu.$(CLUSTER).log
	request_gpus   = 1
	request_memory = 10G
	+MaxRuntime    = 3600
	queue

Where ``gpu_task.sh`` is the actual steering shell script to be run

.. code-block::

	source setconda.sh
	conda activate icenet
	python icecool.py

where ``source setconda.sh`` has the Conda (system specific) init commands.


Conda virtual environment commands
-----------------------------------

.. code-block::

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

.. code-block::

	ldd --version
