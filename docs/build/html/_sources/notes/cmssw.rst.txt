CMSSW setup
=======================

The framework installation goes as follows.

.. contents::
    :local:


Preliminaries: SSH public key to github
----------------------------------------
.. code-block:: none

	cat ~/.ssh/id_rsa.pub

Copy the public key to [github.com / SETTINGS / ssh keys]


Preliminaries: CMSSW release setup
-----------------------------------
.. code-block:: none

	cd ~
    source /cvmfs/cms.cern.ch/cmsset_default.sh
    export SCRAM_ARCH=slc7_amd64_gcc700

    scramv1 project CMSSW CMSSW_10_2_22 [or `cmsrel CMSSW_10_2_22`]
    cd CMSSW_10_2_22/src
    eval `scram runtime -sh` [OR `cmsenv`]

    git clone https://github.com/mieskolainen/nanotron nanotron
    scram b
	cmsRun nanotron/producer/test/produceNANO.py

