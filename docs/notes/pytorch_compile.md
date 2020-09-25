## Step-by-Step: Compiling pytorch inside CONDA virtual environment on linux

### Create new virtual environment
```
conda create -y --name pytorch python==3.8.5
conda activate pytorch
```

### Install gcc/g++ 7 chain within conda
N.B. 8.2.0 does not work with CUDA11 drivers, 9.3.0 does not work with cudatoolkit 10.2

```
conda install -c conda-forge gcc_linux-64=7.5.0
conda install -c conda-forge gxx_linux-64=7.5.0
#conda install -c conda-forge gfortran_linux-64 # not needed here

conda install -c conda-forge cxx-compiler # This helper links 'g++' to 'x86_64-conda-linux-gnu-g++' etc.
```
For information, see: https://docs.conda.io/projects/conda-build/en/latest/resources/compiler-tools.html


### Install Pytorch basic dependencies (needed)
```
conda install numpy ninja pyyaml setuptools cmake cffi typing_extensions future six requests dataclasses
```

### Install NVIDIA MAGMA (LAPACK on GPU) (optional)
```
conda install -c pytorch magma-cuda102  # or [ magma-cuda101 | magma-cuda100 | magma-cuda92 ] depending on your cuda version
```

### Install NVIDIA cuDNN (needed)
```
#conda install -c nvidia cudatoolkit # this does not include the necessary headers (.h) and tool binaries, need to use other
conda install -c nvidia cudnn
```

### Install NVIDIA nccl (needed)
```
conda install -c nvidia nccl
```

### Install Intel MKL (needed)
```
conda install -c intel mkl mkl-include
```

### Get pytorch source with a specific tag
```
git clone --single-branch -b v1.6.0 --recursive https://github.com/pytorch/pytorch
cd pytorch
```

### Change CXX_ABI to C++11 version
Add the following line in the beginning of CMakeLists.txt 
```
set(GLIBCXX_USE_CXX11_ABI 1)
```
### Set C++11 flag for nvcc (NVIDIA compiler)
```
export CUDA_HOST_COMPILER=cc
#export TORCH_NVCC_FLAGS=""
```

### Setup CUDA toolkit and other paths

```
export CONDA_HOME=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export LD_LIBRARY_PATH=$CONDA_HOME/lib:$LD_LIBRARY_PATH
export PATH=$CONDA_HOME/bin:$PATH
export CMAKE_PREFIX_PATH=$CONDA_HOME # important!

export CC=x86_64-conda-linux-gnu-gcc # check that you have right one with 'x86_64-conda-linux-gnu-g++ --version'
export CXX=x86_64-conda-linux-gnu-g++

export PATH=$PATH:/usr/local/cuda-10.2/bin
export CUDA_HOME=/usr/local/cuda-10.2

export USE_CUDA=1
export USE_CUDNN=1
export MAX_JOBS=12
```

### Fix missing cublas_v2.h (hack)
```
ln -s /usr/include/cublas_v2.h $CONDA_HOME/include/cublas_v2.h
``` 

### Compile and install pytorch
```
python setup.py install
```

### After installation, check pytorch ABI (application binary interface) version
Go into another folder than ./pytorch (crucial) and run:
```
python -c "import torch; print(torch.compiled_with_cxx11_abi())"
```

## PROBLEMS:
If the compilation crashes with "internal compiler error", there is probably some problem with GCC libraries, e.g. with your libstdc++.so.6.
You may want to change your LD_LIBRARY_PATH (strip all unnecessary paths) and check which version of gcc is seen.
```
which gcc
```

If you get the error "Expected GLOO_USE_CUDA to be defined", you probably compiled first without CUDA. 
```
Clean up (delete) everything, and start from scratch. There are some broken files.
```

Fatal error: when writing output to /tmp/tmpxft_000061be_00000000-19_reduce_scatter.compute_60.cpp1.ii: No space left on device
compilation terminated.
```
Your /tmp path has not enough disk space (for you).
```

Internal compiler error: Killed (program cc1plus).
```
Try limiting the number of compilation jobs with 'export MAX_JOBS=4'.
```

