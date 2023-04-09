# Set library paths
# (use after activating the conda environment)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Set system memory limits
ulimit -s unlimited  # stack
#ulimit -u 131072     # num of threads
#ulimit -v unlimited # virtual memory
