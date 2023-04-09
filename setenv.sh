# Set library paths
# (use after activating the conda environment)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo $LD_LIBRARY_PATH

echo "Before:"
ulimit -a

# Set system memory limits
ulimit -s unlimited  # stack
ulimit -u 1048576    # num of processes (for Sun Grid Engine use)
#ulimit -v unlimited # virtual memory

echo ""
echo "After:"
ulimit -a
