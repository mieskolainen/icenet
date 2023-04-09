# Set library paths
# (use after activating the conda environment)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo $LD_LIBRARY_PATH

echo "Before:"
echo " ulimit -s: $(ulimit -s) (stack)" 
echo " ulimit -u: $(ulimit -u) (threads)" 
echo " ulimit -v: $(ulimit -v) (virtual memory)" 

# Set system memory limits
ulimit -s unlimited  # stack
ulimit -u 131072     # num of threads (for Sun Grid Engine use)
#ulimit -v unlimited # virtual memory

echo "After:"
echo " ulimit -s: $(ulimit -s)" 
echo " ulimit -u: $(ulimit -u)" 
echo " ulimit -v: $(ulimit -v)" 
