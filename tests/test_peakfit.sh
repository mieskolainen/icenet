#!/bin/sh
#
# Test peakfit

git clone https://github.com/mieskolainen/actions-stash.git

for L in chi2 huber nll; do
    for T in single dual ; do
        python icefit/peakfit.py --num_cpus 1 --test_mode --analyze --group --loss_type "$L" --fit_type "$T" --output_name "test_${L}_${T}" > "peakfit_${L}_${T}.log"
    done
done
