# DQCD steering code
# 
# m.mieskolainen@imperial.ac.uk, 2024

import sys
sys.path.append(".")

# Configure plotting backend
import matplotlib
matplotlib.use('Agg')

from icenet.tools import process
from icedqcd import common, optimize

def main():
    args, runmode = process.generic_flow(rootname='dqcd', func_loader=common.load_root_file, func_factor=common.splitfactor)

    if runmode == 'optimize':
        optimize.optimize_selection(args=args)


if __name__ == '__main__' :
    main()
