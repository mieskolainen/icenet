# DQCD steering code
# 
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

import sys
sys.path.append(".")

# Configure plotting backend
import matplotlib
matplotlib.use('Agg')

from icenet.tools import process
from icedqcd import common, significance

def main():
    args, runmode = process.generic_flow(rootname='dqcd', func_loader=common.load_root_file, func_factor=common.splitfactor)

    if runmode == 'optimize':
        significance.optimize_selection(args=args)

if __name__ == '__main__' :
    main()
