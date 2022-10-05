# DQCD deployment steering code
# 
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

import os
import sys
sys.path.append(".")

# Configure plotting backend
import matplotlib
matplotlib.use('Agg')

from icenet.tools import process
from icedqcd import common
from icedqcd import deploy


def main():
    args, runmode = process.generic_flow(rootname='dqcd', func_loader=common.load_root_file, func_factor=common.splitfactor)

    deploy.process_data(args=args)

if __name__ == '__main__' :
    main()

