# Electron ID steering code
#
# m.mieskolainen@imperial.ac.uk, 2024

import sys
sys.path.append(".")

# Configure plotting backend
import matplotlib
matplotlib.use('Agg')

from icenet.tools import process
from iceid import common

def main():
    args = process.generic_flow(rootname='eid', func_loader=common.load_root_file, func_factor=common.splitfactor)

if __name__ == '__main__' :
    main()
