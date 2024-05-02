# Electron ID steering code
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

import sys
sys.path.append(".")

# Configure plotting backend
import matplotlib
matplotlib.use('Agg')

from icenet.tools import process
from lptele import common

def main():
    args = process.generic_flow(rootname='lptele', func_loader=common.load_root_file, func_factor=common.splitfactor)

if __name__ == '__main__' :
    main()
