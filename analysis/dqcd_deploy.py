# DQCD deployment steering code
# 
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

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

    path      = '/home/user/travis-stash/input/icedqcd/bparkProductionV2/'
    process   = 'HiddenValley_vector_m_10_ctau_10_xiO_1_xiL_1_privateMC_11X_NANOAODSIM_v2_generationForBParking'
    
    filename  = process + '/output_100.root'
    
    deploy.process_data(args=args, path=path, filename=filename)


if __name__ == '__main__' :
    main()
