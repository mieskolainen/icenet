# Electron HLT trigger [EVALUATION] code
#
# m.mieskolainen@imperial.ac.uk, 2021

# icenet system paths
import _icepaths_

# icenet
from icenet.tools import process

#from configs.trg.mvavars import *
from icetrg import common


# Main function
#
def main() :

    ### Get input
    data, args, features = common.init()
    args['features'] = features

    ### Split and factor data
    data, data_kin = common.splitfactor(data=data, args=args)
    
    # Evaluate classifiers
    process.evaluate_models(outputname='trg', data=data, data_kin=data_kin, args=args)
    
    print(__name__ + ' [Done]')


if __name__ == '__main__' :

   main()

