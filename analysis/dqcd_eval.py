# DQCD [EVALUATION] code
#
# m.mieskolainen@imperial.ac.uk, 2022

# icenet system paths
import sys
sys.path.append(".")

# icenet
from icenet.tools import process
from icenet.tools import reweight

#from configs.dqcd.mvavars import *
from icedqcd import common


# Main function
#
def main() :

    ### Get input
    data, args = common.init()
    args['features'] = data.ids
    
    ### Compute reweighting weights for the evaluation (before split&factor !)
    if args['eval_reweight']:
        tst_weights = reweight.compute_ND_reweights(x=data.tst.x, y=data.tst.y, ids=data.ids, args=args['reweight_param'])
    else:
        tst_weights = None
    
    ### Split and factor data
    data, data_kin = common.splitfactor(data=data, args=args)
    
    # Evaluate classifiers
    process.evaluate_models(data=data, data_kin=data_kin, weights=tst_weights, args=args)
    
    print(__name__ + ' [Done]')


if __name__ == '__main__' :

   main()

