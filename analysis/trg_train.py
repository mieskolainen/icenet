# High Level Trigger [TRAINING] steering code
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

# icenet system paths
import sys
sys.path.append(".")

# icenet
from icenet.tools import process
from icenet.tools import prints

# icetrg
from icetrg import common
from configs.trg.mvavars import *

# Main function
#
def main() :
    
    args, cli = process.read_config(config_path='./configs/trg')
    data      = process.read_data(args=args, func_loader=common.load_root_file, func_factor=common.splitfactor, 
        train_mode=True, imputation_vars=globals()[args['imputation_param']['var']])

    ### Print ranges
    prints.print_variables(X=data['trn']['data'].x, ids=data['trn']['data'].ids)

    process.make_plots(data=data['trn'], args=args)
    process.train_models(data_trn=data['trn'], data_val=data['val'], args=args)

    print(__name__ + ' [done]')

if __name__ == '__main__' :
   main()


"""
### Plot some kinematic variables
    targetdir = aux.makedir(f'./figs/{args["rootname"]}/{args["config"]}/reweight/1D_kinematic/')
    for k in ['x_hlt_pt', 'x_hlt_eta']:
        plots.plotvar(x = data.trn.x[:, data.ids.index(k)], y = data.trn.y, weights = trn_weights, var = k, nbins = 70,
            targetdir = targetdir, title = f"training re-weight reference class: {args['reweight_param']['reference_class']}")
"""
