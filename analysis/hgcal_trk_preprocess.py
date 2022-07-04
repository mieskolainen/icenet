# HGCAL [TRAINING] steering code
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk

# icenet system paths
import sys
sys.path.append(".")
import pickle

# icenet
from icenet.tools import process
from icenet.tools import aux
from icenet.tools import plots

# icehgcal
from icehgcal import common
from icehgcal import preprocess
from icehgcal import graphio


from configs.hgcal.mvavars import *

# Main function
#
def main() :
    
    cli, cli_dict  = process.read_cli()
    runmode   = cli_dict['runmode']
    args, cli = process.read_config(config_path=f'configs/hgcal', runmode=runmode)
    
    # Create trackster data
    data      = preprocess.event_loop(files=args['root_files'], maxevents=args['maxevents'], 
                    directed=args['graph_param']['directed'])
    X         = graphio.parse_graph_data_trackster(data=data, weights=None)

    # Pickle to disk
    path = aux.makedir(f"output/{args['rootname']}/{args['config']}")
    filename = f"{path}/{cli.tag}.pkl"
    with open(filename, "wb") as fp:
        pickle.dump(X, fp)

    print(__name__ + f' Saved output to "{filename}"')
    print(__name__ + f' [done]')

if __name__ == '__main__' :
   main()
