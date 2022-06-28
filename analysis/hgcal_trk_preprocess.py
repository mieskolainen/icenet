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
    
    args, cli = process.read_config(config_path='./configs/hgcal')
    
    # Create trackster data
    data      = preprocess.event_loop(files=args['root_files'], maxevents=args['maxevents'])
    X         = graphio.parse_graph_data_trackster(data=data, weights=None)

    # Pickle to disk
    aux.makedir("./output/")
    outputfile  = "output/" + f"{cli.tag}.pkl"
    with open(outputfile, "wb") as fp:
        pickle.dump(X, fp)

    print(__name__ + f' Saved output to <{outputfile}>')
    print(__name__ + f' [done]')

if __name__ == '__main__' :
   main()
