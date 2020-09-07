# B/RK analyzer common input & data reading routine
#
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk


import math
import numpy as np
import torch
import argparse
import pprint
import os
import datetime
import json
import pickle
import sys
import yaml

import uproot


def init():
    """ Data initialization function.

    Args:
        None

    Returns:
        paths :
        args  :
        cli   :
        iodir :

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type = str, default='default')
    parser.add_argument("--datapath", type = str, default=".")
    parser.add_argument("--datasets", type = str, default="0")
    parser.add_argument("--tag",      type = str, default='tag0')

    cli = parser.parse_args()

    
    # Input is [0,1,2,..]
    cli.datasets = cli.datasets.split(',')

    ## Read configuration
    config_yaml_file = cli.config + '.yml'
    with open('./configs/brk/' + config_yaml_file, 'r') as stream:
        try:
            args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    args['config'] = cli.config
    print(args)
    print(torch.__version__)
    
    ### Load data
    paths = []
    for i in cli.datasets:
        paths.append(cli.datapath + '/BParkNANO_mc_relaxed_Kee_' + str(i) + '.root')
    
    ### Input-Output directory
    iodir = f'./output/brk/{cli.tag}/'; os.makedirs(iodir, exist_ok = True)

    return paths,args,cli,iodir
