# Common input & data reading routine for train.py and eval.py
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

from icenet.tools import io

from electronid import ereader


def common():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type = str, default='default')
    parser.add_argument("--datapath", type = str, default=".")
    parser.add_argument("--datasets", type = str, default="0")

    cli = parser.parse_args()

    # Input is [0,1,2,..]
    cli.datasets = cli.datasets.split(',')

    ## Read configuration
    config_yaml_file = cli.config + '.yml'
    with open('./configs/' + config_yaml_file, 'r') as stream:
        try:
            args = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    args['config'] = cli.config
    print(args)
    print(torch.__version__)
    args['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu:0')
    
    ### Load data
    paths = []
    for i in cli.datasets:
        paths.append(cli.datapath + '/output_' + str(i) + '.root')

    # background (0) and signal (1)
    class_id = [0,1]
    data  = io.DATASET(func_loader = ereader.load_root_file, files = paths, class_id = class_id, frac = args['frac'], rngseed = args['rngseed'])

    return data, args
