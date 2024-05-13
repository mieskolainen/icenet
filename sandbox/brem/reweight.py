################################################################################
# Imports

from argparse import ArgumentParser
import os
import pandas as pd
from utils import *

################################################################################
# CLI

parser = ArgumentParser()
parser.add_argument('--verbose',action='store_true')
parser.add_argument('--nevents',default=-1,type=int)
parser.add_argument('--reweight',action='store_true')
parser.add_argument('--nbins',default=0,type=int)
args = parser.parse_args()
print("Command line args:",vars(args))

################################################################################
# Variables

if args.verbose :
   print('Features:',columns)
   print('Additional:',additional)
   print('Labelling:',labelling)

################################################################################
# Parse files

icenet_base = os.environ['ICEPATH']
brem_base = f'{icenet_base}/sandbox/brem'
inputs = f'{brem_base}/inputs'
outputs = f'{brem_base}/outputs'
files = [f'{icenet_base}/travis-stash/input/icebrem/output_signal_10k.root']
print('Input files:', ', '.join(["'{:s}'".format(f) for f in files]))
df = parse(files,args.nevents,args.verbose)

################################################################################
# Preprocessing

df = preprocess(df)

# print summary info
if args.verbose :
   print(df.info())
   with pd.option_context('display.width',None,
                          'display.max_rows',None, 
                          'display.max_columns',None,
                          'display.float_format','{:,.2f}'.format) :
      print(df.describe(include='all').T)

################################################################################
# Reweighting (e.g. by pT,eta)

reweight_features = ['log_trk_pt','trk_eta']
nbins = args.nbins if args.nbins > 0 else None

# Determine weights and write to file
df,clusterizer,dct = extract_weights(
    df=df,
    reweight_features=reweight_features,
    nbins=nbins,
    base=f'{inputs}',
    filename='weights',
    write=True,
    verbose=args.verbose)

df,clusterizer,weights_dct = extract_weights(
    df=df,
    reweight_features=reweight_features,
    nbins=nbins,
    base=f'{inputs}',
    filename='weights',
    write=False,
    verbose=args.verbose)

plot_weights(
    df=df,
    clusterizer=clusterizer,
    weights=weights_dct['weights'],
    counts=weights_dct['counts'],
    reweight_features=weights_dct['features'],
    base=f'{outputs}',
    verbose=args.verbose)

# Unweighted
training_dct = train_bdt(
    df=df,
    reweight_features=weights_dct['features'],
    weights=None,
    verbose=args.verbose)

roc_curves(
    df=df,
    dct=training_dct,
    reweight_features=weights_dct['features'],
    title='roc_unweighted',
    base=f'{outputs}',
    verbose=args.verbose)

# Weighted
training_dct = train_bdt(
    df=df,
    reweight_features=weights_dct['features'],
    weights=weights_dct['weights'],
    verbose=args.verbose)

roc_curves(
    df=df,
    dct=training_dct,
    reweight_features=weights_dct['features'],
    title='roc_weighted',
    base=f'{outputs}',
    verbose=args.verbose)
