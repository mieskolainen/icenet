import uproot as uproot
import awkward as ak
import pandas as pd
import numpy as np
import os
import json
import inspect
import joblib
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
#from icebrem_utils import roc_curves

################################################################################
# Variables

idx = 0
prefix = ['2019Aug07','eid'][idx]

features = [ # ORDER IS VERY IMPORTANT ! 
    f'{prefix}_rho',
    f'{prefix}_ele_pt',
    f'{prefix}_sc_eta',
    f'{prefix}_shape_full5x5_sigmaIetaIeta',
    f'{prefix}_shape_full5x5_sigmaIphiIphi',
    f'{prefix}_shape_full5x5_circularity',
    f'{prefix}_shape_full5x5_r9',
    f'{prefix}_sc_etaWidth',
    f'{prefix}_sc_phiWidth',
    f'{prefix}_shape_full5x5_HoverE',
    f'{prefix}_trk_nhits',
    f'{prefix}_trk_chi2red',
    f'{prefix}_gsf_chi2red',
    #f'{prefix}_brem_frac', #@@ this branch is broken, do not include???
    f'{prefix}_gsf_nhits',
    f'{prefix}_match_SC_EoverP',
    f'{prefix}_match_eclu_EoverP',
    f'{prefix}_match_SC_dEta',
    f'{prefix}_match_SC_dPhi',
    f'{prefix}_match_seed_dEta',
    f'{prefix}_sc_E',
    f'{prefix}_trk_p',
    f'{prefix}_gsf_bdtout1' if idx == 0 else 'gsf_bdtout1' ,
]

additional = [
    'gen_pt','gen_eta', 
    'trk_pt','trk_eta','trk_dr','trk_charge',
    'gsf_pt','gsf_eta','gsf_dr','gsf_bdtout2','gsf_mode_pt',
    'ele_pt','ele_eta','ele_dr',
    'ele_mva_value_2019Aug07' if idx == 0 else 'ele_mva_value',
    #'ele_mva_value_retrained','ele_mva_value_depth10','ele_mva_value_depth15',
    'run', 'lumi', 'evt',
    'weight','rho',
    'tag_pt','tag_eta',
    'gsf_dxy','gsf_dz','gsf_nhits','gsf_chi2red',
]

labelling = [
    'is_e','is_egamma',
    'has_trk','has_seed','has_gsf','has_ele',
]

columns = features + additional + labelling
columns = list(set(columns))

################################################################################
# Parse files

def parse(files,nevents=-1,verbose=False):

    if isinstance(files[0],str):
        for i,f in enumerate(files): files[i] = (None,f)

    df = None
    if verbose: print('Input files:')
    for ifile,(label,f) in enumerate(files):
        if verbose: print(f'  #{ifile}: {f} (class label = {label}')
        tree = uproot.open(f).get('ntuplizer/tree')
        print('Available branches: ',tree.keys())
        tmp = ak.to_dataframe(tree.arrays(columns))
        if   label == 1: tmp['label'] = True
        elif label == 0: tmp['label'] = False
        else:            tmp['label'] = tmp['is_e']
        if verbose : print(f'ifile={ifile:.0f}, file={f:s}, entries={tmp.shape[0]:.0f}')
        df = tmp if ifile == 0 else pd.concat([df,tmp])
        if nevents > 0 and df.shape[0] > nevents :
            df = df.head(nevents)
            print(f"Consider only first {nevents:.0f} events ...")
            break
    return df

################################################################################
# Preprocessing

def preprocess(df) :

    # Filter based on sample label
    keep = ( (df.label==True) & (df.is_e==True) ) | ( (df.label==False) & (df.is_e==False) )
    df = df[keep]

    # Filter based on tag muon pT and eta
    thr_tag_pt = 7.0
    thr_tag_eta = 1.5

    keep = (df.tag_pt < -9.999) & (df.tag_eta < -9.999)
    df = df[keep | ((df.tag_pt>thr_tag_pt)&(np.abs(df.tag_eta)<thr_tag_eta))]

    print(
        f"threshold_tag_pt: {thr_tag_pt:.2f}, "\
        f"threshold_tag_eta: {thr_tag_eta:.2f}, "\
        f"df.shape: {df.shape}")
    
    # Clip and take log of trk_pt
    df['trk_eta'] = df['trk_eta'].clip(lower=-3.) # clip to trk_eta > -3.
    df['trk_pt'] = df['trk_pt'].clip(upper=100.)  # clip trk_pt < 100.
    log_trk_pt = np.log10(df['trk_pt'])
    log_trk_pt[np.isnan(log_trk_pt)] = -1.
    df['log_trk_pt'] = log_trk_pt

    # Clip and take log of trk_pt
    df['gsf_eta'] = df['gsf_eta'].clip(lower=-3.) # clip to gsf_eta > -3.
    df['gsf_pt'] = df['gsf_pt'].clip(upper=100.)  # clip gsf_pt < 100.
    log_gsf_pt = np.log10(df['gsf_pt'])
    log_gsf_pt[np.isnan(log_gsf_pt)] = -1.
    df['log_gsf_pt'] = log_gsf_pt

    # uint64 for some event scalars
    df['run'] = pd.to_numeric(df['run'], errors="coerce").fillna(0).astype('uint64')
    df['lumi'] = pd.to_numeric(df['lumi'], errors="coerce").fillna(0).astype('uint64')
    df['evt'] = pd.to_numeric(df['evt'], errors="coerce").fillna(0).astype('uint64')

    # Dummy constant value 
    df['dummy'] = 0.
    df['dummy'] = df['dummy'].astype('float32')

    # Filter only electrons, based on pt and eta
    keep = (df.has_gsf == True) & (df.has_ele == True) & (df.trk_pt > 1.) & (df.trk_eta.abs() < 2.5)
    df = df[keep]

    # Sort and shuffle
    df = df[sorted(df.columns.tolist())] # sort cols by headers
    df = df.sort_values(by=['run','lumi','evt','rho']) # sort rows by scalars
    df = shuffle(df,random_state=0) # shuffle (if set, deterministic)

    return df

################################################################################
# Common (vectorized) method to apply weights 

apply_weight = np.vectorize(lambda x,y : y.get(x),excluded={1})

################################################################################
# Extract weights and add to data frame

def extract_weights(
    df,
    reweight_features=['log_trk_pt','trk_eta'],
    nbins=None,
    base='.',
    filename='weights',
    write=True,
    verbose=False) :

    print(f'Accessing files in dir "{base}"...')
    if not os.path.isdir(base) : os.makedirs(base)
    
    # Remove filename suffix, if any
    filename = filename.split(".")[0]

    if nbins == None: nbins = int( len(df.index) / 100. ) # an average of 100 counts (S+B) per bin
    print(f'Attempting to cluster {len(df.index)} entries into {nbins} bins ...')
    
    # Determine clusters
    print('Obtaining clusterizer model...')
    if write == True:
        print(f'Training clusterizer using entries in the data frame...')
        clusterizer = MiniBatchKMeans(
            n_clusters=nbins,
            #init='random',
            #max_no_improvement=None,
            #batch_size=3000,
            #n_jobs=3,
            random_state=0, # if set, deterministic
            verbose=verbose,
            )
        clusterizer.fit(df[reweight_features])
        print(f'Writing clusterizer model to file "{base}/{filename}.pkl"...')
        joblib.dump(clusterizer,f'{base}/{filename}.pkl',compress=True)
    else:
        print(f'Reading clusterizer model from file "{base}/{filename}.pkl"...')
        clusterizer = joblib.load(f'{base}/{filename}.pkl')

    print(f'Evaluate clusterizer for all entries in the data frame...')
    df['cluster'] = clusterizer.predict(df[reweight_features])

    # Determine weights and counts for each cluster
    print('Obtaining weights...')
    dct = {'weights':{},'counts':{},'features':[]}
    if write == True:
        groups = df.groupby('cluster')
        n_sig_tot = 0
        n_bkg_tot = 0
        for cluster,group in groups :
            n_sig = group.is_e.sum()
            n_bkg = np.invert(group.is_e).sum()
            if n_sig == 0 : RuntimeError(f'Bin {cluster} has no signal events, reduce the number of bins!')
            if n_bkg == 0 : RuntimeError(f'Bin {cluster} has no bkgd events, reduce the number of bins!')
            dct['weights'][int(cluster)] = float(n_bkg)/float(n_sig) if n_sig > 0 else 1.
            dct['counts'][int(cluster)] = float(min(n_sig,n_bkg))
            n_sig_tot += n_sig
            n_bkg_tot += n_bkg
        dct['features'] = reweight_features
        with open(f'{base}/{filename}.json','w') as f : json.dump(dct,f)
        print(f'Written weights to file "{base}/{filename}.json"')
    else:
        with open(f'{base}/{filename}.json') as f : dct = json.load(f)
        dct['weights'] = { int(key):float(val) for key, val in dct['weights'].items() }
        dct['counts'] = { int(key):float(val) for key, val in dct['counts'].items() }
        print(f'Read weights from file "{base}/{filename}.json"')

    # Apply weights to df via vectorized method
    print('Applying weights...')
    df['weight'] = np.invert(df.is_e) + df.is_e * apply_weight(df['cluster'],dct['weights'])

    # Return: data frame with new weights branch; clusterizer model; dct (containing weights, counts, and features)
    return df,clusterizer,dct

################################################################################
# Plot weights

def plot_weights(
    df,
    clusterizer,
    weights,
    counts,
    reweight_features = ['log_trk_pt','trk_eta'],
    base='.',
    verbose=False) :

    print(f'Producing plots in directory "{base}"...')
    if not os.path.isdir(base) : os.makedirs(base)

    if len(reweight_features) == 1:
        reweight_features.append('dummy')

    x_feature = reweight_features[0]
    y_feature = reweight_features[1]

    ##########
    # Used to plot the decision boundary (assign colour to each)
    mesh_size = 0.01 # Fine granularity
    x_min,x_max = df[x_feature].min()-0.3, df[x_feature].max()+0.3
    y_min,y_max = df[y_feature].min()-0.3, df[y_feature].max()+0.3
    xx,yy = np.meshgrid(np.arange(x_min,x_max,mesh_size,dtype=np.float32),
                        np.arange(y_min,y_max,mesh_size,dtype=np.float32))
    
    print(f'Evaluate clusterizer for chosen binning...')
    cluster = clusterizer.predict(np.c_[xx.ravel(),yy.ravel()]) # Evaluate (for each point in the mesh)

    ##########
    # Plot (2D) the decision boundaries (i.e. binning)
    filename='weights_binning.png'
    print(f'Creating "{filename}" in directory "{base}"...')
    Z = cluster.reshape(xx.shape)
    plt.figure(figsize=[8,8])
    plt.imshow(
        Z, 
        interpolation='nearest',
        extent=(xx.min(),xx.max(),yy.min(),yy.max()),
        cmap=plt.cm.tab10, # plt.cm.Paired,
        aspect='auto', 
        origin='lower')
    plt.title('binning')
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.savefig(f'{base}/{filename}',bbox_inches='tight')
    plt.clf()

    ##########
    # Plot (2D) the weights per bin
    filename='weights_values.png'
    print(f'Creating "{filename}" in directory "{base}"...')
    Z = apply_weight(cluster,weights).reshape(xx.shape)
    plt.figure(figsize=[8,8])
    plt.imshow(
        Z, 
        interpolation='nearest',
        extent=(xx.min(),xx.max(),yy.min(),yy.max()),
        cmap=plt.cm.coolwarm, # plt.cm.seismic,
        norm=LogNorm(vmin=max(min(weights.values()),1.e-4),vmax=min(max(weights.values()),1.e4)),
        aspect='auto', 
        origin='lower')
    plt.title('weights')
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.colorbar()
    plt.savefig(f'{base}/{filename}',bbox_inches='tight')
    plt.clf()
    
    ##########
    # Plot (2D) event counts per bin
    filename='weights_counts.png'
    print(f'Creating "{filename}" in directory "{base}"...')
    Z = apply_weight(cluster,counts).reshape(xx.shape)
    plt.figure(figsize=[8,8])
    plt.imshow(
        Z, 
        interpolation='nearest',
        extent=(xx.min(),xx.max(),yy.min(),yy.max()),
        cmap=plt.cm.Reds, # plt.cm.seismic,
        norm=LogNorm(vmin=0.1,vmax=max(counts.values())),
        aspect='auto', 
        origin='lower')
    plt.title('min(counts(sig,bkgd))')
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.colorbar()
    plt.savefig(f'{base}/{filename}',bbox_inches='tight')
    plt.clf()

    ##########
    # Plot (1D) distribution of weights
    filename='weights_distribution.png'
    print(f'Creating "{filename}" in directory "{base}"...')
    plt.figure(figsize=[8,8])
    xmin = df['weight'][df['weight']>0.].min()*0.3
    xmax = df['weight'].max()*3.
    bins = np.logspace(np.log(xmin),np.log(xmax),100)
    entries,_,_ = plt.hist(df['weight'],bins,histtype='stepfilled')
    plt.title('')
    plt.xlabel('weight')
    plt.ylabel('a.u.')
    plt.xlim(xmin,xmax)
    plt.ylim(0.3,entries.max()*3.)
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.savefig(f'{base}/{filename}',bbox_inches='tight')
    plt.clf()

    ##########
    # Plot (1D) distribution of counts
    filename='counts_distribution.png'
    print(f'Creating "{filename}" in directory "{base}"...')
    plt.figure(figsize=[8,8])
    values = np.array(sorted(list(counts.values())))
    xmin = 0.
    xmax = int(values.max()*1.1)
    bins = np.linspace(xmin,float(xmax),xmax)
    entries,_,_ = plt.hist(values,bins,histtype='stepfilled')
    plt.title('')
    plt.xlabel('min(counts(sig,bkgd))')
    plt.ylabel('a.u.')
    plt.xlim(xmin,xmax)
    plt.ylim(0.,entries.max()*1.1) #0.3,entries.max()*3.)
    #plt.gca().set_yscale('log')
    plt.savefig(f'{base}/{filename}',bbox_inches='tight')
    plt.clf()

    ##########
    # Plot (1D) weighted and unweighted distributions for each kinematical variable
    for var in reweight_features:
        filename=f'weights_{var:s}.png'
        print(f'Creating "{filename}" in directory "{base}"...')
        xmin = min(df[df.is_e][var].min(),df[np.invert(df.is_e)][var].min())
        xmax = max(df[df.is_e][var].max(),df[np.invert(df.is_e)][var].max())
        kwargs={'bins':20,'density':True,'histtype':'step','range':(xmin,xmax),'linewidth':2}
        plt.hist(
            df[df.is_e][var],
            color='green',ls='dashed',label='signal (unweighted)',
            **kwargs)
        plt.hist(
            df[np.invert(df.is_e)][var],
            color='red',ls='solid',label='bkgd',
            **kwargs)
        plt.hist(
            df[df.is_e][var],
            weights=df['weight'][df.is_e],
            color='green',ls='solid',label='signal (weighted)',
            **kwargs)
        plt.legend(loc='best')
        plt.xlabel(var)
        plt.ylabel('a.u.')
        plt.gca().set_yscale('log')
        plt.savefig(f'{base}/{filename}',bbox_inches='tight')
        plt.clf()

################################################################################
# ROC curves

def train_bdt(
    df,
    reweight_features = ['log_trk_pt','trk_eta'],
    weights=None,
    verbose=False) :

    features = ", ".join([x for x in reweight_features])
    print(f'Training BDT to discriminate based on: "{features}"...')

    # Split data set
    trn,val = train_test_split(
        df,
        test_size=0.2,
        shuffle=True,
        random_state=0)
    trn,tst = train_test_split(
        trn,
        test_size=0.2,
        shuffle=False,
        random_state=0)
    
    if verbose :
        print(f'Data set sizes: trn={trn.shape[0]}, tst={tst.shape[0]}, val={val.shape[0]}')

    # Consider just kinematic variables (and label)
    X_trn,y_trn,w_trn = trn[reweight_features], trn.is_e.astype(int), trn.weight.astype(float)
    X_tst,y_tst,w_tst = tst[reweight_features], tst.is_e.astype(int), tst.weight.astype(float)
    X_val,y_val,w_val = val[reweight_features], val.is_e.astype(int), val.weight.astype(float)

    # Train to discriminate based on (kinematical) 'reweight features'
    kwargs = {
        'eval_metric':['logloss','auc'],
        'early_stopping_rounds':10
        }
    clf = xgb.XGBClassifier(**kwargs)
    kwargs = {
        'eval_set':[(X_trn,y_trn),(X_tst,y_tst)],
        }
    clf.fit(X_trn,y_trn,**kwargs)

    # Evaluate
    score_trn = clf.predict_proba(X_trn)[:,-1]
    score_tst = clf.predict_proba(X_tst)[:,-1]
    score_val = clf.predict_proba(X_val)[:,-1]

    dct = {
        'trn':{'label':y_trn,'score':score_trn,'vars':X_trn[reweight_features],'weight':w_trn if weights != None else None},
        'tst':{'label':y_tst,'score':score_tst,'vars':X_tst[reweight_features],'weight':w_tst if weights != None else None},
        'val':{'label':y_val,'score':score_val,'vars':X_val[reweight_features],'weight':w_val if weights != None else None},
        }
    return dct

################################################################################
# ROC curves

def roc_curves(
    df,
    dct={
        'trn':{'label':None,'score':None,'vars':None,'weight':None},
        'tst':{'label':None,'score':None,'vars':None,'weight':None},
        'val':{'label':None,'score':None,'vars':None,'weight':None},
        },
    reweight_features = ['log_trk_pt','trk_eta'],
    title='test',
    base='.',
    verbose=False) :

    print(f'Producing plots in directory "{base}"...')
    if not os.path.isdir(base) : os.makedirs(base)
    
    # ROC curves
    plt.clf()
    plt.figure(figsize=[8,8])
    
    x = np.logspace(-4.,0.,100)
    plt.plot(x,x,linestyle='--',color='black',linewidth=1) # by chance

    # Train ROC 
    if dct['trn']['label'] is not None and dct['trn']['score'] is not None :
        fpr,tpr,thresholds = roc_curve(
            y_true=dct['trn']['label'],
            y_score=dct['trn']['score'],
            sample_weight=dct['trn']['weight'])
        auc = roc_auc_score(
            dct['trn']['label'],
            dct['trn']['score'],
            sample_weight=dct['trn']['weight'])
        plt.plot(fpr,tpr,label=f'Train (AUC={auc:5.3f})')

    # Test ROC 
    if dct['tst']['label'] is not None and dct['tst']['score'] is not None :
        fpr,tpr,thresholds = roc_curve(
            y_true=dct['tst']['label'],
            y_score=dct['tst']['score'],
            sample_weight=dct['tst']['weight'])
        auc = roc_auc_score(
            dct['tst']['label'],
            dct['tst']['score'],
            sample_weight=dct['tst']['weight'])
        plt.plot(fpr,tpr,label=f'Test (AUC={auc:5.3f})')

    # Validation ROC 
    if dct['val']['label'] is not None and dct['val']['score'] is not None :
        fpr,tpr,thresholds = roc_curve(
            y_true=dct['val']['label'],
            y_score=dct['val']['score'],
            sample_weight=dct['val']['weight'])
        auc = roc_auc_score(
            dct['val']['label'],
            dct['val']['score'],
            sample_weight=dct['val']['weight'])
        plt.plot(fpr,tpr,label=f'Validation (AUC={auc:5.3f})')

    # ROC for each (transformed) variable
    for var in reweight_features:

        # transform
        if 'eta' in var:
            tmp = dct['val']['vars'][var]
            dct['val']['vars'][var] = 0.-np.abs(tmp) # more than abs(var)
        if 'rho' in var:
            tmp = dct['val']['vars'][var]
            dct['val']['vars'][var] = 0.-tmp # less than var

        fpr,tpr,thresholds = roc_curve(
            y_true=dct['val']['label'],
            y_score=dct['val']['vars'][var],
            sample_weight=dct['val']['weight'])
        auc = roc_auc_score(
            dct['val']['label'],
            dct['val']['vars'][var],
            sample_weight=dct['val']['weight'])
        plt.plot(fpr,tpr,label=f'{var} (AUC={auc:5.3f})')
      
    plt.title(title)
    plt.ylabel('Efficiency')
    plt.xlabel('Mistag rate')
    plt.legend(loc='best')
    plt.xlim(0.,1.)
    plt.ylim(0.,1.)
    plt.savefig(f'{base}/{title:s}.png',bbox_inches='tight')
    plt.gca().set_xscale('log')
    plt.xlim(1.e-4,1.)
    plt.savefig(f'{base}/{title:s}_logy.png',bbox_inches='tight')
    plt.clf()
