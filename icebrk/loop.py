# B/RK analyzer main event loop [CODE IS ROTTEN / UNFIXED due to awkward0 --> awkward1 changes]
#
#
# Mikael Mieskolainen, 2022
# m.mieskolainen@imperial.ac.uk


import copy
import uproot
import h5py
import awkward

import numpy as np
from tqdm import tqdm
from termcolor import colored

from iceplot import iceplot
from icenet.tools.io import *
from icenet.tools import aux
from icenet.tools import prints

import icebrk.features as features
import icebrk.tools as tools
import icebrk.histos as histos
import icebrk.cutstats as cutstats
import icebrk.fasthistos as fasthistos


def hdf5_write_handles(filename, N_weights, rwmode = 'w'):
    """Create HDF5 file handles.

    Args:
        filename:
        N_weights:
        rwmode

    Returns:
        f:
        datasets
    """

    f = h5py.File(filename, rwmode)

    # Keys: Observables
    datasets = dict.fromkeys(histos.obs_all.keys(), 0)
    init_shape = (0,)
    for key in datasets.keys():
        datasets[key] = f.create_dataset(key, shape=init_shape, maxshape=(None,),
            dtype=np.float32, compression='gzip', compression_opts=9)

    # Keys: Weights
    for i in range(N_weights):
        key = 'W'+str(i)
        datasets[key] = f.create_dataset(key, shape=init_shape, maxshape=(None,),
            dtype=np.float32, compression='gzip', compression_opts=9)

    return f, datasets


def hdf5_append(datasets, key, chunk):
    """Append chunk of data to HDF5 file.
    
    Args:
        datasets:
        key:
        chunk:

    Returns:
        f:
        datasets
    """

    print(datasets[key])
    row_count = datasets[key].size

    # Resize the dataset to accommodate the next chunk of rows
    datasets[key].resize((row_count + chunk.shape[0],))

    # Write the next chunk
    datasets[key][row_count:] = chunk
    
    # Increment the row count
    row_count += chunk.shape[0]

    return row_count



'''
--------------------------------------------------------------------------

y_tru = [1, 0, 0, 0, 0]
y_est = [0.9, 0.05, 0.0001, 0.00001, 0.05]

Scheme 1: MAXN = 1

BMAT =
[0 0 0 0] = class 0 (no signal)
[0 0 0 1] = class 1 (signal in last slot)
[0 0 1 0] = class 2  ...
[0 1 0 0] = class 3  ...
[1 0 0 0] = class 4 (signal in first slot)


--------------------------------------------------------------------------
Scheme 2: MAXN <= MAXT3

BMAT =
[0 0 0 0] = class 0 (no signal)
[0 0 0 1] = class 1 (signal in last slot)
[0 0 1 0] = class 2 (signal in second last slot)
[0 0 1 1] = class 3 (signal objects in two slots)
...

'''
def poweranalysis(evt_index, batch_obs, obs, func_predict, x, y, qsets, MAXT3, MAXN, isMC, reco, BMAT, WNORM):
    """Powerset analysis of the event
    N.B.this is already CONDITIONED that we select a maximum MAXT3 triplets!
    
    Args:
        evt_index:
        batch_obs:
        obs:
        func_predict:
        x:
        y:
        qsets:
        MAXT3:
        MAXN:
        isMC:
        reco:
        BMAT:
        WNORM:

    Returns:
        w:
    """

    # Create weights
    w = dict()
    if isMC:
        w['0'] = np.zeros(BMAT.shape[0])
        for c in range(BMAT.shape[0]):
            w['0'][c] = 1 if aux.binvec_are_equal(y, BMAT[c,:]) else 0
    else:
        for i in range(len(func_predict)):
            w[str(i)] = (func_predict[i](np.array([x]))).flatten()

    # Get indices we use for the kinematic reconstruction [UPDATE HERE **** !!!]
    set_ind = tools.get_first_indices(qsets, MAXT3=MAXT3)


    # Over all powerset combinations (one object, two objects, ... )
    # with the total number being dependent on number of rows in BMAT
    wnorm = {'S': 0, 'B': 0}

    for c in range(BMAT.shape[0]):

        # Event (object) normalization strategy
        if   WNORM == 'event':
            wnorm['S'] = np.sum(BMAT[c,:] == 1) # Number of signal objects
            wnorm['B'] = np.sum(BMAT[c,:] == 0) # Number of background objects
            
        elif WNORM == 'unit':
            wnorm['S'] = wnorm['B'] = 1.0

        else:
            raise Exception(__name__ + '.poweranalysis: unknown WNORM parameter')

        # Sweep over all triplets of this combination
        for j in range(len(BMAT[c,:])):

            # SIGNAL / BACKGROUND COMPONENT ACTIVE (= POWERSET DEFINITION)
            ID = 'S' if (BMAT[c,j]) else 'B'

            # Computed observables
            for key in histos.obs_all.keys():

                if   key in batch_obs:

                    """
                    print(key)
                    print(evt_index)
                    print(set_ind[j])
                    
                    print(batch_obs[key][evt_index])
                    print(batch_obs[key][evt_index][set_ind[j]])
                    """
                    xx = awkward.to_list( batch_obs[key][evt_index][set_ind[j]] )

                    # ** SOMETHING IS BROKEN HERE, WE SHOULD NOT HAVE A LIST AT THIS POINT ** 
                    if isinstance(xx, list):
                        reco[ID]['x'][key].add( xx[0] )
                    else:
                        reco[ID]['x'][key].add( xx )
                
                elif key in obs:
                    reco[ID]['x'][key].add( obs[key] )

            # Weights
            if isMC:
                reco[ID]['w']['0'].add(w['0'][c] / wnorm[ID])

            else:
                for algo in w.keys():
                    reco[ID]['w'][algo].add(w[algo][c] / wnorm[ID])
    return w


def hist_flush(reco, hobj, h5datasets = None):
    """ Histogram observables with accumulation of previous histograms, and flush buffer arrays
    
    Args:
        reco:
        hobj:
        h5datasets:

    Returns:
        w:
    """

    # ====================================================================
    ### Accumulate histograms

    for obs in reco['x'].keys():
        if (reco['x'][obs].size > 0): # Histogram only non-empty buffer

            for algo in reco['w'].keys():
                hobj[algo][obs] += iceplot.hist_obj(x = reco['x'][obs].values(),
                    weights=reco['w'][algo].values(), bins=histos.obs_all[obs]['bins'])

    # ====================================================================
    ### Dump data to HDF5 files

    if h5datasets is not None:

        for obs in reco['x'].keys():
            if histos.obs_all[obs]['pickle']:
                hdf5_append(datasets=h5datasets, key=obs, chunk=reco['x'][obs].values())

        for algo in reco['w'].keys():
            hdf5_append(datasets=h5datasets, key='W'+algo, chunk=reco['w'][algo].values())

    # ====================================================================
    ### Flush buffers

    # Observables
    for obs in reco['x'].keys():
        reco['x'][obs].reset()

    # Weight
    for algo in reco['w'].keys():
        reco['w'][algo].reset()
    
    # ====================================================================


def initarrays(BUFFER, func_predict, isMC):
    """ Init histogramming arrays and objects
    
    Args:
        BUFFER:
        func_predict:
        isMC:

    Returns:
        reco:
        hobj:
    """

    reco_EMPTY = {
        'x' : dict.fromkeys(histos.obs_all.keys()),
        'w' : dict()
    }
    for obs in histos.obs_all.keys():
        reco_EMPTY['x'][obs] = fastarray1(BUFFER)

    if isMC:
        reco_EMPTY['w']['0'] = fastarray1(BUFFER)
    else:
        for i in range(len(func_predict)):
            reco_EMPTY['w'][str(i)] = fastarray1(BUFFER)


    ## Final state event buffers
    reco      = {'S': None, 'B': None}
    reco['S'] = copy.deepcopy(reco_EMPTY)
    reco['B'] = copy.deepcopy(reco_EMPTY)


    ## Final state histogram objects
    hobj = {'S': dict(), 'B': dict()}

    if isMC:
        hobj['S']['0'] = dict.fromkeys(histos.obs_all.keys(), iceplot.hobj())
        hobj['B']['0'] = dict.fromkeys(histos.obs_all.keys(), iceplot.hobj())
    
    else:
        for i in range(len(func_predict)):
            algo = str(i)
            hobj['S'][algo] = dict.fromkeys(histos.obs_all.keys(), iceplot.hobj())
            hobj['B'][algo] = dict.fromkeys(histos.obs_all.keys(), iceplot.hobj())

    return reco, hobj


def process(paths=[], func_predict=None, isMC=True, MAXT3=5, MAXN=2, MAXEVENTS=10000000000,
    EVTGROUPSIZE=1024, CHUNKBUFFER=512, VERBOSE=False, BMAT=[], WNORM=[], SUPERSETS=True,
    hd5dir=None, outputXY=False, outputP=False, **kwargs):
    """ Main event processing loop
    
    Args:

    Returns:

    """

    # Check critical inputs

    if isMC is False and (func_predict is None or func_predict is []):
        raise Exception(__name__ + '.process: input ''func_predict'' should not be empty')

    # --------------------------------------------------------------------
    # @ Basic variables @

    NCLASS = BMAT.shape[0]       # Number of classes
    D = features.getdimension()  # Numbef of dimensions per triplet
    DTOT = D * MAXT3             # Total number of dimensions

    SORTKEY = 'BToKEE_fit_cos2D' # Triplet ranking variable


    # --------------------------------------------------------------------
    # @ HDF5 dump @

    h5file     = {'S': None, 'B': None}
    h5datasets = {'S': None, 'B': None}

    if hd5dir is not None:

        if isMC:
            h5file['S'], h5datasets['S'] = hdf5_write_handles(filename=hd5dir + '/MC_S.hd5', N_weights=1)
            h5file['B'], h5datasets['B'] = hdf5_write_handles(filename=hd5dir + '/MC_B.hd5', N_weights=1)
        else:
            h5file['S'], h5datasets['S'] = hdf5_write_handles(filename=hd5dir + '/DA_S.hd5', N_weights=len(func_predict))
            h5file['B'], h5datasets['B'] = hdf5_write_handles(filename=hd5dir + '/DA_B.hd5', N_weights=len(func_predict))


    # --------------------------------------------------------------------
    # @ MC AI/ML-training arrays @
    X = None
    Y = None
    P = None
    x = None
    y = None

    if outputXY:
        X = np.zeros(shape=(MAXEVENTS, DTOT))

    if outputP:
        P = dict()
        for i in range(len(func_predict)): P[str(i)] = np.zeros(shape=(MAXEVENTS, NCLASS))
    
    if isMC:
        Y = np.zeros(shape=(MAXEVENTS, MAXT3))


    # --------------------------------------------------------------------
    # @ CUTS AND INFOFLOW @

    cutflow, infostats_BC, mcinfostats_BC = cutstats.init_stat_objects()
    infostats_AC   = copy.deepcopy(infostats_BC)
    mcinfostats_AC = copy.deepcopy(mcinfostats_BC)


    # --------------------------------------------------------------------
    # @ WEIGHTED OBSERVABLE CONTAINERS @

    T = CHUNKBUFFER * (BMAT.shape[0] * BMAT.shape[1])
    reco, hobj = initarrays(BUFFER=T, func_predict=func_predict, isMC=isMC)


    # --------------------------------------------------------------------
    # @ FAST HISTOGRAMS @

    #hobjraw_MC = fasthistos.initialize()
        

    # --------------------------------------------------------------------
    # @ LOOP VARIABLES @

    passcount = 0
    evtcount  = 0

    l1_p4 = dict()
    l2_p4 = dict()
    k_p4  = dict()


    # ==================================================================== ROOT file loop >>

    def print_input(events):
        prints.printbar()
        print(__name__ + f'.process: Input: {paths[z]}')
        print(__name__ + f'.process: {events.name} {events.title}')

        print(__name__ + f'.process: D          = {features.getdimension()}')
        print(__name__ + f'.process: DTOT       = {DTOT} (D x MAXT3)')
        print(__name__ + f'.process: NCLASS     = {NCLASS}')
        prints.printbar()
        print(__name__ + f'.process: MAXT3      = {MAXT3}')
        print(__name__ + f'.process: MAXN       = {MAXN}')
        print(__name__ + f'.process: WNORM      = {WNORM}')
        print(__name__ + f'.process: SUPERSETS  = {SUPERSETS}')
        print('')
        print(__name__ + f'.process: MAXEVENTS  = {MAXEVENTS}')
        prints.printbar()
    

    # Loop over different files
    for z in range(len(paths)):

        file = uproot.open(paths[z])
        print(file)
        events = file['Events']

        if z == 0:
            events.show()
            print_input(events)

        ## Loop over this file
        skips = 0
        for evt_i, evtgroup in enumerate(events.iterate(step_size = EVTGROUPSIZE)):

            print(f'evtgroup = {evt_i} [EVTGROUPSIZE = {EVTGROUPSIZE}]')
            d = copy.deepcopy(evtgroup)

            """
            if not isMC:
                if skips < 1:
                    # Skip events hack
                    print('Skipping evtgroup due to simulation')
                    skips += 1
                    continue;
                else:
                    True
            """
            
            # --------------------------------------------------------------------
            # *** FULLY VECTORIZED KINEMATICS CONSTRUCTION (JAGGED/COLUMNAR) ***

            ## Construct new branches
            features.construct_new_branches(d)

            ## Construct kinematics (4-vectors)
            tools.construct_kinematics(d, l1_p4, l2_p4, k_p4)

            ## Construct final state observables
            batch_obs = histos.calc_batch_observables(l1_p4, l2_p4, k_p4)
            

            if isMC:
                tools.construct_MC_tree(d)
                tools.construct_MC_truth(d)
                batch_MC_obs = histos.calc_batch_MC_observables(d, l1_p4, l2_p4, k_p4)

            # --------------------------------------------------------------------
            # Loop over events of this group
            N_EVENTS = len(d['event'])
            iterator = tqdm(range(N_EVENTS))

            for evt_index in iterator:

                if (evtcount == MAXEVENTS): 
                    print(colored(__name__ + f': Maximum event count {MAXEVENTS} reached', 'red'))
                    iterator.close() # So we don't get ghost messages
                    break # this event group
                    
                # ====================================================================
                ## EVENT SELECTION AND COMBINATORICS
                # ====================================================================
                evtcount += 1
                
                ## Find triplets merged together
                if SUPERSETS:
                    qsets = aux.los2lol(tools.find_connected_triplets(evt_index, l1_p4, l2_p4, k_p4))

                # All individually, no fusion
                else:
                    qsets = []
                    for kk in range( len(awkward.to_list(l1_p4['e'][evt_index])) ):
                        qsets.append( [kk] )
                

                # Sort the triplets
                def svprob_rank(tripletset):
                    for ind in tripletset:
                        # Return first of the (possible) superset [** SOMETHING BROKEN HERE PROBABLY **]
                        return d[SORTKEY][evt_index][ind]

                qsets.sort(key=svprob_rank, reverse=True) # Reverse gives the biggest values first!


                # ====================================================================
                # FILL FAST HISTOGRAMS

                '''
                # Get indices we use for the kinematic reconstruction [UPDATE HERE **** !!!]
                set_ind = tools.get_first_indices(qsets, MAXT3=MAXT3)

                for obs in fasthistos['S'].keys():
                    for set_ind
                '''

                # ====================================================================
                ## Construct input (and output) vectors

                x = tools.construct_input_vec(evt_index, d, l1_p4=l1_p4, l2_p4=l2_p4, k_p4=k_p4, qsets=qsets, MAXT3=MAXT3)
                if outputXY:
                    X[passcount,:] = copy.deepcopy(x)
                
                if isMC:
                    y = tools.construct_output_vec(evt_index, d, qsets=qsets, MAXT3=MAXT3)
                    if outputXY:
                        Y[passcount,:] = copy.deepcopy(y)


                # ====================================================================
                ## CUTS
                # ====================================================================
                cutstats.collect_info_stats(d = d, evt_index = evt_index, infostats = infostats_BC)
                if isMC: cutstats.collect_mcinfo_stats(d=d, evt_index=evt_index, y=y,
                    qsets=qsets, MAXT3=MAXT3, mcinfostats=mcinfostats_BC)
                
                # --------------------------------------------------------------------
                if (cutstats.apply_cuts(d, evt_index, cutflow) == False):
                    continue
                # --------------------------------------------------------------------
                cutstats.collect_info_stats(d = d, evt_index = evt_index, infostats = infostats_AC)
                if isMC: cutstats.collect_mcinfo_stats(d=d, evt_index=evt_index, y=y,
                    qsets=qsets,MAXT3=MAXT3, mcinfostats=mcinfostats_AC)
                    

                # ====================================================================
                # Event-by-event (non-vectorizable) observables

                obs        = histos.calc_observables(evt_index, d, l1_p4, l2_p4, k_p4, qsets, MAXT3)
                if isMC:
                    MC_obs = histos.calc_MC_observables(evt_index, d, l1_p4, l2_p4, k_p4, qsets, MAXT3)
                
                if VERBOSE:
                    tools.print_MC_event(evt_index, d, l1_p4, l2_p4, k_p4, qsets)


                # ====================================================================
                ## ANALYSIS
                # ====================================================================
                if func_predict is not None:

                    ## Analyze
                    w = poweranalysis(evt_index=evt_index, batch_obs=batch_obs, obs=obs, func_predict=func_predict,
                        x=x, y=y, qsets=qsets, MAXT3=MAXT3, MAXN=MAXN, reco=reco, isMC=isMC, BMAT=BMAT, WNORM=WNORM)

                    if outputP:
                        for i in range(len(func_predict)): P[str(i)][passcount,:] = w[str(i)]

                    # Fill histograms & flush buffers
                    if ((passcount > 0) & (passcount % (CHUNKBUFFER-1) == 0)):
                        for ID in reco.keys(): hist_flush(reco=reco[ID], hobj=hobj[ID], h5datasets=h5datasets[ID])

                ## Update counts
                passcount += 1

        # Last fill histograms & flush buffers
        for ID in reco.keys(): hist_flush(reco=reco[ID], hobj=hobj[ID], h5datasets=h5datasets[ID])

    # ==================================================================== << ROOT file loop

    # Close HDF5 files
    if hd5dir is not None:
        for ID in reco.keys(): h5file[ID].close()

    ## Remove buffer space
    if outputXY:
        X = X[0:passcount-1,:]
        if isMC:
            Y = Y[0:passcount-1,:]

    if outputP:
        for i in range(len(func_predict)):
            P[str(i)] = P[str(i)][0:passcount-1,:]


    ## Update total event counts
    cutflow['total']      = evtcount
    infostats_BC['total'] = evtcount
    infostats_AC['total'] = evtcount

    if isMC:
        mcinfostats_BC['total'] = evtcount
        mcinfostats_AC['total'] = evtcount

    ## Print flows
    print('\n<< INFOSTATS [BC] >>'); prints.print_flow(infostats_BC)
    if isMC: print('\n<< MC-ONLY-INFOSTATS [BC] >>'); prints.print_flow(mcinfostats_BC)
    print('\n<< CUTFLOW >>'); prints.print_flow(cutflow)
    print('\n<< INFOSTATS [AC] >>'); prints.print_flow(infostats_AC)
    if isMC: print('\n<< MC-ONLY-INFOSTATS [AC] >>'); prints.print_flow(mcinfostats_AC)
    print('\n')
    
    output = {
        'X'    : X,
        'Y'    : Y,
        'P'    : P,
        'hobj' : hobj
    }
    
    return output