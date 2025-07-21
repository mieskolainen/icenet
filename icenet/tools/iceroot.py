# ROOT file processing tools
#
# m.mieskolainen@imperial.ac.uk, 2025

import numpy as np
import uproot
import awkward as ak
import ray
import multiprocessing
import os
import copy
import gc
import time
from tqdm import tqdm

from icenet.tools import io
from icenet.tools import aux
from icenet.tools import iceroot

# ------------------------------------------
from icenet import print
# ------------------------------------------


def read_single(process_func, process, root_path, param, class_id, dtype=None, num_cpus=0, verbose=False):
    """
    Loop over different MC / data processes as defined in the yaml files
    
    [awkward compatible only]
    
    Args:
        process_func:  data processing function
        process:       MC / data process dictionary (from yaml)
        root_path:     main path of files
        param:         parameters of 'process_func'
        class_id:      class identifier (integer), e.g. 0, 1, 2 ...
        num_cpus:      number of CPUs used (set 0 for automatic)
        verbose:       verbose output print
    
    Returns:
        X, Y, W, ids, info (awkward array format)
    """

    print(f'{process}')

    # --------------------
    
    datasets        = process['path'] + '/' + process['files']
    xs              = process['xs']
    model_param     = process['model_param'] if 'model_param' in process.keys() else None
    force_xs        = process['force_xs']
    maxevents_scale = process['maxevents_scale']
    isMC            = process['isMC']
    
    # --------------------

    print(datasets)
    print(root_path)

    rootfile = io.glob_expand_files(datasets=datasets, datapath=root_path)
    
    # Custom scale event statistics
    if param['maxevents'] is None:
        maxevents = None
    else:
        maxevents = np.max([1, int(param['maxevents'] * maxevents_scale)])
    
    # Load file
    X_uncut, ids = iceroot.load_tree(rootfile=rootfile, tree=param['tree'],
                    entry_start=param['entry_start'], entry_stop=param['entry_stop'],
                    maxevents=maxevents, ids=param['load_ids'], library='ak', dtype=dtype,
                    num_cpus=num_cpus, verbose=verbose)
    
    N_before = len(X_uncut)

    ## ** Here one could save X before any cuts **
    
    # Apply selections
    X,ids,stats = process_func(X=X_uncut, ids=ids, isMC=isMC, class_id=class_id, **param)
    N_after = len(X)
    
    eff_acc = N_after / N_before
    
    print(f'efficiency x acceptance = {eff_acc:0.6f}')

    Y = class_id * ak.Array(np.ones(N_after, dtype=np.int32))

    # -------------------------------------------------
    # Visible cross-section weights
    # (keep float64 to avoid possible precision problems -- conversion to float32 done later)
    if not force_xs:
        # Sum over W yields --> (eff_acc * xs)
        W = (ak.Array(np.ones(N_after, dtype=np.float64)) / N_after) * (xs * eff_acc)
    
    # 'force_xs' mode is useful e.g. in training with a mix of signals with different eff x acc, but one wants
    # to have equal proportions for e.g. theory conditional MVA training.
    # (then one uses the same input 'xs' value for each process in the steering .yaml file)
    else:
        # Sum over W yields --> xs
        W = (ak.Array(np.ones(N_after, dtype=np.float64)) / N_after) * xs
    
    # Save statistics information
    info = {'yaml':      process,
            'cut_stats': stats,
            'eff_acc':   eff_acc,
            'W_sum':     ak.sum(W),
            'W2_sum':    ak.sum(W**2)}
    
    # -------------------------------------------------
    # Add conditional (theory param) variables
    
    if model_param is not None:
        
        print(f'Adding conditional theory (model) parameters')
        print(model_param)
        
        for var in model_param.keys():
            
            value = float(model_param[var]) if model_param[var] is not None else np.nan
            
            # Create a new 'record' (column) to ak-array [input for MVA]
            col_name    = f'MODEL_{var}'
            X[col_name] = value # Broadcasting happens to all events
            
            # A "mirror" copy
            col_name    = f'GEN_{var}'
            X[col_name] = value
    
    # This as a last
    ids = ak.fields(X)
    
    return {'X': X, 'Y': Y, 'W': W, 'ids': ids, 'info': info}


def read_multiple(process_func, processes, root_path, param, class_id, dtype=None, num_cpus=0, verbose=False):
    """
    Loop over different MC / data processes as defined in the yaml files
    
    Args:
        process_func:  data processing function
        processes:     MC processes dictionary (from yaml)
        root_path:     main path of files
        param:         parameters of 'process_func'
        class_id:      class identifier (integer), e.g. 0, 1, 2 ...
        num_cpus:      number of CPUs used (set 0 for automatic)
        verbose:       verbose output print
    
    Returns:
        X, Y, W, ids, info (awkward array format)
    """

    # Combine results
    X_list, Y_list, W_list, ids, info = [], [], [], None, {}

    for i, key in enumerate(processes):
        data = read_single(
            process_func=process_func, process=processes[key],
            root_path=root_path, param=param, class_id=class_id,
            dtype=dtype, num_cpus=num_cpus, verbose=verbose
        )

        # Append data to the lists
        X_list.append(data['X'])
        Y_list.append(data['Y'])
        W_list.append(data['W'])

        # Assuming ids are the same for all processes
        if i == 0:
            ids = copy.deepcopy(data['ids'])
        
        info[key] = copy.deepcopy(data['info'])

        io.showmem()
        gc.collect() #!

    # Concatenate all lists into final arrays
    tic = time.time()
    
    X = aux.concatenate_and_clean(X_list, axis=0)
    Y = aux.concatenate_and_clean(Y_list, axis=0)
    W = aux.concatenate_and_clean(W_list, axis=0)
    
    toc = time.time() - tic
    print(f'Final concatenation took: {toc:0.2f} sec')
    
    io.showmem()
    
    return X,Y,W,ids,info


def load_tree_stats(rootfile, tree, key=None, verbose=False):
    """
    Load the number of events in a list of rootfiles

    Args:
        rootfile: a list of rootfiles
        tree:     tree name to open
        key:      key (variable name) to use to get the number of events, if None then use the first one
        verbose:  verbose output print
    
    Returns:
        number of events
    """

    if type(rootfile) is not list:
        rootfile = [rootfile]

    num_events = np.zeros(len(rootfile), dtype=int)
    
    for i in range(len(rootfile)):
        with uproot.open(rootfile[i]) as file:

            events   = file[tree]
            key_name = events.keys()[0] if key is None else key    

            num_events[i] = len(events.arrays(key_name))
            
            if verbose:
                print(f'{rootfile[i]}')
                print(f'keys(): {events.keys()}')
                print(f'values(): {events.values()}')
            
            file.close()

    return num_events


def events_to_jagged_numpy(events, ids, entry_start=0,
                           entry_stop=None, maxevents=None, label=None):
    """
    Process uproot tree to a jagged numpy (object) array
    
    Args:
        events:      uproot tree
        ids:         names of the variables to pick
        entry_start: first event to consider
        entry_stop:  last event to consider
    
    Returns:
        X
    """
    if label is not None:
        print(f'Loading: {label}', 'yellow')

    # -------------------------------
    if entry_start is None:
        entry_start = 0

    N_all = len(events.arrays(ids[0]))
    N     = len(events.arrays(ids[0], entry_start=entry_start, entry_stop=entry_stop))
    
    if (maxevents is not None) and N > maxevents:
        entry_stop_final = entry_start + maxevents
    else:
        entry_stop_final = entry_stop if entry_stop is not None else N
    # -------------------------------
    
    X = np.empty((entry_stop_final - entry_start, len(ids)), dtype=object) 

    for j in tqdm(range(len(ids))):
        X[:,j] = np.asarray(events.arrays(ids[j],
                    entry_start=entry_start, entry_stop=entry_stop_final, library="np", how=list))
    
    print(f'Entry_start = {entry_start}, entry_stop = {entry_stop}, maxevents = {maxevents} | realized = {len(X)} ({100*len(X)/N_all:0.3f} % | available = {N_all})', 'green')
    
    return X, ids


def load_tree(rootfile, tree, entry_start=0, entry_stop=None, maxevents=None,
              ids=None, library='np', dtype=None, num_cpus=0, verbose=False):
    """
    Load ROOT files

    Args:
        rootfile:      Name of root file paths (string or a list of strings)
        tree:          Tree to read out
        entry_start:   First event to read per file
        entry_stop:    Last event to read per file
        maxevents:     Maximum number of events in total (over all files)
        ids:           Names of the variables to read out from the root tree
        library:       Return type 'np' (jagged numpy) or 'ak' (awkward) of the array
        num_cpus:      Number of processes used (set 0 for automatic)
        verbose:       Verbose output
    
    Returns:
        array of type 'library'
    """
    
    if type(rootfile) is not list:
        rootfile = [rootfile]

    print(f'Opening rootfile {rootfile} with a tree key <{tree}>', 'yellow')

    # Add Tree handles here using the string-syntax of uproot {file:tree}
    files = [rootfile[i] + f':{tree}' for i in range(len(rootfile))]
    
    # ----------------------------------------------------------
    ### Select variables
    with uproot.open(files[0]) as events:
        all_ids = events.keys()
    
    load_ids = aux.process_regexp_ids(ids=ids, all_ids=all_ids)

    if verbose:
        print(f'All variables ({len(all_ids)}): \n{all_ids} \n', 'green')
        print('')
        print(f'Loading variables ({len(load_ids)}): \n{load_ids} \n', 'green')
        print('')
    else:
        print(f'All variables ({len(all_ids)}) | Loading variables ({len(load_ids)})', 'green')
        print('')
    
    print(f'Reading {len(files)} root files | Conversion to object library "{library}" ')
    print('')
    
    if int(num_cpus) == 0:
        num_workers = min(len(files), multiprocessing.cpu_count() // 2) # min handles the case #files < #cpu
    else:
        num_workers = int(num_cpus)
    
    if   library == 'np':
        
        if maxevents is None:
            maxevents = int(1e10)
            print(f'maxevents is None, setting maxevents = {maxevents}', 'red')
        
        # Non-multiprocessed version for single files
        if len(files) == 1:
            
            with uproot.open(files[0]) as events:
                
                param = {'events': events, 'ids': load_ids,
                          'entry_start': entry_start, 'entry_stop': entry_stop, 'maxevents': maxevents, 'label': files[0]}
                X, _ = events_to_jagged_numpy(**param)

                # Clear
                gc.collect()
                io.showmem()

                if (maxevents is not None) and (len(X) >= maxevents):
                    X = X[0:maxevents]
                    print(f'Maximum event count {maxevents} reached', 'red')
            
            print(f'Total number of entries = {len(X)}')        
            
            return X, load_ids

        else:
            
            # ======================================================
            # Multiprocessing version for multiple files
            
            ray.init(num_cpus=num_workers, _temp_dir=f'{os.getcwd()}/tmp/')

            chunk_ind    = aux.split_start_end(range(len(files)), num_workers)
            submaxevents = aux.split_size(range(maxevents), num_workers)
            futures      = []
            
            print(f'submaxevents per ray process: {submaxevents}')

            for k in range(num_workers):    
                futures.append(read_file_np.remote(files[chunk_ind[k][0]:chunk_ind[k][-1]],
                                                   load_ids, entry_start, entry_stop, submaxevents[k], dtype))

            print(f'Get futures')
            results = ray.get(futures) # synchronous read-out
            ray.shutdown()
            
            # Combine future returned sub-arrays
            print(f'Concatenating results from futures')
            
            X = aux.concatenate_and_clean(results, axis=0)
            io.showmem()
            
            # Concatenate one-by-one
            #for k in tqdm(range(len(results))):
            #    X = copy.deepcopy(results[k]) if (k == 0) else np.concatenate((X, results[k]), axis=0)
            #
            #    results[k] = None # free memory
            #    gc.collect()
            #    io.showmem()
        
        print(f'Total number of entries = {len(X)}')        
        
        return X, load_ids

    elif library == 'ak':

        if maxevents is None:
            maxevents = int(1e10)
            print(f'maxevents is None, setting maxevents = {maxevents}', 'red')
        
        # Non-multiprocessed version for single files
        
        if len(files) == 1:

            # Get the number of events
            num_events = get_num_events(rootfile=files[0])
            
            with uproot.open(files[0]) as events:
                
                X = events.arrays(load_ids, entry_start=entry_start,
                                  entry_stop=entry_stop, library='ak', how='zip')

                if dtype is not None:
                    X = X.values_astype(X, dtype)
                
                if (maxevents is not None) and (len(X) > maxevents):
                    X = X[0:maxevents]
                    print(f'Maximum event count {maxevents} reached (had {num_events})', 'red')
        
        else:
            
            # ======================================================
            # Multiprocessing version for multiple files
            
            ray.init(num_cpus=num_workers, _temp_dir=f'{os.getcwd()}/tmp/')

            chunk_ind    = aux.split_start_end(range(len(files)), num_workers)
            submaxevents = aux.split_size(range(maxevents), num_workers)
            futures      = []
            
            print(f'submaxevents per ray process: {submaxevents}')

            for k in range(num_workers):    
                futures.append(read_file_ak.remote(files[chunk_ind[k][0]:chunk_ind[k][-1]],
                                                   load_ids, entry_start, entry_stop, submaxevents[k], dtype))

            print(f'Get futures')
            results = ray.get(futures) # synchronous read-out
            ray.shutdown()
            
            # Combine future returned sub-arrays
            print(f'Concatenating results from futures')

            X = aux.concatenate_and_clean(results, axis=0)
            io.showmem()
            
            # Concatenate one-by-one
            #for k in tqdm(range(len(results))):
            #    X = copy.deepcopy(results[k]) if (k == 0) else ak.concatenate((X, results[k]), axis=0)
            #
            #    results[k] = None # free memory
            #    gc.collect()
            #    io.showmem()
        
        print(f'Total number of entries = {len(X)}')        
            
        return X, ak.fields(X)
        
    else:
        raise Exception(__name__ + f'.load_tree: Unknown library support')

@ray.remote
def read_file_np(files, ids, entry_start, entry_stop, maxevents, dtype=None):
    """
    Ray multiprocess read-wrapper-function per root file.
    
    Remark:
        Multiprocessing within one file did not yield good
        performance empirically, but performance scales linearly for separate files.
    
    Args:
        files:        list of root filenames
        ids:          variables to read out
        entry_start:  first entry to to read
        entry_stop:   last entry to read
        maxevents:    maximum number of events (in total over all files)
        dtype:        Data type
    
    Returns:
        awkward array
    """
    for i in range(len(files)):

        # Get the number of entries
        #num_entries = get_num_events(rootfile=files[i])
        #print(__name__ + f'.read_file_ak: Found {num_entries} entries from the file: {files[i]}')
        
        with uproot.open(files[i]) as events:
            
            param = {'events': events, 'ids': ids,
                    'entry_start': entry_start, 'entry_stop': entry_stop, 'maxevents': maxevents, 'label': files[0]}
            data, loaded_ids = events_to_jagged_numpy(**param)
            
            # Concatenate with other file results
            X = copy.deepcopy(data) if (i == 0) else np.concatenate((X, data), axis=0)
            del data
            gc.collect()
            io.showmem()
            
            if (maxevents is not None) and (len(X) >= maxevents):
                X = X[0:maxevents]
                print(f'Maximum event count {maxevents} reached', 'red')
                break
    
    return X

@ray.remote
def read_file_ak(files, ids, entry_start, entry_stop, maxevents, dtype=None):
    """
    Ray multiprocess read-wrapper-function per root file.
    
    Remark:
        Multiprocessing within one file did not yield good
        performance empirically, but performance scales linearly for separate files.
    
    Args:
        files:        list of root filenames
        ids:          variables to read out
        entry_start:  first entry to to read
        entry_stop:   last entry to read
        maxevents:    maximum number of events (in total over all files)
        dtype:        Data type
    
    Returns:
        awkward array
    """
    for i in range(len(files)):

        # Get the number of entries
        #num_entries = get_num_events(rootfile=files[i])
        #print(__name__ + f'.read_file_ak: Found {num_entries} entries from the file: {files[i]}')
        
        with uproot.open(files[i]) as events:
            
            data = events.arrays(ids, entry_start=entry_start, entry_stop=entry_stop, library='ak', how='zip')
            
            if dtype is not None:
                data = data.values_astype(data, dtype)
            
            # Concatenate with other file results
            X = copy.deepcopy(data) if (i == 0) else ak.concatenate((X, data), axis=0)
            del data
            gc.collect()
            io.showmem()
            
            if (maxevents is not None) and (len(X) >= maxevents):
                X = X[0:maxevents]
                print(__name__ + f'Maximum event count {maxevents} reached', 'red')
                break
    return X


def get_num_events(rootfile, key_index=0):
    """
    Get the number of entries in a rootfile by reading a key
    
    Args:
        rootfile:   rootfile string (with possible Tree name appended with :)
        key_index:  which variable use as a dummy
    
    Returns:
        number of entries
    """
    with uproot.open(rootfile) as events:
        return len(events.arrays(events.keys()[key_index]))

@ray.remote
def events_to_jagged_ak(events, ids, entry_start=None, entry_stop=None, library='ak', how='zip'):
    """ Wrapper for Ray """
    return events.arrays(ids, entry_start=entry_start, entry_stop=entry_stop, library=library, how=how)

