# Common input & data reading routines for the HLT electron trigger studies
# 
# Mikael Mieskolainen, 2021
# m.mieskolainen@imperial.ac.uk


from icenet.tools import io
from icenet.tools import aux
from icenet.tools import plots
from icenet.tools import prints
from icenet.tools import process


def init(MAXEVENTS=None):
    """ Initialize electron HLT trigger data.
	
    Args:
        Implicit commandline and yaml file input.
    
    Returns:
        jagged array data, arguments
    """
    
    args, cli = process.read_config('./configs/trg')
    features  = globals()[args['imputation_param']['var']]
    
    
    # --------------------------------------------------------------------
    ### SET GLOBALS (used only in this file)
    global ARGS
    ARGS = args

    if MAXEVENTS is not None:
        ARGS['MAXEVENTS'] = MAXEVENTS
    
    print(__name__ + f'.init: inputvar   =  {args["inputvar"]}')
    print(__name__ + f'.init: cutfunc    =  {args["cutfunc"]}')
    print(__name__ + f'.init: targetfunc =  {args["targetfunc"]}')
    # --------------------------------------------------------------------

    ### Load data

    # Background (0) and signal (1)
    class_id = [0,1]
    data     = io.DATASET(func_loader=load_root_file_new, files=args['root_files'], class_id=class_id, frac=args['frac'], rngseed=args['rngseed'])
    
    # @@ Imputation @@
    if args['imputation_param']['active']:

        special_values = args['imputation_param']['values'] # possible special values
        print(__name__ + f': Imputing data for special values {special_values} for variables in <{args["imputation_param"]["var"]}>')

        # Choose active dimensions
        dim = np.array([i for i in range(len(data.VARS)) if data.VARS[i] in features], dtype=int)

        # Parameters
        param = {
            "dim":        dim,
            "values":     special_values,
            "labels":     data.VARS,
            "algorithm":  args['imputation_param']['algorithm'],
            "fill_value": args['imputation_param']['fill_value'],
            "knn_k":      args['imputation_param']['knn_k']
        }
        
        # NOTE, UPDATE NEEDED: one should save here 'imputer_trn' to a disk -> can be used with data
        data.trn.x, imputer_trn = io.impute_data(X=data.trn.x, imputer=None,        **param)
        data.tst.x, _           = io.impute_data(X=data.tst.x, imputer=imputer_trn, **param)
        data.val.x, _           = io.impute_data(X=data.val.x, imputer=imputer_trn, **param)
        
    else:
        # No imputation, but fix spurious NaN / Inf
        data.trn.x[np.logical_not(np.isfinite(data.trn.x))] = 0
        data.val.x[np.logical_not(np.isfinite(data.val.x))] = 0
        data.tst.x[np.logical_not(np.isfinite(data.tst.x))] = 0

    cprint(__name__ + f""".common: Process RAM usage: {io.process_memory_use():0.2f} GB 
        [total RAM in use: {psutil.virtual_memory()[2]} %]""", 'red')
    
    return data, args, features


def load_root_file_new(root_path, VARS=None, entrystart=0, entrystop=None, class_id = [], args=None):
    """ Loads the root file.
    
    Args:
        root_path : paths to root files
        class_id  : class ids
    
    Returns:
        X,Y       : input, output matrices
        VARS      : variable names
    """

    # -----------------------------------------------
    # ** GLOBALS **

    if args is None:
        args = ARGS

    CUTFUNC    = globals()[args['cutfunc']]
    TARFUNC    = globals()[args['targetfunc']]
    FILTERFUNC = globals()[args['filterfunc']]

    if entrystop is None:
        entrystop = args['MAXEVENTS']
    # -----------------------------------------------

    def showmem():
        cprint(__name__ + f""".load_root_file: Process RAM usage: {io.process_memory_use():0.2f} GB 
            [total RAM in use {psutil.virtual_memory()[2]} %]""", 'red')
    
    ### From root trees
    print('\n')
    cprint( __name__ + f'.load_root_file: Loading with uproot from file ' + root_path, 'yellow')
    cprint( __name__ + f'.load_root_file: entrystart = {entrystart}, entrystop = {entrystop}')

    file   = uproot.open(root_path)
    events = file["ntuplizer"]["tree"]

    print(events)
    print(events.name)
    print(events.title)
    #cprint(__name__ + f'.load_root_file: events.numentries = {events.numentries}', 'green')

    ### All variables
    if VARS is None:
        VARS = events.keys() #[x for x in events.keys()]
    #VARS_scalar = [x.decode() for x in events.keys() if b'image_' not in x]
    #print(VARS)

    # Check is it MC (based on the first event)
    X_test = events.arrays('is_mc', entry_start=entrystart, entry_stop=entrystop)
    
    isMC   = bool(X_test[0]['is_mc'])
    N      = len(X_test)
    print(__name__ + f'.load_root_file: isMC: {isMC}')
    
    # Now read the data
    print(__name__ + '.load_root_file: Loading root file variables ...')

    # --------------------------------------------------------------
    # Important to lead variables one-by-one (because one single np.assarray call takes too much RAM)

    # Needs to be of object type numpy array to hold arbitrary objects (such as jagged arrays) !
    X = np.empty((N, len(VARS)), dtype=object) 

    for j in tqdm(range(len(VARS))):
        x = events.arrays(VARS[j], library="np", how=list, entry_start=entrystart, entry_stop=entrystop)
        X[:,j] = np.asarray(x)
    # --------------------------------------------------------------
    Y = None


    print(__name__ + f'common: X.shape = {X.shape}')
    showmem()

    prints.printbar()

    # =================================================================
    # *** MC ONLY ***

    if isMC:

        # @@ MC target definition here @@
        cprint(__name__ + f'.load_root_file: Computing MC <targetfunc> ...', 'yellow')
        Y = TARFUNC(events, entrystart=entrystart, entrystop=entrystop, new=True)
        Y = np.asarray(Y).T

        print(__name__ + f'common: Y.shape = {Y.shape}')

        # For info
        labels1 = ['is_e', 'is_egamma']
        aux.count_targets(events=events, names=labels1, entrystart=entrystart, entrystop=entrystop, new=True)

        prints.printbar()

        # @@ MC filtering done here @@
        cprint(__name__ + f'.load_root_file: Computing MC <filterfunc> ...', 'yellow')
        indmc = FILTERFUNC(X=X, VARS=VARS, xcorr_flow=args['xcorr_flow'])

        cprint(__name__ + f'.load_root_file: Prior MC <filterfunc>: {len(X)} events', 'green')
        cprint(__name__ + f'.load_root_file: After MC <filterfunc>: {sum(indmc)} events ', 'green')
        prints.printbar()
        
        
        X = X[indmc]
        Y = Y[indmc].squeeze() # Remove useless dimension
    # =================================================================
    
    # -----------------------------------------------------------------
    # @@ Observable cut selections done here @@
    cprint(colored(__name__ + f'.load_root_file: Computing <cutfunc> ...'), 'yellow')
    cind = CUTFUNC(X=X, VARS=VARS, xcorr_flow=args['xcorr_flow'])
    # -----------------------------------------------------------------
    
    N_before = X.shape[0]

    ### Select events
    X = X[cind]
    if isMC: Y = Y[cind]

    N_after = X.shape[0]
    cprint(__name__ + f".load_root_file: Prior <cutfunc> selections: {N_before} events ", 'green')
    cprint(__name__ + f".load_root_file: Post  <cutfunc> selections: {N_after} events ({N_after / N_before:.3f})", 'green')
    print('')

    showmem()
    prints.printbar()

    # ** REMEMBER TO CLOSE **
    file.close()

    return X, Y, VARS

