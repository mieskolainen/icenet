# Input data containers
# 
# Mikael Mieskolainen, 2020
# m.mieskolainen@imperial.ac.uk


import numpy as np

from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer


# Main class for datasets
#
class DATASET:

    def __init__(self, func_loader, files, frac, rngseed, class_id = []):

        if (class_id == []) :
            class_id = [0,1] # By default two classes [0,1]
        
        self.trn = Data()
        self.val = Data()
        self.tst = Data()
        
        for f in files :
            X, Y, self.VARS = func_loader(f, class_id)
            trn, val, tst   = split_data(X, Y, frac, rngseed, class_id)

            self.trn += trn
            self.val += val
            self.tst += tst

        self.n_dims  = self.trn.x.shape[1]

        print(__name__ + '.__init__: n_dims = %d' % self.n_dims)


# x is data                [# vectors x # dimensions]
# y is target output data  [# vectors]
#
class Data:

    # constructor
    def __init__(self, x = np.array([]), y = np.array([])):
        self.N = x.shape[0]
        self.x = x
        self.y = y.flatten()

    # + operator
    def __add__(self, other):

        if (len(self.x) == 0): # still empty
            return other

        x = np.concatenate((self.x, other.x), axis=0)
        y = np.concatenate((self.y, other.y.flatten()), axis=0)

        return Data(x, y)

    # += operator
    def __iadd__(self, other):

        if (len(self.x) == 0): # still empty
            return other

        self.x = np.concatenate((self.x, other.x), axis=0)
        self.y = np.concatenate((self.y, other.y.flatten()), axis=0)
        self.N = len(self.y)
        return self

    # filter operator
    def classfilter(self, classid):

        x = self.x[self.y == classid, :]
        y = self.y[self.y == classid]
        
        return Data(x, y)


### Split into [A = train & validation] + [B = test] sets
#
#
def split_data(X, Y, frac, rngseed, class_id = []):

    ### Permute events to have random mixing between classes (a must!)
    np.random.seed(int(rngseed)) # seed it!
    randi = np.random.permutation(X.shape[0])
    X = X[randi, :]
    Y = Y[randi]
    
    N     = X.shape[0]
    N_A   = round(N * frac)
    N_B   = N - N_A

    N_trn = round(N_A * frac)
    N_val = N_A - N_trn
    N_tst = N_B

    # A. Train
    X_trn = X[0:N_trn, :]
    Y_trn = Y[0:N_trn]

    # A. Validation
    X_val = X[N_trn:N_trn + N_val, :]
    Y_val = Y[N_trn:N_trn + N_val]

    # B. Test
    X_tst = X[N-N_tst:N, :]
    Y_tst = Y[N-N_tst:N]

    trn = Data()
    val = Data()
    tst = Data()

    # No spesific class selected
    if (class_id == []) : 

        trn = Data(x = X_trn, y = Y_trn)
        val = Data(x = X_val, y = Y_val)
        tst = Data(x = X_tst, y = Y_tst)
    
    # Loop over all classes selected
    else:

        for c in class_id : 
            ind  = (Y_trn == c)
            trn += Data(x = X_trn[ind, :], y = np.ones(ind.sum())*c)

            ind  = (Y_val == c)
            val += Data(x = X_val[ind, :], y = np.ones(ind.sum())*c)

            ind  = (Y_tst == c)
            tst += Data(x = X_tst[ind, :], y = np.ones(ind.sum())*c)

    ### Permute events once again to have random mixing between classes
    def mix(data):
        randi  = np.random.permutation(data.x.shape[0])
        data.x = data.x[randi,:]
        data.y = data.y[randi]
        return data
    
    trn = mix(trn)
    val = mix(val)
    tst = mix(tst)
    
    print(__name__ + ".split_data: fractions [train: {:0.3f}, validate: {:0.3f}, test: {:0.3f}]".
        format(X_trn.shape[0]/ N, X_val.shape[0]/ N, X_tst.shape[0] / N))

    return trn, val, tst


### Data inputation (treatment of missing values) 
#
#
def inpute_data(X, values = [-1], labels = []):

    if labels == []:
        labels = np.zeros(X.shape[1])

    N = X.shape[0]

    # Loop over dimensions
    for j in range(X.shape[1]):

        # Set NaN for special values
        M_tot = 0
        for z in values:
            
            ind = np.isclose(X[:,j],z)
            X[ind, j] = np.nan

            M = np.sum(ind)
            M_tot += M

            if (M/N > 0):
                print(__name__ + ': Column {} fraction [{:0.3E}] with value {} [{}]'.format(j, M/N, z, labels[j]))

        if (M_tot == N): # Protection, if all are now NaN
            # Set to zero so Imputer Function below does not remove the full column!!
            X[:,j] = 0.0

    # Treat infinities
    for j in range(X.shape[1]):

        inf_ind = np.isinf(X[:,j])
        X[inf_ind, j] = np.nan
        if np.sum(inf_ind) > 0:
            print(__name__ + ': Column {} Number of {} Inf found [{}]'.format(j, np.sum(inf_ind), labels[j]))
    
    # Fill missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    X = imputer.fit_transform(X)
    
    print('[done]')

    return X


# Calculate 0-mean & unit-variance normalization
#
# Input with [#vector x #dimensions]
#
def calc_zscore(X : np.array):

    X_mu  = np.zeros((X.shape[1]))
    X_std = np.zeros((X.shape[1]))

    # Calculate mean and std based on the training data
    for i in range(X.shape[1]) :
        X_mu[i]  = np.mean(X[:,i])
        X_std[i] = np.std(X[:,i])

        if (np.isnan(X_std[i]) | np.isinf(X_std[i])):
            sys.exit(__name__ + ': Fatal error with std[index = {}] is Inf or NaN'.format(i))

    return X_mu, X_std


def apply_zscore(X : np.array, X_mu, X_std):

    Y = X[:] # Make copy with [:]
    
    for i in range(len(X_mu)) :
        Y[:,i] = (X[:,i] - X_mu[i]) / X_std[i]

    return Y


# Choose the active set of input variables
#
def pick_vars(data : DATASET, set_of_variables):

    newind  = np.where(np.isin(data.VARS, set_of_variables))
    newind  = np.array(newind).flatten()
    newvars = []
    for i in newind :
        newvars.append( data.VARS[i] )

    return newind, newvars
