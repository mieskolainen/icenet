# "Pandas style" frame wrapper for columnar data with advanced indexing
#
# m.mieskolainen@imperial.ac.uk, 2024

import numpy as np
from icenet.tools import stx


class icemap:
    """
    Args:
        x   : data                [N vectors x ... x D dimensions]
        ids : variable names      [D strings]

        OR

        x   : data dictionary
    """
    
    # constructor
    def __init__(self, x, ids = None):

        # Input is a dictionary which contains arrays
        if ids is None:
            self.ids = list(x.keys())
            self.x   = np.empty((len(x[self.ids[0]]), len(self.ids)), dtype=object)
            j = 0
            for key in self.ids:
                self.x[:,j] = np.asarray(x[key])
                j+=1

        # Input is arrays and keys separately
        else:
            self.x   = np.array(x)
            self.ids = ids

    # + operator
    def __add__(self, other):

        if (len(self.x) == 0):    # still empty
            return other

        x = np.concatenate((self.x, other.x), axis=0)
        return icemap(x, self.ids)

    # += operator
    def __iadd__(self, other):

        if (len(self.x) == 0):    # still empty
            return other

        self.x = np.concatenate((self.x, other.x), axis=0)
        return self
    
    def __getitem__(self, key):
        """ Advanced indexing """

        if key in self.ids:        # direct access
            """
            Return numpy object
            """
            return self.x[..., self.ids.index(key)]

        elif isinstance(key, str): # might be a cut string, try that
            """
            Return icemap object
            """            
            ind = stx.eval_boolean_syntax(expr=key, X=self.x, ids=self.ids)
            return icemap(self.x[ind, ...], self.ids)

        else:                      # [:,] numpy type indexing
            """
            Return numpy object
            """
            return self.x[key]


def test_icecube_concat():
    """ Unit tests
    """

    X1   = np.array([[1,2,3], [4,5,6]])
    X2   = np.array([[7,8,9], [10,11,12]])
    ids  = ['var0', 'var1', 'var2']

    new1 = icemap(x=X1, ids=ids)
    new2 = icemap(x=X2, ids=ids)

    new3 = new1 + new2

    assert(np.all(new3.x == np.concatenate((X1,X2), axis=0)))


def test_icecube_indexing():
    """ Unit tests
    """
    
    X = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15]])
    print(X)

    ids = ['pt', 'eta', 'phi']
    new = icemap(x = X, ids = ids)

    # Test variable name indexing
    assert np.all(new['eta'] == [2,5,8,11,14])

    # Test direct indexing
    assert np.all(new[:,0] == [1, 4, 7, 10, 13])
    assert np.all(new[4,:] == [13, 14, 15])

    # Test boolean selection indexing
    assert np.all(new['pt > 0'].x == X)
    assert np.all(new['pt > 3 AND eta <= 5'].x == [4,5,6])
    assert np.all(new['pt > 3 && eta <= 5'].x == [4,5,6])



