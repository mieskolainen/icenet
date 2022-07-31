# Functions to evaluate (a boolean) expression tree for vector (component) data
#
# m.mieskolainen@imperial.ac.uk, 2021

import pyparsing as pp
import numpy as np

from icenet.tools import aux

def construct_columnar_cuts(X, ids, cutlist):
    """
    Construct cuts and corresponding names.

    Args:
        X       : Input columnar data matrix
        ids     : Variable names for each column of X
        cutlist : Selection cuts as strings, such as ['ABS__eta < 0.5', 'trgbit == True']
    
    Returns:
        cuts, names
    """
    cuts  = []
    names = []

    for expr in cutlist:

        treelist = parse_boolean_exptree(expr)
        treeobj  = construct_exptree(treelist)
        output   = eval_boolean_exptree(root=treeobj, X=X, ids=ids)

        cuts.append(output)
        names.append(expr)

    return cuts,names


def powerset_cutmask(cut):
    """ Generate powerset 2**|cuts| masks
    
    Args:
        cut : list of pre-calculated cuts, each list element is a boolean array
    Returns:
        (2**|cuts| x num_events) sized boolean mask matrix
    """
    print(cut)

    num_events = len(cut[0])
    num_cuts   = len(cut)
    BMAT       = aux.generatebinary(num_cuts)

    power      = BMAT.shape[0]
    powerset   = np.zeros((power, num_events), dtype=bool)

    # Loop over each boolean
    # [0,0,..0], [0,0,...,1] ... [1,1,..., 1] cut set combination
    for i in range(power):
    
        # Loop over each event
        for evt in range(num_events):
            
            # Loop over each individual cut result
            result = np.array([(cut[k][evt] == BMAT[i,k]) for k in range(num_cuts)], dtype=bool)
            powerset[i, evt] = np.all(result)

    return powerset


def apply_cutflow(cut, names, xcorr_flow=True, EPS=1E-12):
    """ Apply cutflow
    
    Args:
        cut             : list of pre-calculated cuts, each list element is a boolean array
        names           : list of names (description of each cut, for printout only)
        xcorr_flow      : compute full N-point correlations
        return_powerset : return each of 2**|cuts| as a separate boolean mask vector
    
    Returns:
        ind             : list of indices, 1 = pass, 0 = fail
    """
    print(__name__ + '.apply_cutflow: \n')
    
    # Apply cuts in series
    N   = len(cut[0])
    ind = np.ones(N, dtype=np.uint8)
    for i in range(len(cut)):
        ind = np.logical_and(ind, cut[i])

        # Print out "serial flow"
        print(f'cut[{i}][{names[i]:>50}]: pass {np.sum(cut[i]):>10}/{N} = {np.sum(cut[i])/(N+EPS):.4f} | total = {np.sum(ind):>10}/{N} = {np.sum(ind)/(N+EPS):0.4f}')
    
    # Print out "parallel flow"
    if xcorr_flow:
        print_parallel_cutflow(cut=cut, names=names)
    
    return ind


def print_parallel_cutflow(cut, names, EPS=1E-12):
    """
    Print boolean combination cutflow results
    
    Args:
        cut             : list of pre-calculated cuts, each list element is a boolean array
        names           : list of names (description of each cut, for printout only)
    """
    print('\n')
    print(__name__ + '.print_parallel_cutflow: Computing N-point parallel flow <xcorr_flow = True>')
    vec = np.zeros((len(cut[0]), len(cut)))
    for j in range(vec.shape[1]):
        vec[:,j] = np.array(cut[j])

    intmat = aux.binaryvec2int(vec)
    BMAT   = aux.generatebinary(vec.shape[1])
    print(f'Number of boolean combinations for {names}: \n')
    for i in range(BMAT.shape[0]):
        print(f'{BMAT[i,:]} : {np.sum(intmat == i):>10} ({np.sum(intmat == i) / (len(intmat) + EPS):.4f})')
    print('\n')


def parse_boolean_exptree(instring):
    """
    A boolean expression tree parser.
    
    Args:
        instring : input string, e.g. "pt > 7.0 AND (x < 2 OR x >= 4)"
    
    Returns:
        A syntax tree as a list of lists
    
    Information:
        See: https://stackoverflow.com/questions/11133339/
             parsing-a-complex-logical-expression-in-pyparsing-in-a-binary-tree-fashion
    """

    # Functions use internally AND and OR
    instring = instring.replace("&&", "AND")
    instring = instring.replace("||", "OR")    

    def makeLRlike(numterms):
        """
        parse action -maker

        See: https://stackoverflow.com/questions/4571441/
             recursive-expressions-with-pyparsing/4589920#4589920
        """
        
        if numterms is None:
            # None operator can only by binary op
            initlen = 2
            incr = 1
        else:
            initlen = {0:1,1:2,2:3,3:5}[numterms]
            incr = {0:1,1:1,2:2,3:4}[numterms]

        # Define parse action for this number of terms,
        # to convert flat list of tokens into nested list
        def pa(s,l,t):
            t = t[0]
            if len(t) > initlen:
                ret = pp.ParseResults(t[:initlen])
                i = initlen
                while i < len(t):
                    ret = pp.ParseResults([ret] + t[i:i+incr])
                    i += incr
                return pp.ParseResults([ret])
        return pa

    # -------------------------

    operator        = pp.Regex(">=|<=|!=|>|<|==").setName("operator")
    number          = pp.Regex(r"[+-]?\d+(:?\.\d*)?(:?[eE][+-]?\d+)?")
    identifier      = pp.Word(pp.alphas, pp.alphanums + "_")
    comparison_term = identifier | number
    condition       = pp.Group(comparison_term + operator + comparison_term)

    # OR before AND precedence convention
    # makeLRLike guarantees recursive binary tree structure
    expr            = pp.operatorPrecedence(condition,[
                                ("OR",  2, pp.opAssoc.LEFT, makeLRlike(2)),
                                ("AND", 2, pp.opAssoc.LEFT, makeLRlike(2)),
                                ])

    output = expr.parseString(instring)

    # Remove excess []
    return output[0] if len(output) == 1 else output

class tree_node:
    """
    Class to represent the nodes of an expression tree.
    """
    def __init__(self, value):
        self.left  = None
        self.data  = value
        self.right = None

    def __str__(self):
        return f'{{left: {self.left}, data: {self.data}, right: {self.right}}}'

def construct_exptree(root):
    """
    Construct an expression (syntax) tree via recursion.
    
    Args:
        root : List of lists, for example
               [['10', '>', '7'], 'AND', [['4', '>=', '2'], 'AND', ['2', '<=', '4']]]
    Returns:
        an expression tree object with 'tree_node' objects
    """

    # Empty tree
    if root is None:
        return 0

    # Leaf node
    if isinstance(root, str):
        return tree_node(root)

    prev_root       = tree_node(root[1])

    # Evaluate left and right trees
    prev_root.left  = construct_exptree(root[0])
    prev_root.right = construct_exptree(root[2])

    return prev_root

def print_exptree(root):
    """
    Print out an expression tree object via recursion.
    
    Args:
        root: expression tree (object type)
    """
    # Empty tree
    if root is None:
        return 0

    # Leaf node
    if root.left is None and root.right is None:
        return print(f'-- leaf: {root.data}')

    # Print left and right trees
    left_eval  = print_exptree(root.left)
    right_eval = print_exptree(root.right)

    print(f'inter | L: {root.left}, D: {root.data}, R: {root.right}')


def eval_boolean_exptree(root, X, ids):
    """
    Evaluation of a (boolean) expression tree via recursion.
    
    Args:
        root : expression tree object
        X    : data matrix (N events x D dimensions)
        ids : variable names for each D dimension
    
    Returns:
        boolean selection list of size N
    """

    def is_in(a, b):
        """ Helper function """
        if isinstance(a, str):
            
            # Split '__' because of function operators (see below)
            return a.split('__')[-1] in b
        else:
            return False

    # Empty tree
    if root is None:
        return 0

    # Leaf node
    if root.left is None and root.right is None:

        if is_in(root.data, ids):
            return root.data
        else:
            if   isinstance(root.data, str) and root.data in ['True', 'true']:
                return True
            elif isinstance(root.data, str) and root.data in ['False', 'false']:
                return False
            else:
                try:
                    out = float(root.data)
                    return out
                except:
                    raise Exception(f'{root.data} variable is not a leaf or found in {ids}')
    
    # Evaluate left and right trees
    left_sum  = eval_boolean_exptree(root=root.left,  X=X, ids=ids)
    right_sum = eval_boolean_exptree(root=root.right, X=X, ids=ids)

    # Apply the binary operator
    if root.data == 'AND':
        return np.logical_and(left_sum, right_sum)

    if root.data == 'OR':
        return np.logical_or(left_sum,  right_sum)

    # We have ended up with numerical comparison
    operator = root.data

    if isinstance(left_sum, str) and is_in(left_sum, ids):

        lhs = left_sum
        rhs = right_sum

    # Flip the direction
    else:
        lhs = right_sum
        rhs = left_sum

        flips    = {'<=':'>=', '>=':'<=', '>':'<', '<':'>', '==':'==', '!=':'!='}
        operator = flips[operator]

    # Vector index
    split = lhs.split('__')
    ind   = ids.index(split[-1])
    
    # -------------------------------------------------
    # Construct possible function operator
    f = lambda x : x

    if len(split) == 2: # We have 'OPERATOR__x' type input
        
        func_name = split[0]

        if   func_name == 'ABS':
            f = lambda x : np.abs(x)
        elif func_name == 'POW2':
            f = lambda x : x**2
        elif func_name == 'SQRT':
            f = lambda x : np.sqrt(x)
        elif func_name == 'INV':
            f = lambda x : 1.0/x
        elif func_name == 'BOOL':
            f = lambda x : x.astype(bool)
        else:
            raise Exception(__name__ + f'.eval_boolean_exptree: Unknown function {func_name}')
        
        print(__name__ + f'.eval_boolean_exptree: Operator f={func_name}() chosen for "{ids[ind]}"')
    # -------------------------------------------------

    # Middle binary operators g(x,y)
    if   operator == '<':
        g = lambda x,y : x  < float(y)
    elif operator == '>':
        g = lambda x,y : x  > float(y)
    elif operator == '<=':
        g = lambda x,y : x <= float(y)
    elif operator == '>=':
        g = lambda x,y : x >= float(y)
    elif operator == '!=':
        g = lambda x,y : x != int(y)
    elif operator == '==':
        g = lambda x,y : x == int(y)
    else:
        raise Exception(__name__ + f'.eval_boolean_exptree: Unknown binary operator "{operator}"')

    # Evaluate
    return g(f(X[:, ind]), rhs)

def eval_boolean_syntax(expr, X, ids):
    """
    A complete wrapper to evaluate boolean syntax.

    Args:
        expr : boolean expression string, e.g. "pt > 7.0 AND (x < 2 OR x >= 4)"
        X    : input data (N x dimensions)
        ids  : variable names as a list
    
    Returns:
        boolean list of size N
    """

    treelist = parse_boolean_exptree(expr)
    print(treelist)

    treeobj  = construct_exptree(treelist)
    output   = eval_boolean_exptree(root=treeobj, X=X, ids=ids)

    return output


def test_syntax_tree_parsing():
    """
    Unit tests
    """
    expr_A     = 'x < 0.2 AND y < 2 AND z >= 4'
    expr_B     = 'x < 0.2 AND (y < 2 && z >= 4)'
    expr_C     = 'x < 0.2 && y < 2 || z >= 4'
    
    treelist_A = parse_boolean_exptree(expr_A)
    treelist_B = parse_boolean_exptree(expr_B)
    treelist_C = parse_boolean_exptree(expr_C)

    assert(f'{treelist_A}' == "[[['x', '<', '0.2'], 'AND', ['y', '<', '2']], 'AND', ['z', '>=', '4']]")
    assert(f'{treelist_B}' == "[['x', '<', '0.2'], 'AND', [['y', '<', '2'], 'AND', ['z', '>=', '4']]]")
    assert(f'{treelist_C}' == "[['x', '<', '0.2'], 'AND', [['y', '<', '2'], 'OR', ['z', '>=', '4']]]")


def test_syntax_tree_simple():
    """
    Units tests
    """

    # Create random input and repeat tests
    for i in range(100):

        X        = np.random.rand(20,3) * 50
        ids      = ['x', 'y', 'z']

        ### TEST 1
        expr     = 'x >= 0.0 AND POW2__z < 2500'

        treelist = parse_boolean_exptree(expr)
        treeobj  = construct_exptree(treelist)
        output   = eval_boolean_exptree(root=treeobj, X=X, ids=ids)
        assert np.all(output)

        # One-Liner
        output2  = eval_boolean_syntax(expr=expr, X=X, ids=ids)
        assert np.all(output2)

        assert np.all(output == output2)
        
def test_syntax_tree_flip():
    """
    Unit tests
    """

    # Create random input and repeat tests
    for i in range(100):

        X            = np.random.rand(20,3) * 50
        ids          = ['x', 'y', 'z']

        ### TEST
        expr_strings = ['ABS__x > 3.8 OR (x > 7 AND (y >= 2 OR z <= 4))', '((y >= 2 OR z <= 4) AND 7 < x) OR 3.8 < ABS__x']
        output = []
        
        # Test both syntax strings
        for i in range(len(expr_strings)):
            treelist = parse_boolean_exptree(expr_strings[i])

            print(treelist)

            treeobj  = construct_exptree(treelist)
            output.append(eval_boolean_exptree(root=treeobj, X=X, ids=ids))

        assert np.all(output[0] == output[1])
