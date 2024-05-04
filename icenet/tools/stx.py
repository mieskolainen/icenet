# Functions to evaluate (a boolean) expression tree and cuts for vector (component) data
#
# m.mieskolainen@imperial.ac.uk, 2024

import pyparsing as pp
import numpy as np
import copy
import pprint

from icenet.tools import aux
from functools import reduce

def print_stats(mask, text):
    """
    Print filter mask category statistics
    
    Args:
        mask:   computed event filter mask
        text:   filter descriptions
    """    
    N = mask.shape[1]
    print(__name__ + f'.print_stats: mask.shape = {mask.shape}')
    
    for i in range(mask.shape[0]):
        count = np.sum(mask[i,:])
        print(f'cat[{text[i]}]: {count} ({count / N:0.5E})')
    print('')


def filter_constructor(filters, X, ids, y=None):
    """
    Filter product main constructor
    
    Args:
        filters: yaml file input
        X:       columnar input data
        ids:     data column keys (list of strings)
        y:       class labels (default None), used for diplomat (always passing) classes
        
    Returns:
        mask matrix, mask text labels (list), mask path labels (list)
    """
    def process_set(input_set, filter_ID=None):
        
        expand = input_set['expand']
        cutset = input_set['cutset']

        if   expand == 'set':
            mask,text,path = set_constructor(cutset=cutset, X=X, ids=ids, veto=False)
        elif expand == 'vetoset':
            mask,text,path = set_constructor(cutset=cutset, X=X, ids=ids, veto=True)
        elif expand == 'powerset':
            mask,text,path = powerset_constructor(cutset=cutset, X=X, ids=ids)
        else:
            raise Exception(__name__ + f'.filter_constructor: expand type should be "set" or "powerset"')
        
        # Special always passing classes
        if 'diplomat_classes' in input_set.keys():
            for c in input_set['diplomat_classes']:
                print(__name__ + f'.filter_constructor: Passing through diplomat class "{c}"')
                cind = (y == c)
                mask[:, cind] = True
        
        if filter_ID is not None:
            text = [f'F[{filter_ID}]: ' + s for s in text]
            path = [f'F[{filter_ID}]:'  + s for s in path]
        
        return mask,text,path
    
    # Loop over all filter definitions
    ind = 0
    all_mask_matrix    = None
    all_text, all_path = [],[]

    while True:
        try:
            f    = filters[f'filter[{ind}]']
            filter_ID = ind
            ind += 1
        except:
            break
        
        sets      = f['sets']      # a list of set indices
        operator  = f['operator']  # an operator to apply (between sets)

        if type(sets) is not list:
            raise Exception(__name__ + f'.filter_constructor: Input "sets" should be a list')

        # A single set
        if len(sets) == 1:
            index = sets[0]
            mask_matrix, text_set, path_set = process_set(filters[f'set[{index}]'], filter_ID=filter_ID)
        
        # Several sets (2,3,4 ...) combined
        else:
            masks, texts, paths = [],[],[]
            
            running_index = {}    
            for k in range(len(sets)):
                index = sets[k]
                running_index[index] = k

                m,t,p = process_set(filters[f'set[{index}]'])
                masks.append(m), texts.append(t), paths.append(p)

                if   operator == 'cartesian_and':
                    logical_func = np.logical_and.reduce
                    name = '&'
                elif operator == 'cartesian_or':
                    logical_func = np.logical_or.reduce
                    name = '|'
                    
                else:
                    raise Exception(__name__ + f'.filter_constructor: Operator should be "and" or "cartesian_or" multiple sets')
                
            try:
                match = []
                if f['match'] is not None: # Match set indices correctly to running indices
                    for index in f['match']:
                        match.append(running_index[index])
            except:
                match = None
            
            # Loop over all cartesian combinations
            mask_set, text_set, path_set = [],[],[]
            
            dim = [None]*len(sets)
            for i in range(len(sets)):
                dim[i] = np.array(range(masks[i].shape[0]), dtype=int)
            dimgrid = aux.cartesian_product(*dim)
            
            if match is not None:
                # Pick only pair (or N-tuplet)-wise equal combinations
                pick_ind = []
                for i in range(dimgrid.shape[0]):
                    if len(np.unique(dimgrid[i, match])) == 1:
                        pick_ind.append(i)
                dimgrid = dimgrid[pick_ind, :]
            
            print(__name__ + f'.filter_constructor: Filter combinations {dimgrid.shape} combined with {name} and match {match} (running index)')
            print(dimgrid)
            
            # Loop over all combinations
            for i in range(dimgrid.shape[0]):
                
                comb = dimgrid[i,:]
                A,T,P = [None]*len(sets), [None]*len(sets), [None]*len(sets)
                
                for j in range(len(sets)):
                    A[j],T[j],P[j] = masks[j][comb[j]], texts[j][comb[j]], paths[j][comb[j]]
                
                mask_set.append( logical_func(A) )
                text_set.append( reduce(lambda a, b: f'{a} {name} {b}', T) )
                path_set.append( reduce(lambda a, b: f'{a}{name}{b}', P) )
            
            # Add filter ID
            text_set = [f'F[{filter_ID}]: ' + s for s in text_set]
            path_set = [f'F[{filter_ID}]:'  + s for s in path_set]
            
            pprint.pprint(text_set)
            
            # Turn into matrix [num of masks x num of events]
            mask_matrix = np.zeros((len(mask_set), len(mask_set[0])), dtype=np.bool_)
            for i in range(len(mask_set)):
                mask_matrix[i,:] = mask_set[i]
        
        ## ** Add to lists of all filter products **
        if all_mask_matrix is not None:
            all_mask_matrix = np.concatenate((all_mask_matrix, mask_matrix), axis=0)
        else:
            all_mask_matrix = copy.deepcopy(mask_matrix)
        all_text += text_set
        all_path += path_set
    
    return all_mask_matrix, all_text, all_path


def set_constructor(cutset, X, ids, veto=False):
    """
    Direct set filter constructor
    
    Returns:
        mask matrix, mask text labels (list), mask path labels (list)
    """
    cutlist  = [cutset[k]['cut']   for k in range(len(cutset))]
    textlist = [cutset[k]['latex'] for k in range(len(cutset))]

    # Construct cuts and apply
    mask_set, names = construct_columnar_cuts(X=X, ids=ids, cutlist=cutlist)

    # Turn into matrix
    mask_matrix = np.zeros((len(mask_set), len(mask_set[0])), dtype=np.bool_)

    # Create description strings
    text_set, path_set = [],[]

    for i in range(len(mask_set)):

        if veto == False:
            # take passing (index = 1)
            mask_matrix[i,:] = mask_set[i]
            string = textlist[i][1]
        else:
            # take non-passing (veto) (index = 0)
            mask_matrix[i,:] = ~mask_set[i]
            string = textlist[i][0]

        text_set.append(string)
        path_set.append(string.replace(' ','').replace('$','').replace('\\',''))

    return mask_matrix, text_set, path_set


def powerset_constructor(cutset, X, ids):
    """
    Powerset (all subsets of boolean combinations) filter constructor
    
    Returns:
        mask matrix, mask text labels (list), mask path labels (list)
    """
    cutlist  = [cutset[k]['cut']   for k in range(len(cutset))]
    textlist = [cutset[k]['latex'] for k in range(len(cutset))]

    # Construct cuts and apply
    masks, names  = construct_columnar_cuts(X=X, ids=ids, cutlist=cutlist)
    print_parallel_cutflow(masks=masks, names=names)
    
    mask_powerset = powerset_cutmask(cut=masks)
    BMAT          = aux.generatebinary(len(masks))

    #print(textlist)

    # Loop over all powerset 2**|cuts| masked selections
    # Create a description latex strings and savepath strings
    text_powerset, path_powerset = [],[]

    for i in range(BMAT.shape[0]):
        string = ''
        for j in range(BMAT.shape[1]):
            bit = BMAT[i,j] # 0 or 1
            string += f'{textlist[j][bit]}'
            if j != BMAT.shape[1] - 1: string += ' '
        string += f' {BMAT[i,:]}'

        text_powerset.append(string)
        path_powerset.append((f'{BMAT[i,:]}').replace(' ','').replace('$','').replace('\\',''))

    return mask_powerset, text_powerset, path_powerset


def construct_columnar_cuts(X, ids, cutlist):
    """
    Construct cuts and corresponding names.

    Args:
        X       : Input columnar data matrix
        ids     : Variable names for each column of X
        cutlist : Selection cuts as strings, such as ['ABS@eta < 0.5', 'trgbit == True']
    
    Returns:
        masks (boolean arrays) in a list, boolean expressions (list)
    """
    masks  = []
    names  = []

    for expr in cutlist:

        treelist = parse_boolean_exptree(expr)
        treeobj  = construct_exptree(treelist)
        mask     = eval_boolean_exptree(root=treeobj, X=X, ids=ids)

        masks.append(mask)
        names.append(expr)
    
    return masks, names


def powerset_cutmask(cut):
    """ Generate powerset 2**|cuts| masks
    
    Args:
        cut : list of pre-calculated cuts, each list element is a boolean array
    Returns:
        mask: (2**|cuts| x num_events) sized boolean mask matrix
    """
    #print(cut)

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
        mask            : boolean mask of size number of events (1 = pass, 0 = fail)
    """
    print(__name__ + '.apply_cutflow: \n')
    
    N    = len(cut[0])
    mask = np.ones(N, dtype=bool)

    # Apply cuts in series
    for i in range(len(cut)):
        mask = np.logical_and(mask, cut[i])

        # Print out "serial flow"
        print(f'cut[{i}][{names[i]:>50}]: pass {np.sum(cut[i]):>10}/{N} = {np.sum(cut[i])/(N+EPS):.4f} | total = {np.sum(mask):>10}/{N} = {np.sum(mask)/(N+EPS):0.4f}')
    
    # Print out "parallel flow"
    if xcorr_flow:
        print_parallel_cutflow(masks=cut, names=names)
    
    return mask


def print_parallel_cutflow(masks, names, EPS=1E-12):
    """
    Print boolean combination cutflow statistics
    
    Args:
        cut   : list of pre-calculated cuts, each list element is a boolean array with size of num of events
        names : list of names (description of each cut, for printout only)
    """
    print('\n')
    print(__name__ + '.print_parallel_cutflow: Computing N-point parallel flow <xcorr_flow = True>')
    vec = np.zeros((len(masks[0]), len(masks)))
    for j in range(vec.shape[1]):
        vec[:,j] = np.array(masks[j])

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
        instring : input string,
            e.g. "pt > 7.0 AND (x < 2 OR x >= 4)"
    
    Returns:
        A syntax tree as a list of lists
    
    Information:
        See: https://stackoverflow.com/questions/11133339/
             parsing-a-complex-logical-expression-in-pyparsing-in-a-binary-tree-fashion
    """
    
    # Check we don't have single & or single | (need to be &&, ||)
    for c in ["&", "|"]:
        if (instring.count(c) % 2) != 0:
            raise Exception(__name__ + f'.parse_boolean_exptree: Problem with {c}, use only C-style && or ||')
    
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
    
    # Extend sets with '_' and '@' special characters
    identifier      = pp.Word(pp.alphas + "_" + "@", pp.alphanums + "_" + "@")
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
            
            # print(f'a:{a}, b:{b}')
            
            # Split at '@' because of function operators (see below)
            return a.split('@')[-1] in b
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
    
    # Pick the columnar index
    split = lhs.split('@') # Treat the possible operator
    ind   = ids.index(split[-1])
    
    # -------------------------------------------------
    # Construct possible function operator
    f = lambda x : x

    if len(split) == 2: # We have 'OPERATOR@x' type input
        
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
        elif func_name == 'FLOAT':
            f = lambda x : x.astype(float)
        elif func_name == 'INT':
            f = lambda x : x.astype(int)
        else:
            raise Exception(__name__ + f'.eval_boolean_exptree: Unknown function {func_name}')
        
        print(__name__ + f'.eval_boolean_exptree: Operator f={func_name}() chosen for "{ids[ind]}"')
    # -------------------------------------------------
    
    # Middle binary operators g(x,y)
    if   operator == '<':
        g = lambda x,y : x  < y
    elif operator == '>':
        g = lambda x,y : x  > y
    elif operator == '<=':
        g = lambda x,y : x <= y
    elif operator == '>=':
        g = lambda x,y : x >= y
    elif operator == '!=':
        g = lambda x,y : ~np.isclose(x,y, rtol=1e-02, atol=1e-03) # custom tolerance
    elif operator == '==':
        g = lambda x,y :  np.isclose(x,y, rtol=1e-02, atol=1e-03)
    else:
        raise Exception(__name__ + f'.eval_boolean_exptree: Unknown binary operator "{operator}"')
    
    if isinstance(rhs, str):
        rhs = float(rhs)
    
    # Evaluate
    return g(f(X[:, ind]), rhs)

def eval_boolean_syntax(expr, X, ids, verbose=False):
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

    if verbose:
        print(__name__ + f'.eval_boolean_syntax:')
        print(treeobj)
        print(f'Selection fraction: {np.sum(output) / len(output):0.4e}')
    
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
        expr     = 'x >= 0.0 AND POW2@z < 2500'

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
        expr_strings = ['ABS@x > 3.8 OR (x > 7 AND (y >= 2 OR z <= 4))', '((y >= 2 OR z <= 4) AND 7 < x) OR 3.8 < ABS@x']
        output = []
        
        # Test both syntax strings
        for i in range(len(expr_strings)):
            treelist = parse_boolean_exptree(expr_strings[i])

            print(treelist)

            treeobj  = construct_exptree(treelist)
            output.append(eval_boolean_exptree(root=treeobj, X=X, ids=ids))

        assert np.all(output[0] == output[1])


def test_powerset():

    cut = [np.array([True, True, False, False, False]),
           np.array([True, False, True, False, False]),
           np.array([True, False, True, False, True])]

    maskmatrix = powerset_cutmask(cut)

    # 0 000
    # 1 001
    # 2 010
    # 3 011
    # 4 100
    # 5 101
    # 6 110
    # 7 111

    reference = np.array(
        [[False, False, False,  True, False],
         [False, False, False, False,  True],
         [False, False, False, False, False],
         [False, False,  True, False, False],
         [False,  True, False, False, False],
         [False, False, False, False, False],
         [False, False, False, False, False],
         [ True, False, False, False, False]])

    assert np.all(maskmatrix == reference)
