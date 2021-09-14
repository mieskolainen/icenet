# Functions to evaluate (a boolean) expression tree for vector (component) data
#
# m.mieskolainen@imperial.ac.uk, 2021

import pyparsing as pp
import numpy as np


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

    operator        = pp.Regex(">=|<=|!=|>|<|==").setName("operator")
    number          = pp.Regex(r"[+-]?\d+(:?\.\d*)?(:?[eE][+-]?\d+)?")
    identifier      = pp.Word(pp.alphas, pp.alphanums + "_")
    comparison_term = identifier | number
    condition       = pp.Group(comparison_term + operator + comparison_term)

    expr            = pp.operatorPrecedence(condition,[
                                ("AND", 2, pp.opAssoc.LEFT, ),
                                ("OR",  2, pp.opAssoc.LEFT, ),
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


def eval_boolean_exptree(root, X, VARS):
    """
    Evaluation of a (boolean) expression tree via recursion.
    
    Args:
        root : expression tree object
        X    : data matrix (N events x D dimensions)
        VARS : variable names for each D dimension
    
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

        if is_in(root.data, VARS):
            return root.data
        else:
            if   is_in(root.data, ['True', 'true']):
                return True
            elif is_in(root.data, ['False', 'false']):
                return False
            else:
                return float(root.data)
    
    # Evaluate left and right trees
    left_sum  = eval_boolean_exptree(root=root.left,  X=X, VARS=VARS)
    right_sum = eval_boolean_exptree(root=root.right, X=X, VARS=VARS)

    # Apply the binary operator
    if root.data == 'AND':
        return np.logical_and(left_sum, right_sum)

    if root.data == 'OR':
        return np.logical_or(left_sum,  right_sum)

    # We have ended up with numerical comparison
    operator = root.data

    if isinstance(left_sum, str) and is_in(left_sum, VARS):

        lhs = left_sum
        rhs = right_sum

    # Flip the direction
    else:
        lhs = right_sum
        rhs = left_sum

        if  operator == '<=':
            operator = '>='
        elif operator == '>=':
             operator = '<='
        elif operator == '>':
            operator = '<'
        elif operator == '<':
            operator = '>'

    # Vector index
    split = lhs.split('__')
    ind   = VARS.index(split[-1])
    
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
        else:
            raise Exception(__name__ + f'.eval_boolean_exptree: Unknown function {func_name}')
    # -------------------------------------------------

    # Middle binary operators g(x,y)
    if   operator == '<':
        g = lambda x,y : x < float(y)
    elif operator == '>':
        g = lambda x,y : x > float(y)
    elif operator == '!=':
        g = lambda x,y : x != int(y)
    elif operator == '==':
        g = lambda x,y : x == int(y)
    elif operator == '<=':
        g = lambda x,y : x <= float(y)
    elif operator == '>=':
        g = lambda x,y : x >= float(y)
    else:
        raise Exception(__name__ + f'.eval_boolean_exptree: Unknown binary operator "{operator}"')

    # Evaluate
    return g(f(X[:, ind]), rhs)


def eval_boolean_syntax(expr, X, VARS):
    """
    A complete wrapper to evaluate boolean syntax.

    Args:
        expr : boolean expression string, e.g. "pt > 7.0 AND (x < 2 OR x >= 4)"
        X    : input data (N x dimensions)
        VARS : variable names as a list

    Returns:
        boolean list of size N
    """

    treelist = parse_boolean_exptree(expr)
    treeobj  = construct_exptree(treelist)
    output   = eval_boolean_exptree(root=treeobj, X=X, VARS=VARS)

    return output


def test_syntax_tree_simple():
    """
    Units tests
    """

    # Create random input and repeat tests
    for i in range(100):

        X        = np.random.rand(20,3) * 50
        VARS     = ['x', 'y', 'z']

        ### TEST 1
        expr     = 'x >= 0.0 AND POW2__z < 2500'

        treelist = parse_boolean_exptree(expr)
        treeobj  = construct_exptree(treelist)
        output   = eval_boolean_exptree(root=treeobj, X=X, VARS=VARS)
        assert np.all(output)

        # One-Liner
        output2  = eval_boolean_syntax(expr=expr, X=X, VARS=VARS)
        assert np.all(output2)

        assert np.all(output == output2)
        

def test_syntax_tree_flip():
    """
    Unit tests
    """

    # Create random input and repeat tests
    for i in range(100):

        X            = np.random.rand(20,3) * 50
        VARS         = ['x', 'y', 'z']

        ### TEST
        expr_strings = ['ABS__x > 3.8 OR (x > 7 AND (y >= 2 OR z <= 4))', '((y >= 2 OR z <= 4) AND 7 < x) OR 3.8 < ABS__x']
        output = []
        
        # Test both syntax strings
        for i in range(len(expr_strings)):
            treelist = parse_boolean_exptree(expr_strings[i])

            print(treelist)

            treeobj  = construct_exptree(treelist)
            output.append(eval_boolean_exptree(root=treeobj, X=X, VARS=VARS))

        assert np.all(output[0] == output[1])
