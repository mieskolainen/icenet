# Binary decay tree recursion to arbitrary depth sandbox code
#
# m.mieskolainen@imperial.ac.uk, 2022

import numpy as np


# A binary tree node
class Node:
  
    # Constructor to create a new node
    def __init__(self, name, mother, amplitude=None):
        
        self.name      = name
        self.mother    = mother
        self.legs      = [None, None]
        self.amplitude = amplitude

def reverseLevelOrder(root):
    """
    Reverse level order traversal
    """
    stack = []

    h = height(root)
    for i in reversed(range(1, h+1)):
        constructGivenLevel(root,i,stack)
    return stack

def constructGivenLevel(root, level, stack):
    """
    Construct nodes at a given level
    """
    if root is None:
        return 
    
    if level == 1 :
        stack.append({'name': root.name, 'level': level, 'mother': root.mother, 'amplitude': root.amplitude})

    elif level > 1:
        constructGivenLevel(root.legs[0], level-1, stack)
        constructGivenLevel(root.legs[1], level-1, stack)


def postOrder(root, stack = []):
    """
    Function to do post order traversal
    """
    h = height(root)

    if root.legs[0] is not None:

        # Traverse left
        postOrder(root.legs[0], stack)

        # Traverse right subtree
        postOrder(root.legs[1], stack)    

    # Collect this branch
    stack.append({'name': root.name, 'level': h, 'mother': root.mother, 'amplitude': root.amplitude})

    return stack


def height(node):
    """
    Compute the height of a tree. The number of 
    nodes along the deepest path from the root node
    down to the most far away leaf.
    """

    if node is None:
        return 0 # -> The first real node gets 1
    else:
        # Compute the height of each subtree

        lheight = height(node.legs[0])
        rheight = height(node.legs[1])
        
        # Use the larger one
        if lheight > rheight :
            return lheight + 1
        else:
            return rheight + 1


def get_amplitude(i,j,k):
    """
    Return amplitude of a triplet (i,j,k), where k is the mother
    """

    # Both legs of k have no daughters)
    if   i['amplitude'] is None and j['amplitude'] is None:
        return k['amplitude'], f"{k['name']}"

    # Leg-i is terminal, Leg-j has daughters
    elif   i['amplitude'] is None:
        return j['amplitude'] @ k['amplitude'], f"({j['name']} {k['name']})"

    # Leg-j is terminal, Leg-i has daughters
    elif j['amplitude'] is None:        
        return i['amplitude'] @ k['amplitude'], f"({i['name']} {k['name']})"

    # Both legs have daughters
    else:
        print(np.kron(i['amplitude'], j['amplitude']))
        print(k['amplitude'])

        return np.kron(i['amplitude'], j['amplitude']) @ k['amplitude'], f"[({i['name']} (x) {j['name']}) {k['name']}]"


def normal_order(stack, i, j):
    # "Normal ordering" (required because tensor product does not commute)
    ind_A, ind_B = i,j

    # If the last 'arrow' is '>'' --> flip
    # (rfind returns the last matching index, -1 if not found at all)
    if stack[i]['name'].rfind('<') < stack[i]['name'].rfind('>'):
        ind_A, ind_B = j,i

    print(f"{stack[ind_A]['name']} | {stack[ind_B]['name']}")

    return ind_A,ind_B


def backward_traverse(stack):
    """
    Backward traverse and collect decay amplitudes from a stack,
    which is post order constructed.
    """
    for i in range(0, len(stack)):
        for j in range(i+1, len(stack)-1):

            if stack[i]['mother'] == stack[j]['mother'] and stack[j]['level'] == 1:
                for k in range(j+1, len(stack)):

                    if k > len(stack)-1: return stack # Protection due to recursion

                    if stack[j]['mother'] == stack[k]['name']:

                        ind_A,ind_B     = normal_order(stack, i, j)
                        amplitude, name = get_amplitude(stack[ind_A], stack[ind_B], stack[k])
                        new  = {'name': name, 'mother': stack[k]['mother'], 'level': 1, 'amplitude': amplitude}
                        
                        # Reverse order pop
                        indices = [k,j,i]
                        for n in sorted(indices, reverse=True): # Reverse order
                            stack.pop(n)

                        stack.insert(0,new) # Insert front
                        
                        print('')
                        for n in range(len(stack)):
                            print(f"backward_traverse: [{n}] name: {stack[n]['name']} | mother: {stack[n]['mother']} | ")

                        stack = backward_traverse(stack)
    return stack


## Test decay tree
np.random.seed(123)

unit = 3

if unit == 0:

    root                 = Node('X', None, np.random.rand(3,3))

elif unit == 1:

    root                 = Node('X', None, np.random.rand(9,9))
    root.legs[0]         = Node('<A1', 'X',  np.random.rand(1,3))
    root.legs[1]         = Node('>B1', 'X',  np.random.rand(1,3))

elif unit == 2:

    root                 = Node('X', None, np.random.rand(9,9))
    root.legs[0]         = Node('<A1', 'X',  np.random.rand(1,3))
    root.legs[1]         = Node('>B1', 'X',  np.random.rand(1,3))

    root.legs[0].legs[0] = Node('<A11', '<A1', None)
    root.legs[0].legs[1] = Node('>A21', '<A1', None)

    root.legs[1].legs[0] = Node('<B11', '>B1', None)
    root.legs[1].legs[1] = Node('>B12', '>B1', None)

elif unit == 3:

    root               = Node('A', None, np.ones((3,3)))
    root.legs[0]       = Node('<A1', 'A', np.ones((1,1)))
    root.legs[1]       = Node('>A2', 'A', np.ones((5,3)))
    
    root.legs[0].legs[0]  = Node('<B1','<A1', None)
    
    root.legs[0].legs[1]  = Node('>B2','<A1', np.ones((1,1)))
    root.legs[0].legs[1].legs[0] = Node('<C1','>B2')
    root.legs[0].legs[1].legs[1] = Node('>C2','>B2')

    root.legs[1].legs[0]  = Node('<D1','>A2', np.ones((9,5)))
    root.legs[1].legs[0].legs[0] = Node('<F1','<D1', np.ones((3,3)))
    root.legs[1].legs[0].legs[1] = Node('>F2','<D1', np.ones((3,3)))

    root.legs[1].legs[1] = Node('>D2','>A2', np.ones((1,1)))


elif unit == 4:

    root               = Node('A', None, np.ones((9,9)))
    root.legs[0]       = Node('<A1', 'A', np.ones((3,3)))
    root.legs[1]       = Node('>A2', 'A', np.ones((3,3)))

    root.legs[0].legs[0]  = Node('<B1','<A1', None)
    #root.legs[0].legs[0].legs[0]   = Node('G1','B1')
    #root.legs[0].legs[0].legs[1]  = Node('G2','B1')

    root.legs[0].legs[1]  = Node('>B2','<A1', np.ones((3,3)))
    root.legs[0].legs[1].legs[0] = Node('<BB1','>B2')
    root.legs[0].legs[1].legs[1] = Node('>BB2','>B2')

    root.legs[1].legs[0]  = Node('<C1','>A2', None)                 
    #root.legs[1].legs[0].legs[0]  = Node('F1','D1')                
    #root.legs[1].legs[0].legs[1] = Node('F2','D1')                

    root.legs[1].legs[1]  = Node('>C2','>A2', np.ones((3,3)))
    root.legs[1].legs[1].legs[0] = Node('<CC1','>C2')
    root.legs[1].legs[1].legs[1] = Node('>CC2','>C2')

elif unit == 5:

    root               = Node('A', None, np.ones((3,3)))
    root.legs[0]       = Node('<A1', 'A', np.ones((1,1)))
    root.legs[1]       = Node('>A2', 'A', np.ones((3,3)))

    root.legs[0].legs[0]  = Node('<B1','<A1', None)
    
    root.legs[0].legs[1]  = Node('>B2','<A1', np.ones((1,1)))
    root.legs[0].legs[1].legs[0] = Node('<C1','>B2')
    root.legs[0].legs[1].legs[1] = Node('>C2','>B2')
    
    root.legs[1].legs[0]  = Node('<D1','>A2', np.ones((3,3)))
    root.legs[1].legs[0].legs[0] = Node('<F1','<D1')
    root.legs[1].legs[0].legs[1] = Node('>F2','<D1')

    root.legs[1].legs[1] = Node('>D2','>A2', np.ones((1,1)))


"""
print("Level Order traversal of a binary tree is")
stack = reverseLevelOrder(root)
print(stack)
print('')
"""

print("Post Order traversal of a binary tree is")
print(f'Height: {height(root)}')
stack = postOrder(root)

for i in range(len(stack)):
    print(f"after post-order: [{i}]: name: {stack[i]['name']} | level: {stack[i]['level']} | mother: {stack[i]['mother']}")

stack = backward_traverse(stack)

if len(stack) != 1:
    print(stack)
    raise Exception('ERROR: stack did not contract')

print('Result:')
print(stack)
