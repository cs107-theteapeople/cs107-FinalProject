# autodiff
# AC207 final project
# Fall 2021

# our dependencies
import numpy as np

# this is a closure that allows to define a new function
# that can be used with autodiff
# for each function, a primary function and its derivative is supplied
def get_function( function , derivative ):

    # generate the function that can be used to elementary mathematical functions
    def inner_function( item, other_item = None ):
        new_node = Node(None, None, function, derivative)
        # we support Node objects or numeric values
        # if a numeric value is passed, we turn it into a constant Node
        if isinstance(item, Node):
            new_node.left = item
        else:
            new_node.left = Node(value=item)
        # some functions are binary and supply an other_item.
        # if this item is a Node, we set that to the right child node
        # if it is a numeric value, we create a new constant node
        if other_item is not None:
            if isinstance(other_item, Node):
                new_node.right = other_item
            else:
                new_node.right = Node(value=other_item)

        return new_node
    # return the inner function
    return inner_function

# this is a helper function that creates a variable node
def var(var_name):
    return Node(var_name=var_name)

# this is a helper function that creates a constant
# with operator overloading, this is not needed
def const(value):
    return Node(value=value)

# user the helper closure above, this is how we define functions
# each function can now be defined in a single line
# for each function, we supply the primary function and its derivative
# lambda functions can be used here
sin = get_function(np.sin,   lambda x,xp : xp * np.cos(x))
cos = get_function(np.cos,   lambda x,xp : -xp * np.sin(x))
exp = get_function(np.exp,   lambda x,xp : xp * np.exp(x))
tan = get_function(np.tan,   lambda x,xp : xp * (1/np.cos(x))**2)
pow = get_function(np.power, lambda x,y,xp,yp : yp * y * (x**(y-1)))


# here is our main class
# this class defines a node of a binary tree
# there are 3 types of nodes:
# 1. A constant node holds a constant numeric value
# 2. A variable node is a primary input variable
#      values can be assigned to these variables when eval is called
# 3. An intermediate node is a node that is neither a variable nor a constant
#      and is used to carry the forward mode to evaluate the function and compute
#      the derivative
class Node:
    # initialize the node
    # we set the node type based on the properties that were passed in
    # we set the nodes left and right children to None
    def __init__(self, var_name=None, value=None,
                 function=None, derivative=None):

        self.var_name = var_name
        self.value = value
        if not var_name is None:
            self.type = 'var'
            self.deriv = 1
        elif value != None:
            self.type = 'const'
            self.deriv = 0
        else:
            self.type = 'inter'
            self.deriv = 0

        # A node can have a function and a derivative
        # these are applied as we traverse the graph in reverse order
        self.function = function
        self.derivative = derivative
        self.left = None
        self.right = None

    # this function recursively evaluates a binary tree starting at
    # a root node
    def eval(self, **kwargs):
        vars = set()
        # the first thing we do is determine what variables are defined in
        # our tree
        Node.get_variables(self, vars)

        # if the variables
        if kwargs.keys() != vars:
            print ('variables do not match')
            print (f'the variables in this tree are {vars}')
            print (f'the variables supplied by eval are {set(kwargs.keys())}')
            return None

        # now we recursively traverse through the tree in postorder
        # computing the value and derivative along the way
        Node.eval_post(self, kwargs)
        # return the value and the derivative
        return {'value': self.value, 'derivative': self.deriv}

    # this is the multiplication operator overload
    def __mul__(self, other):
        # we apply the multiplication function to these two nodes
        new_node = Node(None, None, np.multiply,
                        lambda x,y,xp,yp: x*yp + y*xp)
        # set the child nodes
        new_node.left = self
        if isinstance(other, Node):
            new_node.right = other
        else:
            new_node.right = Node(value=other)
        return new_node

    # this is the divide operator overload
    def __truediv__(self, other):
        # we apply the division function to these two nodes
        new_node = Node(None, None, lambda x,y: x/y,
                        lambda x,y,xp,yp: ((y * xp - x * yp) / (y**2)))
        # set the child nodes
        new_node.left = self
        if isinstance(other, Node):
            new_node.right = other
        else:
            new_node.right = Node(value=other)
        return new_node

    # this is the divide operator overload
    def __rtruediv__(self, other):
        # we apply the divide function in reverse order
        new_node = Node(None, None, lambda x,y: y/x,
                        lambda x,y,xp,yp: ((x * yp - y * xp) / (x**2)))
        # set the child nodes
        new_node.left = self
        if isinstance(other, Node):
            new_node.right = other
        else:
            new_node.right = Node(value=other)
        return new_node

    # right multiplication
    def __rmul__(self, other):
        return self.__mul__(other)

    # this is the addition operator overload
    def __add__(self, other):
        # we apply the add function to these two nodes
        new_node = Node(None, None, np.add, lambda x,y,xp,yp: xp + yp )
        new_node.left = self
        if isinstance(other, Node):
            new_node.right = other
        else:
            new_node.right = Node(value=other)

        return new_node

    # right addition
    def __radd__(self, other):
        return self.__add__(other)

    # this is the addition operator overload
    def __sub__(self, other):
        # we apply the add function to these two nodes
        new_node = Node(None, None, np.subtract, lambda x,y,xp,yp: xp - yp )
        new_node.left = self
        if isinstance(other, Node):
            new_node.right = other
        else:
            new_node.right = Node(value=other)

        return new_node

    # right addition
    def __rsub__(self, other):
        # we apply the add function to these two nodes
        new_node = Node(None, None, lambda x,y: y-x, lambda x,y,xp,yp: yp - xp)
        new_node.left = self
        if isinstance(other, Node):
            new_node.right = other
        else:
            new_node.right = Node(value=other)

        return new_node

    # the power function
    def __pow__(self, other):
        new_node = Node(None, None, np.power,
                        lambda x,y,xp,yp: (x**(y-1)) * (y * xp + x * np.log(x) * yp))
        # set the child nodes
        new_node.left = self
        if isinstance(other, Node):
            new_node.right = other
        else:
            new_node.right = Node(value=other)
        return new_node

    # the power function with self as the exponent
    def __rpow__(self, other):
        new_node = Node(None, None, lambda x,y: float(y)**x,
                        lambda x,y,xp,yp: (y**(x-1)) * (x * yp + y * np.log(y) * xp))
        # set the child nodes
        new_node.left = self
        if isinstance(other, Node):
            new_node.right = other
        else:
            new_node.right = Node(value=other)
        return new_node

    # the unary negative operator
    def __neg__(self):
        new_node = Node(None, None, lambda x: -x,
                        lambda x,xp: -xp)
        # we support Node objects or numeric values
        # if a numeric value is passed, we turn it into a constant Node
        if isinstance(self, Node):
            new_node.left = self
        else:
            new_node.left = Node(value=self)
        return new_node

    def __str__(self):
        if self.type == 'inter':
            rv = f'[type:{self.type} ' \
                f'value:{self.value} ' \
                f'deriv:{self.deriv} ' \
                f'function:{self.function.__name__}]'
        else:
            rv = f' (type:{self.type} name:{self.var_name} ' \
                f'value:{self.value} deriv:{self.deriv})'
        return rv

    # print the binary tree in preorder
    def print(self):
        self.print_preorder(self)

    # this function is recursively called too get a set
    # of all variables that are in the tree
    @staticmethod
    def get_variables(root, vars):
        if root:
            if root.var_name is not None:
                vars.add(root.var_name)
            Node.get_variables(root.left, vars)
            Node.get_variables(root.right, vars)

    # print the tree in preorder recursively
    @staticmethod
    def print_preorder(root):
        if root:
            print(root)
            Node.print_preorder(root.left)
            Node.print_preorder(root.right)

    # this function is used to print the tree in postorder
    def print_reverse(self):
        Node.print_postorder(self)

    # print the tree in postorder recursively
    @staticmethod
    def print_postorder(root):
        if root:
            Node.print_postorder(root.left)
            Node.print_postorder(root.right)
            print(root)

    # this function traverses the tree in postorder and
    # computes the primary and tangent traces
    # keeping track of both the value and the derivative
    @staticmethod
    def eval_post(root, var_values):
        if root:
            Node.eval_post(root.left, var_values)
            Node.eval_post(root.right, var_values)
            # if a function is attached to this node, we apply it to the
            # children
            # this works similar to activation functions in neural networks
            if root.function:
                if root.right is None:
                    root.value = root.function(root.left.value)
                    root.deriv = root.derivative(root.left.value, root.left.deriv)
                else:
                    root.value = root.function(root.left.value, root.right.value)
                    root.deriv = root.derivative(root.left.value, root.right.value,
                                                 root.left.deriv, root.right.deriv)
            # if we have a variable, we set the node value to the value
            # that was set in the eval call
            elif root.var_name:
                root.value = var_values[root.var_name]
                root.deriv = 1
