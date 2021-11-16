# autodiff
# AC207 final project
# Fall 2021

# our dependencies
import numpy as np

# this is a closure that allows to define a new function
# that can be used with autodiff
# for each function, a primary function and its derivative is supplied
def get_function( function , derivative ):
    """This is a closure that generates the necessary
      function to do node insertions into a binary graph
      and set up the necessary valuation and derivative
      function objects.

      arguments:
      function -- the function object to use for evaluation
                  This function takes two arguments x, and y
                  that are used for the calculation.  Both are
                  used in binary operations, and only x is
                  used in unary operations.
      derivative -- the function object to use for evaluation
                    of the derivative using the chain rule
                    This function takes four arguments x, y, xp,
                    and yp, representing the left value, right value
                    derivative of the left value, and derivative of
                    the right value.  These are combined together
                    to perform the chain rule for the computation
                    of our derivative.
      """


    # generate the function that can be used to elementary mathematical functions
    def inner_function( item, other_item = None ):
        """This inner function is what is generated
           and returned through the closure.  This inner function
           is responsible for generating the necessary binary tree
           insertions and storing the required functions for computing
           the function values and the derivatives

              arguments:
              item -- This is the left node or numeric value (both are supported)
              other_item -- This is the right node or numeric value
        """
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
    """This is a helper function that generates a Node that is a variable type

            arguments:
            var_name -- This is the name of the variable to create
      """
    return Node(var_name=var_name)

# this is a helper function that creates a constant
# with operator overloading, this is not needed
def const(value):
    """This is a helper function that generates a Node that is a constant type
        Note that this is not strictly needed as you can use python literals for
        numeric constants.

        arguments:
        value -- This is the value for the new constant
          """
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
    """This is our primary class for building the computation graph, traversing the
    computation graph and performing all of the necessary computations.  We do lazy
    evaluation in that the results are not computed immediately.  Rather, we have
    symbolic variables that act as placeholders for later evaluation.
    These nodes form a recursive binary tree that is used to perform the computations
    of the primary and forward traces.
    """
    # initialize the node
    # we set the node type based on the properties that were passed in
    # we set the nodes left and right children to None
    def __init__(self, var_name=None, value=None,
                 function=None, derivative=None):
        """This is the constructor of our Node class.

            arguments:
            var_name -- the name of the variable if this is a variable node
            value -- the value of this node if it is a constant
            function -- the function to use for computing the value of the function
            derivative -- the function to use for computing the derivative of the function
        """

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
        """This is the main function of our Node object to traverse the graph
           and perform the necessary calculations.  We traverse the binary graph
           in postorder and update the values and derivatives as we traverse the
           graph.

           First, we must ensure that the variables, supplied as named arguments
           match all of the variables in our graph.  If this is not the case, we
           raise a ValueError

           Once this check has passed, we traverse the graph and perform the needed
           evaluations.
        """
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
        """This function overloads the multiplication operator

        arguments:
        self -- the current node
        other -- the other node or numeric value (both are supported)
        """
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
        """This function overloads the true divide operator

           arguments:
           self -- the current node
           other -- the other node or numeric value (both are supported)
           """
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
        """This function overloads the right true divide operator

           arguments:
           self -- the current node
           other -- the other node or numeric value (both are supported)
           """
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
        """This function overloads the multiplication operator.
        Note that we simply call __mul__ since multiplication is commutative
        of the arguments

           arguments:
           self -- the current node
           other -- the other node or numeric value (both are supported)
           """
        return self.__mul__(other)

    # this is the addition operator overload
    def __add__(self, other):
        """This function overloads the add operator

           arguments:
           self -- the current node
           other -- the other node or numeric value (both are supported)
           """
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
        """This function overloads the right add operator
          Note that we simply call __add__ with the correct arguments
          since addition is commutative

           arguments:
           self -- the current node
           other -- the other node or numeric value (both are supported)
           """
        return self.__add__(other)

    # this is the addition operator overload
    def __sub__(self, other):
        """This function overloads the subtraction operator

           arguments:
           self -- the current node
           other -- the other node or numeric value (both are supported)
           """
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
        """This function overloads the right subtraction operator

           arguments:
           self -- the current node
           other -- the other node or numeric value (both are supported)
           """
        # we apply the add function to these two nodes
        new_node = Node(None, None, lambda x,y: y-x, lambda x,y,xp,yp: yp - xp)
        new_node.left = self
        if isinstance(other, Node):
            new_node.right = other
        else:
            new_node.right = Node(value=other)

        return new_node

    # the general derivative for our power function has certain cases that we need to be
    # careful for
    # instead of using a lambda, we use define the function here
    @staticmethod
    def _power_deriv(x, y, xp, yp):
        """Since computing the generic derivative of powers and we need to be
        careful where the derivative doesn't exist under the reals, we have
        a full function here instead of using a lambda

           arguments:
           x - the value of the left node
           y - the value of the right node
           xp - the derivative of the left node
           yp - the derivative of the right node
           """
        if np.isclose(x, 0) or np.isclose(yp, 0):
            return (x ** (y - 1)) * (y * xp)
        elif x < 0:
             raise ValueError('The derivative of a negative value raised to the specified power does not exist')

        else:
            return (x ** (y - 1)) * (y * xp + x * np.log(x) * yp)

    @staticmethod
    def _power_func(x,y):
        """Since computing the powers of numbers has special cases that result
        in complex numbers, we create a special function here for this.

           arguments:
           x - the value of the left node
           y - the value of the right node
       """

        if x < 0 and y.as_integer_ratio()[1] % 2 == 0:
            raise ValueError('Complex numbers are not supported. Cannot raise a negative number to a power'
                             ' that will result in a complex number')
        else:
            return x ** y

    # our generic power function
    # this is a static method that will be called from __pow__ and __rpow__
    @staticmethod
    def _power(left, right):
        """Our generic power function performs the necessary insertions into the
        binary tree when encountered.  This is used by the __pow__ and __rpow__
        operator overloads.

        arguments:
        left -- the left object which can be a node or a numeric value
        right -- the right object which can be a node or a numeric value
        """

        new_node = Node(None, None, Node._power_func, Node._power_deriv)
        # set the child nodes
        if isinstance(left, Node):
            new_node.left = left
        else:
            new_node.left = Node(value=left)

        if isinstance(right, Node):
            new_node.right = right
        else:
            new_node.right = Node(value=right)
        return new_node

    # the power function
    def __pow__(self, other):
        """This function overloads the power operator

        arguments:
        self -- the current node
        other -- the other node or numeric value (both are supported)
        """
        new_node = Node._power(self, other)
        return new_node

    # the power function with self as the exponent
    def __rpow__(self, other):
        """This function overloads the right power operator

        arguments:
        self -- the current node
        other -- the other node or numeric value (both are supported)
        """
        new_node = Node._power(other, self)
        return new_node

    # the unary negative operator
    def __neg__(self):
        """This function overloads the unary negation operator

        arguments:
        self -- the current node
        """
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
        """This function returns a nice string representation of our node

        arguments:
        self -- the current node
        """

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
        """This function prints the binary tree of nodes
        in preorder

        arguments:
        self -- the current node
        other -- the other node or numeric value (both are supported)
        """
        self.print_preorder(self)

    # this function is recursively called too get a set
    # of all variables that are in the tree
    @staticmethod
    def get_variables(root, vars):
        """This function finds a set of all variables that exist in
        the composite function

        arguments:
        root -- the root node to search from
        vars -- the current set of variables that were found
        """
        if root:
            if root.var_name is not None:
                vars.add(root.var_name)
            Node.get_variables(root.left, vars)
            Node.get_variables(root.right, vars)

    # print the tree in preorder recursively
    @staticmethod
    def print_preorder(root):
        """This prints the binary tree in preorder starting at the given root node

        arguments:
        self -- the current node
        """
        if root:
            print(root)
            Node.print_preorder(root.left)
            Node.print_preorder(root.right)

    # this function is used to print the tree in postorder
    def print_reverse(self):
        """This prints the binary tree in post order

        arguments:
        self -- the current node
        """
        Node.print_postorder(self)

    # print the tree in postorder recursively
    @staticmethod
    def print_postorder(root):
        """This prints the binary tree in post order starting at the given node
        This function is called recursively to traverse the tree

        arguments:
        root -- the node to start on
        """
        if root:
            Node.print_postorder(root.left)
            Node.print_postorder(root.right)
            print(root)

    # this function traverses the tree in postorder and
    # computes the primary and tangent traces
    # keeping track of both the value and the derivative
    @staticmethod
    def eval_post(root, var_values):
        """This our primary recursive computation engine for lazy evaluation.
        Our binary tree is traversed and the primary and tangent traces are updated
        in postorder.  All of the existing symbolic variables are substituted with
        the values supplied in the call to eval. This function is used internally
        and is called recursively.

        arguments:
        root -- the current node
        var_values -- the list of variable values supplied to the call to eval
        """
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
