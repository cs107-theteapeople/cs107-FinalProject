# autodiff
# AC207 final project
# Fall 2021

# the visualization extension
from autodiff import visualizer

# our dependencies
import numpy as np
import imageio
import warnings

# this is a closure that allows to define a new function
# that can be used with autodiff
# for each function, a primary function and its derivative is supplied
def get_function( function_name, function , derivative ):
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
        new_node = Node(None, None, function, derivative, function_name)
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
    if isinstance(value, int) or isinstance(value, float):
        return Node(value=value)
    else:
        raise ValueError("Only integers and floating point numbers are supported for constants.")


# user the helper closure above, this is how we define functions
# each function can now be defined in a single line
# for each function, we supply the primary function and its derivative
# lambda functions can be used here
sin = get_function('sin', np.sin,   lambda x,xp : xp * np.cos(x))
cos = get_function('cos', np.cos,   lambda x,xp : -xp * np.sin(x))
exp = get_function('exp', np.exp,   lambda x,xp : xp * np.exp(x))
tan = get_function('tan', np.tan,   lambda x,xp : xp * (1/np.cos(x))**2)
add = get_function('add', lambda x,y: x+y, lambda x,y,xp,yp: xp + yp)

# compute the log to any base (the second argument is the base)
log = get_function('log', lambda x,y: np.log(x) / np.log(y), lambda x,y,xp,yp:
                            (xp*np.log(y)/x - (np.log(x)*yp/y)) / (np.log(y)**2))
ln =  get_function('ln',  lambda x : np.log(x), lambda x,xp: xp/x)
arcsin = get_function('arcsin', lambda x: np.arcsin(x), lambda x,xp: xp / (np.sqrt( 1 - x**2 )))
arccos = get_function('arccos', lambda x: np.arccos(x), lambda x,xp: - xp / (np.sqrt( 1 - x**2)))
arctan = get_function('arctan', lambda x: np.arctan(x), lambda x,xp: xp / (x**2 + 1))
sinh = get_function('sinh', lambda x: np.sinh(x), lambda x, xp: xp * np.cosh(x))
cosh = get_function('cosh', lambda x: np.cosh(x), lambda x, xp: xp * np.sinh(x))
tanh = get_function('tanh', lambda x: np.tanh(x), lambda x, xp: xp * (1 / np.cosh(x))**2)
logistic = get_function('logistic', lambda x: np.exp(x) / (1 + np.exp(x)), lambda x, xp:
                        np.exp(x) * xp / ((1 + np.exp(x)) ** 2))
sqrt = get_function('sqrt', lambda x: np.sqrt(x), lambda x, xp: xp / (2 * np.sqrt(x)))

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
                 function=None, derivative=None,
                 function_name=None):
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
            self.deriv = None
        elif value != None:
            self.type = 'const'
            self.deriv = None
        else:
            self.type = 'inter'
            self.deriv = None

        # A node can have a function and a derivative
        # these are applied as we traverse the graph in reverse order
        self.function = function
        self.derivative = derivative
        self.function_name = function_name
        self.left = None
        self.right = None

        # used for visualization
        self.order = 0
        self.depth = 0

    # this function recursively evaluates a binary tree starting at
    # a root node
    def evaluate(self, **kwargs):
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

        # see if the user requested plotting
        if 'plot' in kwargs:
            plot = kwargs['plot']
        else:
            plot = False

        vars = set()
        # the first thing we do is determine what variables are defined in
        # our tree
        Node.get_variables(self, vars)

        # we remove the keywords from the set of supplied variables
        supplied_vars = set(kwargs.keys()) - {'wrt', 'plot'}

        if 'wrt' in kwargs:

            # the user may supply a list of strings, a list of variables, or a mix of both
            # let's convert this uniformly to a list of strings

            wrt_pre = kwargs['wrt']

            if not (isinstance(wrt_pre, list)):
                raise ValueError('Incorrect type supplied to wrt. '
                                 'Please supply a list of variables or strings of variable names')
            wrt = []


            for item in wrt_pre:
                if not (isinstance(item, Node) or isinstance(item, str)):
                    raise ValueError('Incorrect type supplied to wrt')
                if isinstance(item, Node):
                    if not item.var_name:
                        raise ValueError('Incorrect node type supplied to wrt.  Please supply a variable.')
                    wrt.append(item.var_name)
                else:
                    wrt.append(item)

            if not set(wrt) <= vars:
                raise ValueError('Variables specified in wrt do not match the variables in the equation')
        else:
            wrt = supplied_vars

        # check to see if the variable types are numeric
        for key, val in kwargs.items():
            if key in supplied_vars:
                if not (isinstance(val, int) or isinstance(val, float)):
                    raise ValueError(f'Attempting to assign a non-numeric value to variable {key}:{val}')

        # if the variables do not match, raise an error
        if supplied_vars != vars:
            print ('variables do not match')
            print (f'the variables in this tree are {vars}')

            print (f'the variables supplied by evaluate are {supplied_vars}')
            raise ValueError('Supplied variables do not match those in the equation.')

        # let's reset the values and derivatives in the tree
        Node.reset(self)

        # now we recursively traverse through the tree in postorder
        # computing the value and derivative along the way
        if plot:
            # add the depths and an order of the nodes for plotting
            image, fig, font_size, depth_counts = visualizer.render_first_frame(self)
            images = [image]
            Node.eval_post(self, kwargs, wrt, images, self, fig, font_size, depth_counts)
            file_path = plot
            print()
            # pause the video a bit at the end
            for i in range(6):
                images.append(images[-1])

            print (f'saving render to {file_path}')
            imageio.mimsave(file_path, images, fps=2, format='GIF')
            print ()

        else:
            Node.eval_post(self, kwargs, wrt)

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
                        lambda x,y,xp,yp: x*yp + y*xp, 'x')
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
                        lambda x,y,xp,yp: ((y * xp - x * yp) / (y**2)), '/')
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
                        lambda x,y,xp,yp: ((x * yp - y * xp) / (x**2)), '/')
        # set the child nodes
        new_node.left = self
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
        new_node = Node(None, None, np.add, lambda x,y,xp,yp: xp + yp, '+' )
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
        new_node = Node(None, None, np.subtract, lambda x,y,xp,yp: xp - yp, '-' )
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
        new_node = Node(None, None, lambda x,y: y-x, lambda x,y,xp,yp: yp - xp, '-')
        new_node.left = self
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
        val = x ** y
        # do we end up with a complex number?
        if not (isinstance(val, int) or (isinstance(val, float))):
            raise ValueError('Raising {x} to the {y}th power results in a complex number')

        return val

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

        new_node = Node(None, None, Node._power_func, Node._power_deriv, '^')
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

    # overloaded less than operator
    def __lt__(self, other):
        """This function overloads the less than operator

               arguments:
               self -- the current node
               other -- the other node or numeric value (both are supported)
               """
        # we apply the add function to these two nodes
        new_node = Node(None, None, lambda x,y: int(x < y), lambda x, y, xp, yp: 0, '<')
        new_node.left = self
        if isinstance(other, Node):
            new_node.right = other
        else:
            new_node.right = Node(value=other)
        return new_node

    # overloaded greater than operator
    def __gt__(self, other):
        """This function overloads the greater than operator

               arguments:
               self -- the current node
               other -- the other node or numeric value (both are supported)
               """
        # we apply the add function to these two nodes
        new_node = Node(None, None, lambda x, y: int(x > y), lambda x, y, xp, yp: 0, '>')
        new_node.left = self
        if isinstance(other, Node):
            new_node.right = other
        else:
            new_node.right = Node(value=other)
        return new_node

    # overloaded less than or equal operator
    def __le__(self, other):
        """This function overloads the less than or equal operator

               arguments:
               self -- the current node
               other -- the other node or numeric value (both are supported)
               """
        # we apply the add function to these two nodes
        new_node = Node(None, None, lambda x,y: int(x <= y), lambda x, y, xp, yp: 0, '<=')
        new_node.left = self
        if isinstance(other, Node):
            new_node.right = other
        else:
            new_node.right = Node(value=other)
        return new_node

    # overloaded greater than or equal operator
    def __ge__(self, other):
        """This function overloads the greater than or equal operator

               arguments:
               self -- the current node
               other -- the other node or numeric value (both are supported)
               """
        # we apply the add function to these two nodes
        new_node = Node(None, None, lambda x, y: int(x >= y), lambda x, y, xp, yp: 0, '>=')
        new_node.left = self
        if isinstance(other, Node):
            new_node.right = other
        else:
            new_node.right = Node(value=other)
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
                        lambda x,xp: -xp, 'negation')

        new_node.left = self
        return new_node

    def __str__(self):
        """This function returns a nice string representation of our node

        arguments:
        self -- the current node
        """

        if self.type == 'inter':
            rv = f'[type:{self.type} ' \
                f'value:{self.value} ' \
                f'function:{self.function.__name__} ' \
                f'depth:{self.depth} ' \
                f'order:{self.order}]'
        else:
            rv = f' (type:{self.type} name:{self.var_name} ' \
                f'value:{self.value} ' \
                f'order:{self.order}) '
        return rv

    # print the binary tree in preorder
    def print(self):
        """This function prints the binary tree of nodes
        in preorder.  Eventually we may expand this to
        print a nice table or tree with the nodes.

        arguments:
        self -- the current node
        other -- the other node or numeric value (both are supported)
        """
        self.print_preorder(self)

    # this function is recursively called to reset
    # all of the variables and derivatives in the tree
    @staticmethod
    def reset(root):
        if root:
            if root.type != 'const':
                # we reset the value and deriv for the new graph traversal
                root.value = None
                root.deriv = None
            Node.reset(root.left)
            Node.reset(root.right)

    # this function is recursively called to get a set
    # of all variables that are in the tree
    @staticmethod
    def get_variables(root, vars):
        """This function finds a set of all variables that exist in
        the composite function by searching the binary tree

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
    def eval_post(root, var_values, wrt, images = None, root_render = None,
                  fig = None, font_size = None, depth_counts = None):
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
            Node.eval_post(root.left, var_values, wrt, images, root_render, fig, font_size, depth_counts)
            Node.eval_post(root.right, var_values, wrt, images, root_render, fig, font_size, depth_counts)
            # if a function is attached to this node, we apply it to the
            # children
            # this works similar to activation functions in neural networks
            if root.function:
                root.deriv = {}
                if root.right is None:
                    try:
                        # disable invalid value warnings since we catch them later
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            root.value = root.function(root.left.value)
                    except:
                        raise ValueError(f'Incorrect number of arguments supplied to function: {root.function_name}')
                    if np.isnan(root.value):
                        raise ValueError(f'Invalid value encountered in function: {root.function_name}')

                    for key in wrt:
                        root.deriv[key] = root.derivative(root.left.value, root.left.deriv[key])
                else:
                    try:
                        # disable invalid value warnings since we catch them later
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            root.value = root.function(root.left.value, root.right.value)
                    except:
                        raise ValueError(f'Incorrect number of arguments supplied to function: {root.function_name}')
                    if np.isnan(root.value):
                        raise ValueError(f'Invalid value encountered in function: {root.function_name}')
                    for key in wrt:
                        root.deriv[key] = root.derivative(root.left.value, root.right.value,
                                                          root.left.deriv[key], root.right.deriv[key])
            # if we have a variable, we set the node value to the value
            # that was set in the eval call
            elif root.var_name:
                root.value = var_values[root.var_name]
                # here make a dictionary to store our derivatives
                root.deriv = {}
                for key in wrt:
                    if key == root.var_name:
                        root.deriv[key] = 1
                    else:
                        root.deriv[key] = 0

            # if the node is a constant, we set the derivatives to 0
            elif root.type == 'const':
                root.deriv = {}
                for key in wrt:
                    root.deriv[key] = 0

            # append the frame to the movie
            if images:
                images.extend(visualizer.frame(root_render, fig, font_size, depth_counts, root))

# this our main function to evaluate functions with vector outputs
# for each output, we call the eval method of that node
# to use the visualizer on a particular output, run the eval member function on that node
def evaluate( nodes , **kwargs):
    if not (isinstance(nodes, list) or isinstance(nodes, Node)):
        raise ValueError('Please specify a node or list of nodes as the first argument to eval')

    if 'plot' in kwargs:
        raise ValueError('rendering is only supported for a single output (otherwise the plot gets very busy).'
                         'Please select a single node that you would like to render.')

    # convert the input to a list if only one Node is found
    if isinstance(nodes, Node):
        nodes = [nodes]

    if 'wrt' in kwargs:
        wrt_pre = kwargs['wrt']
    else:
        wrt_pre = list(set(kwargs.keys()) - {'wrt'})

    if not isinstance(wrt_pre, list):
        raise ValueError('Please supply a list of variables or strings for the wrt argument')

    # we need to convert this to a list of variable names so that we can
    # do the needed intersection against each output node
    wrt = []
    for item in wrt_pre:
        if not (isinstance(item, Node) or isinstance(item, str)):
            raise ValueError('Incorrect type supplied to wrt')
        if isinstance(item, Node):
            if not item.var_name:
                raise ValueError('Incorrect type supplied to wrt')
            wrt.append(item.var_name)
        else:
            wrt.append(item)

    # let's build a list of varables within each node and then a master set
    # of all available variables to check that the inputs are correct
    vars = []
    for node in nodes:
        local_vars = set()
        Node.get_variables(node, local_vars)
        vars.append(local_vars)

    all_vars = set.union(*vars)

    # check to make sure our inputs match the full set of possible variables
    supplied_vars = set(kwargs.keys()) - {'wrt'}
    if not (supplied_vars == all_vars):
        raise ValueError('Please specify values for every variable that is present in this vector valued function, '
                         'and only variables that appear in the function.')

    if not set(wrt) <= all_vars:
        raise ValueError('An variable was supplied to wrt that is not in the function')

    # store our list of results
    results = []
    for node, lvars in zip(nodes, vars):
        # for each node, we determine the variables that belong to the binary tree corresponding
        # to the node, perform an intersection of the variables, and supply these variables and
        # (optionally) the wrt parameter to the eval member function of each node to
        # generate the vector output.

        # grab the variables for this subset
        var_values = {i: kwargs[i] for i in lvars}

        supplied_wrt = []
        for w in wrt:
            if w in lvars:
                supplied_wrt.append(w)

        # evaluate the node and append our results
        results.append(node.evaluate( **var_values, wrt = supplied_wrt))

    return results

