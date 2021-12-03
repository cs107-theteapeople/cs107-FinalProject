import pytest
import math
import numpy as np
import sys

# for now to enable git actions for codecov
# once we have create an installable package
# after milestone 2, this will be removed
sys.path.insert(1, './autodiff')
import autodiff as ad


def test_init_var():
    # test simple variable initialization
    # Create a one-dimensional variable
    x = ad.var('x')
    # Assign a value to x
    x.value = 3
    assert x.value == 3, "Error: incorrect initialization of variable"
    f = x
    assert f.eval(x=3)['value'] == 3, "Error: incorrect initialization of function"


def test_mul():
    # test multiplication with a constant
    x = ad.var('x')
    f = x * 3
    assert f.eval(x=3) == {'value': 9, 'derivative': {'x':3}}, "Error: incorrect multiplication"


def test_rmul():
    # test (right) multiplication with a constant
    x = ad.var('x')
    f = 3 * x
    assert f.eval(x=3) == {'value': 9, 'derivative': {'x':3}}, "Error: incorrect right multiplication"


def test_add():
    # test simple addition
    x = ad.var('x')
    f = x + 3
    assert f.eval(x=3) == {'value': 6, 'derivative': {'x':1}}, "Error: incorrect addition"


def test_radd():
    # test right addition
    x = ad.var('x')
    f = 3 + x
    assert f.eval(x=3) == {'value': 6, 'derivative': {'x':1}}, "Error: incorrect right addition"


def test_pow():
    # test raising to a constant power
    x = ad.var('x')
    f = x ** 2
    assert f.eval(x=3) == {'value': 9, 'derivative': {'x':6}}, "Error: incorrect power"


def test_rpow():
    # test raising a constant to a variable power
    x = ad.var('x')
    f = math.e ** x
    assert f.eval(x=2) == {'value': math.e ** 2, 'derivative': {'x':math.e ** 2}}, "Error: incorrect right power"


def test_neg():
    # test negation and adding of negative constants
    x = ad.var('x')
    f = -x
    assert f.eval(x=3) == {'value': -3, 'derivative': {'x':-1}}, "Error: incorrect negative operation"
    g = - 5 + x
    assert g.eval(x=1) == {'value': -4, 'derivative': {'x':1}}, "Error: incorrect negative operation"


def test_truediv():
    # test dividing by a constant
    x = ad.var('x')
    f = x / 3
    assert f.eval(x=3) == {'value': 1, 'derivative': {'x':1 / 3}}, "Error: incorrect truediv"


def test_rtruediv():
    # test dividing a constant by a variable
    x = ad.var('x')
    f = 3 / x
    assert f.eval(x=3) == {'value': 1, 'derivative': {'x':- 1 / 3}}, "Error: incorrect right truediv"


def test_sub():
    # test subtraction
    x = ad.var('x')
    f = x - 3
    assert f.eval(x=1) == {'value': -2, 'derivative': {'x':1}}, "Error: incorrect subtraction"


def test_rsub():
    # test right subtraction
    x = ad.var('x')
    f = 3 - x
    assert f.eval(x=2) == {'value': 1, 'derivative': {'x':-1}}, "Error: incorrect right subtraction"


def test_sin():
    # test the sin function
    x = ad.var('x')
    f = ad.sin(x)
    assert f.eval(x=1) == {'value': np.sin(1), 'derivative': {'x':np.cos(1)}}, "Error: incorrect sine function"


def test_cos():
    # test the cosine function
    x = ad.var('x')
    f = ad.cos(x)
    assert f.eval(x=1) == {'value': np.cos(1), 'derivative': {'x':-np.sin(1)}}, "Error: incorrect cosine function"


def test_tan():
    # test the tangent function
    x = ad.var('x')
    f = ad.tan(x)
    assert f.eval(x=1) == {'value': np.tan(1),
                           'derivative': {'x':1 / ((np.cos(1)) ** 2)}}, "Error: incorrect tangent function"


def test_eval_incorrect_var():
    # test to see if we enter incorrect variable names in eval
    x = ad.var('x')
    f = x + 3
    with pytest.raises(ValueError):
        f.eval(y=3)

def test_to_complex():
    # make sure we raise an exception when complex values would be created
    x = ad.var('x')
    f = x ** (1/2)
    with pytest.raises(ValueError):
        f.eval(x=-2)
    f = x ** 0.505
    with pytest.raises(ValueError):
        f.eval(x=-2)

def test_func_comp():
    # here we test a set of compound function to ensure that the chain rule is working correctly
    x = ad.var('x')
    f1 = ad.sin(2 * x)
    assert f1.eval(x=1) == {'value': np.sin(2), 'derivative': {'x':2 * np.cos(2)}}, "Error: function composite – sin(2x)"

    f2 = ad.exp(2 * ad.sin(x ** 2))
    assert f2.eval(x=1) == {'value': np.exp(2 * np.sin(1)), 'derivative': {'x':4 * np.exp(2 * np.sin(1)) * np.cos(
        1)}}, "Error: function composite – exp(2*sin(x**2))"

    f3 = x ** (ad.tan(x))
    val3, der3 = f3.eval(x=2)['value'], f3.eval(x=2)['derivative']['x']
    assert np.isclose(val3, 2 ** np.tan(2)) and np.isclose(der3, 2 ** np.tan(2) * (
                np.tan(2) / 2 + np.log(2) / (np.cos(2) ** 2))), "Error: function composite – exp(2*sin(x**2))"

    f4 = 2 ** ad.tan(ad.exp(x))
    val4, der4 = f4.eval(x=3)['value'], f4.eval(x=3)['derivative']['x']
    assert np.isclose(val4, 7.332336) and np.isclose(der4, 945.4314), "Error: function composite - 2**(tan(exp(x)))"

    f5 = ad.sin(x) * ad.cos(x ** 2) * ad.exp(7 * x)
    val5, der5 = f5.eval(x=1.1111)['value'], f5.eval(x=1.1111)['derivative']['x']
    assert np.isclose(val5, 705.7684) and np.isclose(der5,
                                                     802.6919), "Error: function composite - sin(x)cos(x**2)exp(7x)"

    f6 = ad.exp(3 * ad.tan(2 ** x))
    val6, der6 = f6.eval(x=1)['value'], f6.eval(x=1)['derivative']['x']
    assert np.isclose(val6, np.exp(3 * np.tan(2))) and np.isclose(der6,
           0.034169), "Error: function composite - sin(x)cos(x**2)exp(7x)"


def test_func_error():
    # make sure that we raise errors when trying to calculate the values or derivatives
    # that shouldn't exist in the reals
    # test if the functions raise errors properly
    x = ad.var('x')
    f1 = (-2) ** x
    # Test for derivative evaluation
    with pytest.raises(ValueError):
        f1.eval(x=1)
    # Test for value evaluation
    with pytest.raises(ValueError):
        f1.eval(x=-0.5)


def newton_solver(func, x, x0, precision, max_iter):
    """This function performs a simple test newton solver using the autodiff package.
    This is a test for iteratively computing the value and derivative of a
    compound function at various inputs.

    arguments:
    func - our autodiff functino to solve
    x - our variable to solve for
    x0 - our initial guess
    precision - the precision to solve to
    max_iter - the maximum number of iterations to use
    """

    iteration = 0
    while abs(func.eval(x=x0)['value']) > precision and iteration < max_iter:
        x0 = x0 - func.eval(x=x0)['value'] / func.eval(x=x0)['derivative']['x']
        iteration += 1
    if iteration == max_iter:
        print(f"Cannot find root with Newton's method with {max_iter} iterations.")
        return None
    return x0


def test_newton_solver():
    # as another test, we implement a simple newton solver and make sure our library
    # correctly converges to the root, calling autodiff iteratively
    x = ad.var('x')
    f = 3 * x ** 2 + 5 * x - 4
    root = newton_solver(f, x, -4, precision=0.01, max_iter=200)
    assert np.isclose(root, (-5 + np.sqrt(73)) / 6) or np.isclose(root, (-5 - np.sqrt(73)) / 6), \
        "Newton solver not working properly, or reaches maximum iterations"

    root = newton_solver(f, x, 10, precision = 0.00000001, max_iter = 3)
    assert root is None, "Newton solver doesn't coverge and returns none"

def test_constants():
    # here we test using various operations with constants
    # let's try a new variable name here to make sure these work ok
    y = ad.var('y')
    b = ad.const(4)

    f = y + b
    results = f.eval(y=1)
    assert results['value'] == 5 and results['derivative']['y'] == 1

    f = b + y
    results = f.eval(y=1)
    assert results['value'] == 5 and results['derivative']['y'] == 1

    f = y * b
    results = f.eval(y=1)
    assert results['value'] == 4 and results['derivative']['y'] == 4

    f = b ** y
    results = f.eval(y=2)
    assert results['value'] == 16 and np.isclose(results['derivative']['y'], np.log(4) * 4 ** 2)

    f = y ** b
    results = f.eval(y=2)
    assert results['value'] == 16 and np.isclose(results['derivative']['y'], 4 * (2 ** 3))

def test_literals():
    # here we test to make sure that ordinary integers and floats work OK with operator overloading
    y = ad.var('y')
    b = 4

    f = y + b
    results = f.eval(y=1)
    assert results['value'] == 5 and results['derivative']['y'] == 1

    f = b + y
    results = f.eval(y=1)
    assert results['value'] == 5 and results['derivative']['y'] == 1

    f = y * b
    results = f.eval(y=1)
    assert results['value'] == 4 and results['derivative']['y'] == 4

    f = b ** y
    results = f.eval(y=2)
    print (results)
    assert results['value'] == 16 and np.isclose(results['derivative']['y'], np.log(4) * 4 ** 2)

    f = y ** b
    results = f.eval(y=2)
    assert results['value'] == 16 and np.isclose(results['derivative']['y'], 4 * (2 ** 3))

def test_divide():
    # here we test a few functions that have a division in them
    x = ad.var('x')

    f = x / 4
    results = f.eval(x=4)
    assert results['value'] == 1 and np.isclose(results['derivative']['x'] , 1/4)

    f = 4 / x
    results = f.eval(x=4)
    assert results['value'] == 1 and np.isclose(results['derivative']['x'] , -4 / (4**2))

def test_subtraction():
    # here we test a few functions that have a division in them
    x = ad.var('x')

    f = x - ad.const(4)
    results = f.eval(x=4)
    assert results['value'] == 0 and results['derivative']['x'] == 1

    f = ad.const(4) - x
    results = f.eval(x=4)
    assert results['value'] == 0 and results['derivative']['x'] == -1

    f = 4 - x
    results = f.eval(x=4)
    assert results['value'] == 0 and results['derivative']['x'] == -1

    f = x - 4
    results = f.eval(x=4)
    assert results['value'] == 0 and results['derivative']['x'] == 1



def test_print():
    # these should return None and not raise an exception
    x = ad.var('x')
    x = x ** x + 2
    result = x.print()
    assert result is None

    result = x.print_reverse()
    assert result is None

def test_subtract_self():
    # testing the case of x - x.  This should return 0 for the value
    # and the derivative
    x = ad.var('x')
    f = x - x
    results = f.eval(x=4)
    assert results['value'] == 0 and results['derivative']['x'] == 0

def test_add_function():
    # test the add function
    # normally a user would likely just use x + y instead of add(x,y)
    # but we also provide this function
    x = ad.var('x')
    f = ad.add(x, 2)
    results = f.eval(x=4)
    assert results['value'] == 6 and results['derivative']['x'] == 1

    f = ad.add(x, x)
    results = f.eval(x=4)
    assert results['value'] == 8 and results['derivative']['x'] == 2

    f = ad.add(2, x)
    results = f.eval(x=4)
    assert results['value'] == 6 and results['derivative']['x'] == 1

def test_bad_constant():
    # this should raise an exception
    with pytest.raises(ValueError):
        c = ad.const('hello world')

def test_self_divide():
    # test dividing a variable by itself.  This ends up as a constant
    # and the derivative should be 0.
    x = ad.var('x')
    f = x / x
    results = f.eval(x=3)
    assert results['value'] == 1 and results['derivative']['x'] == 0

def test_enter_bad_values():
    # test entering a string into the eval function
    # this should raise a ValueError
    x = ad.var('x')
    f = x * x * x

    # entering a string into eval should raise a value error
    with pytest.raises(ValueError):
        f.eval(x='Chicken!')

def test_visualizer():
    x = ad.var('x')
    y = ad.var('y')
    z = ad.var('z')
    
    f = ad.exp(x**y+z+3+0.5)
    results = f.eval(x=1, y=1, z=2)
    assert results['value'] == math.exp(6.5) and results['derivative']['x'] == math.exp(6.5) and results['derivative']['y'] ==  0 and results['derivative']['z'] == math.exp(6.5)   
    print(f.eval(x=1, y=1, z=2, plot = 'animate'))
    f.print()

def test_wrt():
    x = ad.var('x')
    y = ad.var('y')
    f = ad.exp(x+y)
    with pytest.raises(ValueError):
        f.eval(x=1, y=2, wrt=['z', 'q'])
    
    
    
    
    