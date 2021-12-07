import pytest
import math
import numpy as np
import sys
import os

sys.path.insert(1, './src/autodiff')
import autodiff as ad

def test_init_var():
    # test simple variable initialization
    # Create a one-dimensional variable
    x = ad.var('x')
    # Assign a value to x
    x.value = 3
    assert x.value == 3, "Error: incorrect initialization of variable"
    f = x
    assert f.evaluate(x=3)['value'] == 3, "Error: incorrect initialization of function"


def test_mul():
    # test multiplication with a constant
    x = ad.var('x')
    f = x * 3
    assert f.evaluate(x=3) == {'value': 9, 'derivative': {'x': 3}}, "Error: incorrect multiplication"


def test_rmul():
    # test (right) multiplication with a constant
    x = ad.var('x')
    f = 3 * x
    assert f.evaluate(x=3) == {'value': 9, 'derivative': {'x': 3}}, "Error: incorrect right multiplication"


def test_add():
    # test simple addition
    x = ad.var('x')
    f = x + 3
    assert f.evaluate(x=3) == {'value': 6, 'derivative': {'x': 1}}, "Error: incorrect addition"


def test_radd():
    # test right addition
    x = ad.var('x')
    f = 3 + x
    assert f.evaluate(x=3) == {'value': 6, 'derivative': {'x': 1}}, "Error: incorrect right addition"


def test_pow():
    # test raising to a constant power
    x = ad.var('x')
    f = x ** 2
    assert f.evaluate(x=3) == {'value': 9, 'derivative': {'x': 6}}, "Error: incorrect power"


def test_rpow():
    # test raising a constant to a variable power
    x = ad.var('x')
    f = math.e ** x
    assert f.evaluate(x=2) == {'value': math.e ** 2, 'derivative': {'x': math.e ** 2}}, "Error: incorrect right power"


def test_neg():
    # test negation and adding of negative constants
    x = ad.var('x')
    f = -x
    assert f.evaluate(x=3) == {'value': -3, 'derivative': {'x': -1}}, "Error: incorrect negative operation"
    g = - 5 + x
    assert g.evaluate(x=1) == {'value': -4, 'derivative': {'x': 1}}, "Error: incorrect negative operation"


def test_truediv():
    # test dividing by a constant
    x = ad.var('x')
    f = x / 3
    assert f.evaluate(x=3) == {'value': 1, 'derivative': {'x': 1 / 3}}, "Error: incorrect truediv"


def test_rtruediv():
    # test dividing a constant by a variable
    x = ad.var('x')
    f = 3 / x
    assert f.evaluate(x=3) == {'value': 1, 'derivative': {'x': - 1 / 3}}, "Error: incorrect right truediv"


def test_sub():
    # test subtraction
    x = ad.var('x')
    f = x - 3
    assert f.evaluate(x=1) == {'value': -2, 'derivative': {'x': 1}}, "Error: incorrect subtraction"


def test_rsub():
    # test right subtraction
    x = ad.var('x')
    f = 3 - x
    assert f.evaluate(x=2) == {'value': 1, 'derivative': {'x': -1}}, "Error: incorrect right subtraction"


def test_sin():
    # test the sin function
    x = ad.var('x')
    f = ad.sin(x)
    assert f.evaluate(x=1) == {'value': np.sin(1), 'derivative': {'x': np.cos(1)}}, "Error: incorrect sine function"


def test_cos():
    # test the cosine function
    x = ad.var('x')
    f = ad.cos(x)
    assert f.evaluate(x=1) == {'value': np.cos(1), 'derivative': {'x': -np.sin(1)}}, "Error: incorrect cosine function"

    assert f.evaluate(x=1 , wrt = [x]) == {'value': np.cos(1), 
                                           'derivative': {'x': -np.sin(1)}}, "Error: incorrect cosine function"

    with pytest.raises(ValueError):
        f.evaluate(x=2, wrt=[3])

def test_tan():
    # test the tangent function
    x = ad.var('x')
    f = ad.tan(x)
    assert f.evaluate(x=1) == {'value': np.tan(1),
                               'derivative': {'x': 1 / ((np.cos(1)) ** 2)}}, "Error: incorrect tangent function"

def test_arcsin():
    # test the tangent function
    x = ad.var('x')
    f = ad.arcsin(x)
    assert f.evaluate(x=0.5) == {'value': np.arcsin(0.5),
                               'derivative': {'x': 1 / np.sqrt(0.75)}}, "Error: incorrect arc-sin function"
    ## Capture nan value with error, taking square root of a negative number
    with pytest.raises(ValueError):
        f.evaluate(x=10)

def test_arccos():
    # test the tangent function
    x = ad.var('x')
    f = ad.arccos(x)
    assert f.evaluate(x=0.5) == {'value': np.arccos(0.5),
                               'derivative': {'x': -1 / np.sqrt(0.75)}}, "Error: incorrect arc-cos function"
    ## Capture nan value with error, taking square root of a negative number
    with pytest.raises(ValueError):
        f.evaluate(x=10)

def test_arctan():
    x = ad.var('x')
    f = ad.arctan(x)
    assert f.evaluate(x=10) == {'value': np.arctan(10),
                               'derivative': {'x': 1 / 101}}, "Error: incorrect arc-tan function"

def test_sinh():
    x = ad.var('x')
    f = ad.sinh(x)
    assert f.evaluate(x=10) == {'value': np.sinh(10),
                               'derivative': {'x': np.cosh(10)}}, "Error: incorrect sinh function"

def test_cosh():
    x = ad.var('x')
    f = ad.cosh(x)
    assert f.evaluate(x=10) == {'value': np.cosh(10),
                               'derivative': {'x': np.sinh(10)}}, "Error: incorrect cosh function"

def test_tanh():
    x = ad.var('x')
    f = ad.tanh(x)
    assert f.evaluate(x=1) == {'value': np.tanh(1),
                               'derivative': {'x': 1-(np.tanh(1))**2}}, "Error: incorrect tanh function"

def test_logistic():
    x = ad.var('x')
    f = ad.logistic(x)
    results = f.evaluate(x=5)
    assert np.isclose(results['value'], 1/(1+math.e**(-5))) and np.isclose(results['derivative']['x'], math.e**(-5)/(1+math.e**(-5))**2)

def test_evaluate_incorrect_var():
    # test to see if we enter incorrect variable names in evaluate
    x = ad.var('x')
    f = x + 3
    with pytest.raises(ValueError):
        f.evaluate(y=3)

def test_to_complex():
    # make sure we raise an exception when complex values would be created
    x = ad.var('x')
    f = x ** (1 / 2)
    with pytest.raises(ValueError):
        f.evaluate(x=-2)
    f = x ** 0.505
    with pytest.raises(ValueError):
        f.evaluate(x=-2)


def test_func_comp():
    # here we test a set of compound function to ensure that the chain rule is working correctly
    x = ad.var('x')
    f1 = ad.sin(2 * x)
    assert f1.evaluate(x=1) == {'value': np.sin(2),
                                'derivative': {'x': 2 * np.cos(2)}}, "Error: function composite – sin(2x)"

    f2 = ad.exp(2 * ad.sin(x ** 2))
    assert f2.evaluate(x=1) == {'value': np.exp(2 * np.sin(1)), 'derivative': {'x': 4 * np.exp(2 * np.sin(1)) * np.cos(
        1)}}, "Error: function composite – exp(2*sin(x**2))"

    f3 = x ** (ad.tan(x))
    val3, der3 = f3.evaluate(x=2)['value'], f3.evaluate(x=2)['derivative']['x']
    assert np.isclose(val3, 2 ** np.tan(2)) and np.isclose(der3, 2 ** np.tan(2) * (
            np.tan(2) / 2 + np.log(2) / (np.cos(2) ** 2))), "Error: function composite – exp(2*sin(x**2))"

    f4 = 2 ** ad.tan(ad.exp(x))
    val4, der4 = f4.evaluate(x=3)['value'], f4.evaluate(x=3)['derivative']['x']
    assert np.isclose(val4, 7.332336) and np.isclose(der4, 945.4314), "Error: function composite - 2**(tan(exp(x)))"

    f5 = ad.sin(x) * ad.cos(x ** 2) * ad.exp(7 * x)
    val5, der5 = f5.evaluate(x=1.1111)['value'], f5.evaluate(x=1.1111)['derivative']['x']
    assert np.isclose(val5, 705.7684) and np.isclose(der5,
                                                     802.6919), "Error: function composite - sin(x)cos(x**2)exp(7x)"

    f6 = ad.exp(3 * ad.tan(2 ** x))
    val6, der6 = f6.evaluate(x=1)['value'], f6.evaluate(x=1)['derivative']['x']
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
        f1.evaluate(x=1)
    # Test for value evaluation
    with pytest.raises(ValueError):
        f1.evaluate(x=-0.5)


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
    while abs(func.evaluate(x=x0)['value']) > precision and iteration < max_iter:
        x0 = x0 - func.evaluate(x=x0)['value'] / func.evaluate(x=x0)['derivative']['x']
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

    root = newton_solver(f, x, 10, precision=0.00000001, max_iter=3)
    assert root is None, "Newton solver doesn't coverge and returns none"


def test_constants():
    # here we test using various operations with constants
    # let's try a new variable name here to make sure these work ok
    y = ad.var('y')
    b = ad.const(4)

    f = y + b
    results = f.evaluate(y=1)
    assert results['value'] == 5 and results['derivative']['y'] == 1

    f = b + y
    results = f.evaluate(y=1)
    assert results['value'] == 5 and results['derivative']['y'] == 1

    f = y * b
    results = f.evaluate(y=1)
    assert results['value'] == 4 and results['derivative']['y'] == 4

    f = b ** y
    results = f.evaluate(y=2)
    assert results['value'] == 16 and np.isclose(results['derivative']['y'], np.log(4) * 4 ** 2)

    f = y ** b
    results = f.evaluate(y=2)
    assert results['value'] == 16 and np.isclose(results['derivative']['y'], 4 * (2 ** 3))


def test_literals():
    # here we test to make sure that ordinary integers and floats work OK with operator overloading
    y = ad.var('y')
    b = 4

    f = y + b
    results = f.evaluate(y=1)
    assert results['value'] == 5 and results['derivative']['y'] == 1

    f = b + y
    results = f.evaluate(y=1)
    assert results['value'] == 5 and results['derivative']['y'] == 1

    f = y * b
    results = f.evaluate(y=1)
    assert results['value'] == 4 and results['derivative']['y'] == 4

    f = b ** y
    results = f.evaluate(y=2)
    print(results)
    assert results['value'] == 16 and np.isclose(results['derivative']['y'], np.log(4) * 4 ** 2)

    f = y ** b
    results = f.evaluate(y=2)
    assert results['value'] == 16 and np.isclose(results['derivative']['y'], 4 * (2 ** 3))


def test_divide():
    # here we test a few functions that have a division in them
    x = ad.var('x')

    f = x / 4
    results = f.evaluate(x=4)
    assert results['value'] == 1 and np.isclose(results['derivative']['x'], 1 / 4)

    f = 4 / x
    results = f.evaluate(x=4)
    assert results['value'] == 1 and np.isclose(results['derivative']['x'], -4 / (4 ** 2))


def test_subtraction():
    # here we test a few functions that have a division in them
    x = ad.var('x')

    f = x - ad.const(4)
    results = f.evaluate(x=4)
    assert results['value'] == 0 and results['derivative']['x'] == 1

    f = ad.const(4) - x
    results = f.evaluate(x=4)
    assert results['value'] == 0 and results['derivative']['x'] == -1

    f = 4 - x
    results = f.evaluate(x=4)
    assert results['value'] == 0 and results['derivative']['x'] == -1

    f = x - 4
    results = f.evaluate(x=4)
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
    results = f.evaluate(x=4)
    assert results['value'] == 0 and results['derivative']['x'] == 0


def test_add_function():
    # test the add function
    # normally a user would likely just use x + y instead of add(x,y)
    # but we also provide this function
    x = ad.var('x')
    f = ad.add(x, 2)
    results = f.evaluate(x=4)
    assert results['value'] == 6 and results['derivative']['x'] == 1

    f = ad.add(x, x)
    results = f.evaluate(x=4)
    assert results['value'] == 8 and results['derivative']['x'] == 2

    f = ad.add(2, x)
    results = f.evaluate(x=4)
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
    results = f.evaluate(x=3)
    assert results['value'] == 1 and results['derivative']['x'] == 0


def test_enter_bad_values():
    # test entering a string into the evaluate function
    # this should raise a ValueError
    x = ad.var('x')
    f = x * x * x

    # entering a string into evaluate should raise a value error
    with pytest.raises(ValueError):
        f.evaluate(x='Chicken!')


def test_vector_functions_outputs():
    # add some tests for vector outputs
    x = ad.var('x')
    y = ad.var('y')
    f = [x * y, x + y, ad.cos(y - x)]
    results = ad.evaluate(f, x=.2, y=.1, wrt=[x])
    assert np.isclose(results[0]['value'], 0.02) and np.isclose(results[1]['value'], 0.3) and \
           np.isclose(results[2]['value'], 0.9950)
    assert np.isclose(results[0]['derivative']['x'], 0.1) and \
           np.isclose(results[1]['derivative']['x'], 1) and np.isclose(results[2]['derivative']['x'], -0.0998334)


def test_visualizer():
    # test the visualizer extension
    x = ad.var('x')
    y = ad.var('y')
    z = ad.var('z')
    f = ad.exp(x ** y + z + 3 + 0.5)
    results = f.evaluate(x=1, y=1, z=2)
    assert results['value'] == math.exp(6.5) and results['derivative']['x'] == math.exp(6.5) and results['derivative'][
        'y'] == 0 and results['derivative']['z'] == math.exp(6.5)
    print(f.evaluate(x=1, y=1, z=2, plot='./animate.gif'))
    assert os.path.exists('./animate.gif')
    os.unlink('./animate.gif')

def test_incorrect_number_arguments():
    # some functions only take in one input, like exp
    # if the wrong number of arguments are supplied, we raise an exception
    x = ad.var('x')
    y = ad.var('y')
    f = ad.exp(x,y)
    with pytest.raises(ValueError):
        f.evaluate(x=1, y=2, wrt=['x'])

    f = ad.add(x)
    with pytest.raises(ValueError):
        f.evaluate(x=1)

def test_visualizer_without_wrt():
    # test using the visualizer with wrt
    x = ad.var('x')
    y = ad.var('y')
    z = ad.var('z')
    f = ad.exp(x ** y + z + 3 + 0.5)
    results = f.evaluate(x=1, y=1, z=2)
    assert results['value'] == math.exp(6.5) and results['derivative']['x'] == math.exp(6.5) and \
           results['derivative']['y'] == 0 and results['derivative']['z'] == math.exp(6.5)
    print(f.evaluate(x=1, y=1, z=2,plot='./animate_without_wrt.gif'))
    assert os.path.exists('./animate_without_wrt.gif')
    os.unlink('./animate_without_wrt.gif')

def test_visualizer_with_wrt():
    # test using the visualizer with wrt
    x = ad.var('x')
    y = ad.var('y')
    z = ad.var('z')
    f = ad.exp(x ** y + z + 3 + 0.5)
    results = f.evaluate(x=1, y=1, z=2,wrt=['x','y','z'])
    assert results['value'] == math.exp(6.5) and results['derivative']['x'] == math.exp(6.5) and \
           results['derivative']['y'] == 0 and results['derivative']['z'] == math.exp(6.5)
    print(f.evaluate(x=1, y=1, z=2, wrt=['x','y','z'],plot='./animate_with_wrt.gif'))
    assert os.path.exists('./animate_with_wrt.gif')
    os.unlink('./animate_with_wrt.gif')

def test_visualizer_with_wrong_variable():
    # here we test using the wrong wrt argument
    x = ad.var('x')
    y = ad.var('y')
    z = ad.var('z')
    f = ad.exp(x ** y + z + 3 + 0.5)
    with pytest.raises(ValueError):
        # this should raise an error and not render anything
        print(f.evaluate(x=1, y=1, z=2,wrt=['k'],plot='./animate_with_wrt_wrong.gif'))

def test_vector_functions_outputs_x():
    # test vector outputs using the global evaluate function
    x = ad.var('x')
    y = ad.var('y')
    f = [x * y, x + y, ad.cos(y - x)]
    results = ad.evaluate(f, x=.2, y=.1, wrt=[x])
    assert np.isclose(results[0]['value'], 0.02) and \
           np.isclose(results[1]['value'], 0.3) and np.isclose(results[2]['value'], 0.9950)
    assert np.isclose(results[0]['derivative']['x'], 0.1) and \
           np.isclose(results[1]['derivative']['x'],1) and np.isclose(results[2]['derivative']['x'], -0.0998334)

def test_vector_functions_outputs_x_y():
    # test vector outputs
    x = ad.var('x')
    y = ad.var('y')
    f = [x * y, x + y, ad.cos(y - x)]
    results = ad.evaluate(f, x=.2, y=.1, wrt=[x,y])
    assert np.isclose(results[0]['value'], 0.02) and np.isclose(results[1]['value'], 0.3) \
           and np.isclose(results[2]['value'], 0.9950)
    assert np.isclose(results[0]['derivative']['x'], 0.1) and \
           np.isclose(results[1]['derivative']['x'],1) and np.isclose(results[2]['derivative']['x'], -0.0998334)
    assert np.isclose(results[0]['derivative']['y'], 0.2) \
           and np.isclose(results[1]['derivative']['y'],1) and np.isclose(results[2]['derivative']['y'], 0.0998334)


def test_vector_functions_outputs_wrong_variable():
    # test using the wrong variable for wrt in vector outputs
    x = ad.var('x')
    y = ad.var('y')
    k = ad.var('k')
    f = [x * y, x + y, ad.cos(y - x)]
    with pytest.raises(ValueError):
        results = ad.evaluate(f, x=.2, y=.1, wrt=[k])

def test_vector_functions_outputs_without_wrt():
    # test using vector outputs without the wrt argument
    x = ad.var('x')
    y = ad.var('y')
    f = [x * y, x + y, ad.cos(y - x)]
    results = ad.evaluate(f, x=.2, y=.1)
    assert np.isclose(results[0]['value'], 0.02) and \
           np.isclose(results[1]['value'], 0.3) and np.isclose(results[2]['value'], 0.9950)
    assert np.isclose(results[0]['derivative']['x'], 0.1) and \
           np.isclose(results[1]['derivative']['x'],1) and np.isclose(results[2]['derivative']['x'], -0.0998334)
    assert np.isclose(results[0]['derivative']['y'], 0.2) and \
           np.isclose(results[1]['derivative']['y'],1) and np.isclose(results[2]['derivative']['y'], 0.0998334)

def test_incorrect_wrt_types_scalar1():
    # test putting incorrect types into the wrt argument
    x = ad.var('x')
    y = ad.var('y')
    f = [x + y, x*y]
    with pytest.raises(ValueError):
        ad.evaluate(f, x=.2, y=.1, wrt=[1.5])

def test_incorrect_wrt_types_scalar2():
    # test putting incorrect types into the wrt argument
    x = ad.var('x')
    y = ad.var('y')
    f = [x + y, x*y]
    with pytest.raises(ValueError):
        ad.evaluate(f, x=.2, y=.3, wrt=f)

def test_incorrect_wrt_types_scalar3():
    # test putting incorrect types into the wrt argument
    x = ad.var('x')
    y = ad.var('y')
    f = [x + y, x * y]
    with pytest.raises(ValueError):
        ad.evaluate(f, x=.2, y=.2, wrt=[f])

def test_assign_string_to_variable():
    # test putting incorrect types into the wrt argument
    x = ad.var('x')
    y = ad.var('y')
    f = [x + y, x * y]
    with pytest.raises(ValueError):
        ad.evaluate(f, x=.2, y='incorrect')

def test_less_than():
    # test the less than operator
    x = ad.var('x')
    y = ad.var('y')
    f = x + 2 < y + 2
    results = f.evaluate(x = 1, y = 2)
    assert results['value'] == 1 and results['derivative']['x'] == 0

    f = x + 1 < 3
    results = f.evaluate(x = 1)
    assert results['value'] == 1 and results['derivative']['x'] == 0


def test_greater_than():
    # test the greater than operator
    x = ad.var('x')
    y = ad.var('y')
    f = x + 2 > y + 2
    results = f.evaluate(x=2, y=1)
    assert results['value'] == 1 and results['derivative']['x'] == 0

    f = x + 2 > 3
    results = f.evaluate(x=2)
    assert results['value'] == 1 and results['derivative']['x'] == 0


def test_less_than_or_equal():
    # test the less than or equal operator
    x = ad.var('x')
    y = ad.var('y')
    f = x + 2 <= y + 2
    results = f.evaluate(x = 1, y = 2)
    assert results['value'] == 1 and results['derivative']['x'] == 0

    f = x + 1 <= 3
    results = f.evaluate(x = 1)
    assert results['value'] == 1 and results['derivative']['x'] == 0


def test_greater_than_or_equal():
    # test the greater than or equal operator
    x = ad.var('x')
    y = ad.var('y')
    f = x + 2 >= y + 2
    results = f.evaluate(x=2, y=1)
    assert results['value'] == 1 and results['derivative']['x'] == 0

    f = x + 2 >= 3
    results = f.evaluate(x=2)
    assert results['value'] == 1 and results['derivative']['x'] == 0

def test_incorrect_arguments_types():
    # test various incorrect arguments to the global evaluate function
    x = ad.var('x')
    y = ad.var('y')
    f = [x + y, x * y]
    with pytest.raises(ValueError):
        ad.evaluate(f, x=.2)

    with pytest.raises(ValueError):
        # wrt should be a list
        ad.evaluate(f, x=.2, y=.1, wrt=x)

    with pytest.raises(ValueError):
        # wrt should only contain variables
        ad.evaluate(f, x=.2, y=.1, wrt=[f])

    with pytest.raises(ValueError):
        f[0].evaluate(x=.2, wrt=[f[0]])

    with pytest.raises(ValueError):
        # plotting is not supported for vector outputs
        ad.evaluate(f, x=.2, y=.1, plot = 'test.gif')

    results = ad.evaluate(x + y, x=.2, y=.1)
    assert( np.isclose(results[0]['value'],.3))

    with pytest.raises(ValueError):
        # plotting only works for scalar outputs
        ad.evaluate("f", x=.2, y=.1, plot = 'test.gif')

    with pytest.raises(ValueError):
        # test no variable arguments being supplied
        ad.evaluate(f)

    with pytest.raises(ValueError):
        # test no variables supplied to single node
        f[0].evaluate()

    with pytest.raises(ValueError):
        # incorrect type supplied to wrt
        f[0].evaluate(x=0.1, y=0.1, wrt=42)


def test_nans():
    # test functions that would return nans
    # these should raise a ValueError
    x = ad.var('x')
    f = ad.log(2, x)
    with pytest.raises(ValueError):
        f.evaluate(x=-2)

    f = ad.sqrt(x)
    with pytest.raises(ValueError):
        f.evaluate(x=-2)


