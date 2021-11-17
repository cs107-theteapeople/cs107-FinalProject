import pytest
import math
import numpy as np
import sys
sys.path.insert(1,'./autodiff')
import autodiff as ad



def test_init_var():
    print (dir(ad))
    # Create a one-dimensional variable
    x = ad.var('x')
    # Assign a value to x
    x.value = 3
    assert x.value == 3, "Error: incorrect initialization of variable"
    f = x
    assert f.eval(x = 3)['value'] == 3, "Error: incorrect initialization of function"
    
def test_mul():
    x = ad.var('x')
    f = x * 3
    assert f.eval(x=3) == {'value': 9, 'derivative': 3}, "Error: incorrect multiplication"

def test_rmul():
    x = ad.var('x')
    f = 3 * x
    assert f.eval(x=3) == {'value': 9, 'derivative': 3}, "Error: incorrect right multiplication"

def test_add():
    x = ad.var('x')
    f = x + 3
    assert f.eval(x=3) == {'value': 6, 'derivative': 1}, "Error: incorrect addition"

def test_radd():
    x = ad.var('x')
    f = 3 + x
    assert f.eval(x=3) == {'value': 6, 'derivative': 1}, "Error: incorrect right addition"

def test_pow():
    x = ad.var('x')
    f = x ** 2
    assert f.eval(x=3) == {'value': 9, 'derivative': 6}, "Error: incorrect power"

def test_rpow():
    x = ad.var('x')
    f = math.e ** x
    assert f.eval(x=2) == {'value': math.e ** 2, 'derivative': math.e ** 2}, "Error: incorrect right power"

def test_neg():
    x = ad.var('x')
    f = -x
    assert f.eval(x=3) == {'value': -3, 'derivative': -1}, "Error: incorrect negative operation"
    g = - 5 + x
    assert g.eval(x=1) == {'value': -4, 'derivative': 1}, "Error: incorrect negative operation"

def test_truediv():
    x = ad.var('x')
    f = x / 3
    assert f.eval(x=3) == {'value': 1, 'derivative': 1 / 3}, "Error: incorrect truediv"

def test_rtruediv():
    x = ad.var('x')
    f = 3 / x
    assert f.eval(x=3) == {'value': 1, 'derivative': - 1 / 3}, "Error: incorrect right truediv"
    
def test_sub():
    x = ad.var('x')
    f = x - 3
    assert f.eval(x=1) == {'value': -2, 'derivative': 1}, "Error: incorrect subtraction"
    
def test_rsub():
    x = ad.var('x')
    f = 3 - x
    assert f.eval(x=2) == {'value': 1, 'derivative': -1}, "Error: incorrect right subtraction"

def test_sin():
    x = ad.var('x')
    f = ad.sin(x)
    assert f.eval(x=1) == {'value': np.sin(1), 'derivative': np.cos(1)}, "Error: incorrect sine function"

def test_cos():
    x = ad.var('x')
    f = ad.cos(x)
    assert f.eval(x=1) == {'value': np.cos(1), 'derivative': -np.sin(1)}, "Error: incorrect cosine function"

def test_tan():
    x = ad.var('x')
    f = ad.tan(x)
    assert f.eval(x=1) == {'value': np.tan(1), 'derivative': 1 / ((np.cos(1)) ** 2)}, "Error: incorrect tangent function"


def test_eval_incorrect_var():
    x = ad.var('x')
    f = x + 3
    assert f.eval(y=3) == None, "Error: does not deal with incorrect variables in evaluation as expected"

def test_print():
    x = ad.var('x')
    f = x + 3
    assert print(f) == None, "Error: print does not return None"
    
def test_func_comp():
    x = ad.var('x')
    f1 = ad.sin(2 * x)
    assert f1.eval(x=1) == {'value': np.sin(2), 'derivative': 2 * np.cos(2)}, "Error: function composite – sin(2x)"
    f2 =ad.exp(2 * ad.sin(x ** 2))
    assert f2.eval(x=1) == {'value': np.exp(2*np.sin(1)), 'derivative': 4*np.exp(2*np.sin(1))*np.cos(1)}, "Error: function composite – exp(2*sin(x**2))"
    f3 = x ** (ad.tan(x))
    val3, der3 = f3.eval(x=2)['value'], f3.eval(x=2)['derivative']
    assert np.isclose(val3, 2**np.tan(2)) and np.isclose(der3, 2**np.tan(2)*(np.tan(2)/2+np.log(2)/(np.cos(2)**2))), "Error: function composite – exp(2*sin(x**2))"
    f4 = 2 ** ad.tan(ad.exp(x))
    val4, der4 = f4.eval(x=3)['value'], f4.eval(x=3)['derivative']
    assert np.isclose(val4, 7.332336) and np.isclose(der4, 945.4314), "Error: function composite - 2**(tan(exp(x)))"
    f5 = ad.sin(x) * ad.cos(x**2) * ad.exp(7*x)
    val5, der5 = f5.eval(x=1.1111)['value'], f5.eval(x=1.1111)['derivative']
    assert np.isclose(val5, 705.7684) and np.isclose(der5, 802.6919), "Error: function composite - sin(x)cos(x**2)exp(7x)"
    f6 = ad.exp(3*ad.tan(2**x))
    val6, der6 = f6.eval(x=1)['value'], f6.eval(x=1)['derivative']
    assert np.isclose(val6, np.exp(3*np.tan(2))) and np.isclose(der6, 0.034169), "Error: function composite - sin(x)cos(x**2)exp(7x)"
    
def test_func_error():
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
    iteration = 0
    while abs(func.eval(x=x0)['value']) > precision and iteration < max_iter: 
        x0 = x0 - func.eval(x=x0)['value'] / func.eval(x=x0)['derivative']
        iteration += 1
    if iteration == max_iter:
        print(f"Cannot find root with Newton's method with {max_iter} iterations.")
        return None
    return x0
    
def test_newton_solver():
    x = ad.var('x')
    f = 3*x**2 + 5*x - 4
    root = newton_solver(f, x, 1, precision = 0.001, max_iter=20000)
    assert np.isclose(root, (-5+np.sqrt(73))/6) or np.isclose(root, (-5-np.sqrt(73))/6), "Newton solver not working properly, or reaches maximum iterations"
