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
    assert ad.Node.print_postorder(f) == None, "Error: static method print_postorder() does not return None"
    assert ad.Node.print_preorder(f) == None, "Error: static method print_preorder() does not return None"
    assert ad.Node.print_reverse(f) == None, "Error: static method print_reverse() does not return None"
    
def test_func_comp():
    x = ad.var('x')
    f1 = ad.sin(2 * x)
    assert f1.eval(x=1) == {'value': np.sin(2), 'derivative': 2 * np.cos(2)}, "Error: function composite – sin(2x)"
