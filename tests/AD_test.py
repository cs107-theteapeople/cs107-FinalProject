import pytest
import math
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
    