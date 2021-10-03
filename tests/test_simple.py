# a simple function for use in our first test
def get_constant_42():
    return 42


# another simple function for use in our second test
def get_plus1(x):
    return x + 1


# here we define our tests
def test_get_constant():
    assert (get_constant_42() == 42)


def test_get_plus1():
    assert (get_plus1(4) == 5)
