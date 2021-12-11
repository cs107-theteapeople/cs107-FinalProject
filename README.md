# Autodiff-Visualizer Package
### AC 207 Fall 2021   

#### David Berthiaume
#### Lily Qian
#### Maggie Wu
#### Angel Hsu

Codecov:

[![codecov](https://codecov.io/gh/cs107-theteapeople/cs107-FinalProject/branch/main/graph/badge.svg?token=I1Z7BI3O7F)](https://codecov.io/gh/cs107-theteapeople/cs107-FinalProject)

Our full documentation can be found at [here](docs/documentation.md).

## Introduction

Automatic differentiation involves efficiently and accurately evaluating derivatives of numeric functions from a computational perspective. In an era of rapidly advancing technology, automatic differentiation has many broad applications, especially in artificial intelligence and machine learning, allowing many computations to be performed efficiently. This has greatly expanded the scope and coverage of artificial intelligence to widely different applications to enhance human lives. While artificial intelligence is certainly a widely known application of automatic differentiation, the significance of automatic differentiation also derives from its potential applications in other fields such as computational fluid dynamics, atmospheric sciences, and engineering design and optimization.

This software package provides two libraries:
1. Autodiff provides an easy-to-use library that performs automatic differentiation of one or several user-supplied functions.
2. Visualizer visualizes and animates the automatic differentiation process of the user supplied function, so that users can have a clear understanding of the underlying algorithm and calculation processes through looking at the vivid graphs.

## How to use autodiff

A user will interact with the automatic differentiation functionality through the autodiff module. This module uses automatic differentiation to calculate the Jacobian of a user supplied function. 

### Installation

To install the packages, you can either use PyPI or git clone and then the supplied setup.py file.

Use PyPI to install the packages:

`pip install autodiffvis-teapeople`

Use git clone to download the package:

`git clone https://github.com/cs107-theteapeople/cs107-FinalProject`

Once you have cloned the code, navigate to the root directory of the repository.  In the above example, this would be `cs107-FinalProject`.  

Once there, the software can be installed with

`python setup.py install`

The library requirement for our project are numpy(>=1.20.3), matplotlib(>=3.4.3), and imageio(>=2.9.0). All of these dependencies are automatically installed when autodiff is installed.  

If desired, these can be separately installed with pip using the supplied requirements.txt file
with the following command. 

`pip install -r requirements.txt`

### User guide

`import autodiff as ad`

The general structure of usage follows:
1. A user will instantiate scalar variables using the autodiff module 
2. A user will define a function by combining variables, constants, and primitive functions through various operations using the autodiff module.  Vector functions can be defined by using lists of scalar functions.
3. The function value and derivatives will be calculated and returned for specific input values using the `evaluate` function.  
 
The following is an example for single function input:

Instantiate a scalar variable:

`x = ad.var(‘x’)`

Set a user-defined composite function by combining elementary functions available in the autodiff module:

`f = ad.exp(ad.sin(x + 3 * x**2) * ad.cos(x))`

Evaluate the function and derivative at multiple points:

```
import numpy as np
for i in np.linspace(1,3,10):
    print(f.evaluate(x=i))
```

This will return both the value and the derivative of this function with respect to x evaluated at the given points as a dictionary.  For example each call to f.evaluate returns a dictionary that looks like the following:

`{'value': -3.027209981231713, 'derivative': -3.3713769787623757}`

The following is an example for vector inputs:

Instantiate scalar variables:

`x = ad.var('x')`

`y = ad.var('y')`

Set user-defined composite functions that have vector valued outputs:

`f = [x*y, x+y, y+y, ad.cos(y-x), ad.arctan(x), ad.logistic(x+y*2)]`

Evaluate the function and derivative with respect to x:
  

`ad.evaluate(f, x =.2, y =.1, wrt = [x])`

This will return both the value and the derivative of this function with respect to x evaluated at the given points as a list of dictionaries. 

`[{'value': 0.020000000000000004, 'derivative': {'x': 0.1}},`

`{'value': 0.30000000000000004, 'derivative': {'x': 1}},`

`{'value': 0.2, 'derivative': {}}, `

`{'value': 0.9950041652780258, 'derivative': {'x': -0.09983341664682815}},` 

`{'value': 0.19739555984988078, 'derivative': {'x': 0.9615384615384615}},` 

`{'value': 0.598687660112452, 'derivative': {'x': 0.24026074574152917}}]`

Note the use of the `wrt` parameter argument.  This can be used to limit which derivatives are returned.  If this is not
specified, all derivatives are returned for the function.  In this example above, it would return the derivative with respect to both x and y.

### Advanced usage - seed dictionary

A custom seed dictionary can be supplied to the evaluate method instead of supplying variables to the wrt argument.  The following examples show how to use this functionality.

```
x = ad.var('x')
y = ad.var('y')
f = ad.cos(x) + ad.sin(y)
f.evaluate(x=.1, y=.1, seed_dict={'x':1, 'y':1})
f.evaluate(x=.1, y=.1, seed_dict={'x':0, 'y':1})
```

By default, if a seed value isn't supplied for a variable, it is set to 1 in that pass, meaning that the normal derivative is calculated for that variable. Note that each derivative is computed individually, and this seed dictionary applies a scale to that derivative.  For example, using a seed dictionary of `{'x':1, 'y':1}` will compute the derivatives as normal, but using a seed dictionary of `{'x':1, 'y':0}` will set the y derivative to 0.  This is very different from the seed **vector** used in class. Also, unlike the `wrt` argument, this will set the derivatives to 0 and still return them.  The `wrt` argument is used to only return a subset of the derivatives. 


### How to use the forward computation visualizer

The visualizer package comes along with the autodiff package when installing. It is located in the same folder as the autodiff package. It is imported into the autodiff package and is used as part of the autodiff module. Hence, when you import autodiff module, the visualizer package will be automatically imported. 

Below is an example of using visualizer for a scalar-input function:

Instantiate a scalar variable:

```
x = ad.var('x')
```
Set a user-defined function that has scalar valued outputs:

```
f = ad.exp(x*ad.sin(2*x))
```

Generate the animated visualization of the function evaluation processes:

```
print(f.evaluate(x=1, plot='<filepath>/animate_demo_scalar.gif'))
```

The animate_demo_scalar.gif file will be stored at the filepath the user specified in the plot argument.

![](docs/animate_demo_scalar.gif)

Below is an example of using visualizer for a function with multiple input variables. 

Instantiate scalar variables:

```
x = ad.var('x')
y = ad.var('y')
z = ad.var('z')
```

Set the user-defined composite function

`f = ad.exp(x ** y + z + 3 + 0.5)`

Generate the animated visualization of the function evaluation processes with respect to x,y,z evaluated at x=1, y=1, and z=2:

```
print(f.evaluate(x=1, y=1, z=2, plot='<filepath>/animate_demo_scalar.gif'))
```

The animate_demo.gif file will be stored at the filepath the user specified in the plot argument.

![](docs/animate_demo.gif)

The user is able to see the forward mode evaluation process, with values and traces of derivatives being displayed at each step. 

## Broader impact and inclusivity statement

### Broader impact

As one of the fundamental algorithms, automatic differentiation is used extensively across almost every area in science fields, ranging from physics, biology, genetics, applied mathematics, optimization, statistics, machine learning, and health science. Our goal with this software is to provide an automatic differentiation library that is easy to understand, read, and modify. We wish that our software can serve both an educational purpose and also be of practical use. We wish that through reading our code, using our software and plotting animated visualizations, potential users can understand the forward mode automatic differentiation thoroughly, and hence can better apply this algorithm in their own disciplines.

### Inclusivity statement

Tea-people encourage users to modify the code and experiment with various techniques. We include elaborate documentation and detailed user guide so that users new to this package will not have a difficult time navigating the functionality of our package. As we are developing this package, the pull requests are reviewed and approved by every member of our team. We also welcome users to make pull requests and provide recommendations in every aspect of this software, from implementation, code efficiency, software organization, to additional features that would be great to include. Despite this package is written in English, we are still eager to hear different opinions from users in our diverse coding community. 

