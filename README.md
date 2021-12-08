# Autodiff-Visualizer Package
### AC 207 Fall 2021   

#### David Berthiaume
#### Lily Qian
#### Maggie Wu
#### Angel Hsu

Codecov:

[![codecov](https://codecov.io/gh/cs107-theteapeople/cs107-FinalProject/branch/main/graph/badge.svg?token=I1Z7BI3O7F)](https://codecov.io/gh/cs107-theteapeople/cs107-FinalProject)

## Introduction

Automatic differentiation involves efficiently and accurately evaluating derivatives of numeric functions from a computational perspective. In an era of rapidly advancing technology, automatic differentiation has many broad applications, especially in artificial intelligence and machine learning, allowing many computations to be performed efficiently. This has greatly expanded the scope and coverage of artificial intelligence to widely different applications to enhance human lives. While artificial intelligence is certainly a widely known application of automatic differentiation, the significance of automatic differentiation also derives from its potential applications in other fields such as computational fluid dynamics, atmospheric sciences, and engineering design and optimization.

This software package provides two libraries:
1. Autodiff provides an easy-to-use library that performs automatic differentiation of one or several user-supplied functions.
2. Visualizer visualizes and animates the automatic differentiation process of the user supplied function, so that users can have a clear understanding of the underlying algorithm and calculation processes through looking at the vivid graphs.

## Installation

To install the packages, you can either use PyPI or git clone.

Use PyPI to install the packages:

`pip install autodiffvis-teapeople`

Use git clone to install the packages:

`git clone https://github.com/cs107-theteapeople/cs107-FinalProject`

Once you have cloned the code, navigate to the root directory of the repository.  In the above example, this would be `cs107-FinalProject`.  

The library requirement for our project are numpy(>=1.20.3), matplotlib(>=3.4.3), and imageio(>=2.9.0). These can be installed with pip using the supplied requirements.txt file with the following command:

`pip install -r requirements.txt`

## User Guide

In the root directory of the repository, you can import the main autodiff module with the following code: 

`import autodiff as ad`

The general structure of usage will be as follows:
1. A user will instantiate a scalar variable using the autodiff module 
2. A user will define a function by combining variables, constants, and primitive functions through various operations using the autodiff module. Our autodiff library supports vector functions input, which means that the user can pass in multiple functions concurrently. 
3. Function value and derivatives will be calculated and returned for specific input points with respect to the specified variable using the `evaluate` function.  
 
Here is a toy example with a multi-variable vector function: 

Instantiate scalar variables:

`x = ad.var('x')`
`y = ad.var('y')`

Set user-defined composite functions that have vector valued outputs:

`f = [x*y, x+y, y+y, ad.cos(y-x), ad.arctan(x), ad.logistic(x+y*2)]`

Evaluate the function and derivative with respect to x:
  
```
print(ad.evaluate(f, x =.2, y =.1, wrt = [x]))
```

This will return both the value and the derivative of this function with respect to x evaluated at the given points as a numpy array or scalar value within a dictionary:

`[{'value': 0.020000000000000004, 'derivative': {'x': 0.1}}, 
{'value': 0.30000000000000004, 'derivative': {'x': 1}}, 
{'value': 0.2, 'derivative': {}}, 
{'value': 0.9950041652780258, 'derivative': {'x': -0.09983341664682815}}, 
{'value': 0.19739555984988078, 'derivative': {'x': 0.9615384615384615}}, 
{'value': 0.598687660112452, 'derivative': {'x': 0.24026074574152917}}]`

## Broader Impact and Inclusivity Statement

### Broader Impact

As one of the fundamental algorithms, automatic differentiation is used extensively across almost every area in science fields, ranging from physics, biology, genetics, applied mathematics, optimization, statistics, machine learning, health science, etc. Our goal with this software is to provide an automatic differentiation library that is easy to understand, read, and modify. We wish that our software can serve both an educational purpose as well as a practical use. We wish that through reading our code, using our software and plotting animated visualizations, potential users can understand the forward mode automatic differentiation thoroughly, and hence can better apply this algorithm in their own disciplines.

### Inclusivity Statement

Tea-people encourage users to modify the code and experiment with various techniques. We include elaborate documentation and detailed user guide so that users new to this package will not have a difficult time navigating the functionality of our package. As we are developing this package, the pull requests are reviewed and approved by every member of our team. We also welcome users to make pull requests and provide recommendations in every aspects of this software, from implementation, code efficiency, software organization, to additional features that would be great to include. Despite this package is written in English, we are still eager to hear different opinions from users in our diverse coding community. 

