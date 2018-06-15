# vimpy: nonparametric variable importance assessment in python

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


**Author:** Brian Williamson

## Introduction

In predictive modeling applications, it is often of interest to determine the relative contribution of subsets of features in explaining an outcome; this is often called variable importance. It is useful to consider variable importance as a function of the unknown, underlying data-generating mechanism rather than the specific predictive algorithm used to fit the data. This package provides functions that, given fitted values from predictive algorithms, compute nonparametric estimates of deviance- and variance-based variable importance, along with asymptotically valid confidence intervals for the true importance.

## Installation

You may install a stable release of `vimpy` using `pip` by running `python pip install vimpy` from a Terminal window. Alternatively, you may install within a `virtualenv` environment.

You may install the current dev release of `vimpy` by downloading this repository directly.

## Issues

If you encounter any bugs or have any specific feature requests, please [file an issue](https://github.com/bdwilliamson/vimpy/issues).

## Example

This example shows how to use `vimpy` in a simple setting with simulated data and using a single regression function. For more examples and detailed explanation, please see the `R` vignette (to come).

```python
## load required libraries
import numpy as np
import vimpy
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

## -------------------------------------------------------------
## problem setup
## -------------------------------------------------------------
## define a function for the conditional mean of Y given X
def cond_mean(x = None):
    f1 = np.where(np.logical_and(-2 <= x[:, 0], x[:, 0] < 2), np.floor(x[:, 0]), 0) 
    f2 = np.where(x[:, 1] <= 0, 1, 0)
    f3 = np.where(x[:, 2] > 0, 1, 0)
    
    f6 = np.absolute(x[:, 5]/4) ** 3
    f7 = np.absolute(x[:, 6]/4) ** 5
    
    f11 = (7./3)*np.cos(x[:, 10]/2)
    
    ret = f1 + f2 + f3 + f6 + f7 + f11
    
    return ret

## create data
np.random.seed(4747)
n = 100
p = 15
s = 1 # importance desired for X_1
x = np.zeros((n, p))
for i in range(0, x.shape[1]) :
    x[:,i] = np.random.normal(0, 2, n)

y = cond_mean(x) + np.random.normal(0, 1, n)

## -------------------------------------------------------------
## preliminary step: get regression estimators
## -------------------------------------------------------------
## use grid search to get optimal number of trees and learning rate
ntrees = np.arange(100, 3500, 500)
lr = np.arange(.01, .5, .05)
    
param_grid = [{'n_estimators':ntrees, 'learning_rate':lr}]

## set up cv objects
cv_full = GridSearchCV(GradientBoostingRegressor(loss = 'ls', max_depth = 1), param_grid = param_grid, cv = 5)
cv_small = GridSearchCV(GradientBoostingRegressor(loss = 'ls', max_depth = 1), param_grid = param_grid, cv = 5)

## fit the full regression
cv_full.fit(x, y)
full_fit = cv_full.best_estimator_.predict(x)

## fit the reduced regression
x_small = np.delete(x, s, 1) # delete the columns in s
cv_small.fit(x_small, full_fit)
small_fit = cv_small.best_estimator_.predict(x_small)

## -------------------------------------------------------------
## get variable importance estimates
## -------------------------------------------------------------
## set up the vimp object
vimp = vimpy.vimp_regression(y, x, full_fit, small_fit, s)
## get the naive estimator
vimp.plugin()
## get the corrected estimator
vimp.update()
vimp.onestep_based_estimator()
## get a standard error
vimp.onestep_based_se()
## get a confidence interval
vimp.get_ci()
```