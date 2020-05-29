## utility functions


def deprecated(func):
    '''This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.'''
    import warnings

    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func


## get the measure function, IC function
def get_measure_function(type):
    from .predictiveness_measures import r_squared, accuracy, auc, deviance
    if type == "r_squared":
        measure = r_squared
    elif type == "accuracy":
        measure = accuracy
    elif type == "auc":
        measure = auc
    elif type == "deviance":
        measure = deviance
    else:
        raise ValueError("We do not currently support the entered predictiveness measure. Please provide a different predictiveness measure.")
    return measure


def choose(n, k):
    import math
    return int(math.factorial(n) / (math.factorial(k) * math.factorial(n - k)))


def make_folds(y, V, stratified = True):
    """
    Create folds for CV (potentially stratified)
    """
    import numpy as np
    if stratified:
        y_1 = y == 1
        y_0 = y == 0
        folds_1 = np.resize(np.arange(V), sum(y_1))
        np.random.shuffle(folds_1)
        folds_0 = np.resize(np.arange(V), sum(y_0))
        np.random.shuffle(folds_0)
        folds = np.empty((y.shape[0]))
        folds[np.ravel(y_1)] = folds_1
        folds[np.ravel(y_0)] = folds_0
    else:
        folds = np.resize(np.arange(V), y.shape[0])
        np.random.shuffle(folds)
    return folds


## hypothesis testing with shapley values
def shapley_hyp_test(vs_one_1, v_none_0, sigmas_one, sigma_none, delta = 0, level = 0.05, p = 3):
    """
    Hypothesis testing for Shapley values

    @param vs_one_1: one-feature measures of predictiveness
    @param v_none_0: null-model predictiveness
    @param sigmas_one: ses
    @param sigma_none: null-model se
    @param delta: value for testing
    @param level: significance level

    @return: test_statistics (the test statistics), p_vals (p-values), hyp_tests (the hypothesis testing results)
    """
    import numpy as np
    from scipy.stats import norm

    test_statistics = [(vs_one_1[v] - v_none_0 - delta) / (np.sqrt(sigmas_one[v] ** 2 + sigma_none ** 2)) for v in range(p)]
    p_values = 1. - norm.cdf(test_statistics)
    hyp_tests = p_values < level
    return test_statistics, p_values, hyp_tests
