#######################################################################################
##
## FILE: cv_vim.py
##
## PURPOSE: Python class for variable importance based on the deviance measure
##
## CREATED: 26 October 2018 by Brian Williamson
##
## PARAMETERS: y     - the outcome Y
##             x     - the data
##             f     - the fitted values from the full regression, a list
##             m     - the fitted values from the reduced regression, a list
##             V     - the number of cross-validation folds
##             folds - the folds used for cross-validation: a vector, for no donsker; 
##                     a matrix, for donsker
##         vim_type  - type of regression: "deviance" or "regression"
##             s     - the set of features for importance
#######################################################################################

## import required libraries
import numpy as np
from scipy.stats import norm
from vimpy import vimp_regression
from vimpy import vimp_deviance

class cv_vim:

    ## define initialization values
    """
    y is the outcome, a list
    x is the feature data
    f is the fitted values from a regression of Y on X
    m is the fitted values from a regression of f (or Y) on X(-s)
    V is the number of cross-validation folds
    folds is the folds used for cross-validation
    vim_type is the type of regression to run
    s is the set of features for importance
    """
    def __init__(self, y, x, f, m, V, folds, vim_type, s):
        self.y_ = y
        self.x_ = x
        self.n_ = np.sum([y[v].shape[0] for v in range(V)])
        self.p_ = np.mean([np.mean(y[v], axis = 0) for v in range(V)], axis = 0)
        self.f_ = f
        self.m_ = m
        self.folds_ = folds
        self.V_ = V
        self.vim_type_ = vim_type
        self.s_ = s
        self.naive_ = []
        self.update_ = []
        self.vimp_ = []
        self.se_ = []
        self.ci_ = []

    ## calculate the plug-in estimator
    def plugin(self):
        naive_cv = [None]*self.V_
        for v in range(self.V_):
            if self.vim_type_ == "deviance":
                est_v = vimp_deviance(self.y_[v], self.x_, self.f_[v], self.m_[v], self.s_)
            elif self.vim_type_ == "regression":
                est_v = vimp_regression(self.y_[v], self.x_, self.f_[v], self.m_[v], self.s_)    
            naive_cv[v] = est_v.plugin().naive_
        self.naive_ = np.array([np.mean(naive_cv)])
        return(self)

    ## calculate the update
    def update(self):
        updates = [None]*self.V_
        for v in range(self.V_):
            if self.vim_type_ == "deviance":
                est_v = vimp_deviance(self.y_[v], self.x_, self.f_[v], self.m_[v], self.s_)
            elif self.vim_type_ == "regression":
                est_v = vimp_regression(self.y_[v], self.x_, self.f_[v], self.m_[v], self.s_)    
            updates[v] = est_v.update().update_
        ## get the update (will be already averaged, in case folds have unequal numbers)
        self.update_ = np.array([np.mean(update) for update in updates])
        return self

    ## calculate the variable importance based on the one-step
    def onestep_based_estimator(self):
        self.vimp_ = self.naive_ + np.mean(self.update_)
        return self

    ## calculate the standard error based on the one-step correction
    def onestep_based_se(self):
        ses = [None]*self.V_
        update = [None]*self.V_
        for v in range(self.V_):
            if self.vim_type_ == "deviance":
                est_v = vimp_deviance(self.y_[v], self.x_, self.f_[v], self.m_[v], self.s_)
            elif self.vim_type_ == "regression":
                est_v = vimp_regression(self.y_[v], self.x_, self.f_[v], self.m_[v], self.s_)
            update[v] = est_v.update().update_
            ses[v] = np.sqrt(np.mean(update[v] ** 2))
        self.se_ = np.mean(ses)/np.sqrt(self.n_)
        return self

    ## calculate the ci based on the estimate and the standard error
    def get_ci(self, level = 0.95):
        ## get alpha from the level
        a = (1 - level)/2.
        a = np.array([a, 1 - a])
        ## calculate the quantiles
        fac = norm.ppf(a)
        ## set up the ci array
        ci = np.zeros((self.vimp_.shape[0], 2))
        ## create it
        self.ci_ = self.vimp_ + np.outer((self.se_), fac)
        return self
