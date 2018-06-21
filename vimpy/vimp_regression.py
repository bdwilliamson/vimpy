###############################################################################
##
## FILE:    vimp_regression.py
##
## CREATED: 15 June 2018
##
## AUTHOR:  Brian Williamson
##
## PURPOSE: Python class for variable importance based on squared error measure
## 
## PARAMETERS: y - the outcome Y
##             x - the data
##             f - the fitted values from the full regression
##             m - the fitted values from the reduced regression
##             s - the set of features for importance
################################################################################

## import required libraries
import numpy as np
from scipy.stats import norm

class vimp_regression:
    
    ## define initialization values
    """
    y is the outcome
    x is the feature data
    f is the fitted values from a regression of Y on X
    m is the fitted values from a regression of f (or Y) on X(-s)
    s is the set of features for importance
    """
    def __init__(self, y, x, f, m, s):
        self.y_ = y
        self.x_ = x
        self.n_ = y.shape[0]
        self.p_ = []
        self.f_ = f
        self.m_ = m
        self.s_ = s
        self.naive_ = []
        self.update_ = []
        self.vimp_ = []
        self.se_ = []
        self.ci_ = []

    ## calculate the plug-in estimator
    def plugin(self):
        numerator = np.mean((self.f_ - self.m_) ** 2)
        denominator = np.mean((self.y_ - np.mean(self.y_)) ** 2)
        self.naive_ = np.array([numerator/denominator])
        return(self)

    ## calculate the update
    def update(self):
        if len(self.m_.shape) > 1:
            naive_j = np.mean(np.square(self.f_ - self.m_), axis = 0).reshape(1, self.m_.shape[1])    
        else :
            naive_j = np.mean(np.square(self.f_ - self.m_))
        denominator = np.mean(np.square(self.y_ - np.mean(self.y_)))
        
        ## influence function of the numerator
        d_s = 2*np.multiply(self.y_ - self.f_, self.f_ - self.m_) + np.square(self.f_ - self.m_) - naive_j
        ## influence function of the denominator
        d_denom = np.square(self.y_ - np.mean(self.y_)) - denominator
        ## get the update
        self.update_ = d_s/denominator - naive_j/(denominator**2)*d_denom
        return self

    ## calculate the variable importance based on the one-step
    def onestep_based_estimator(self):
        self.vimp_ = self.naive_ + np.mean(self.update_)
        return self

    ## calculate the standard error based on the one-step correction
    def onestep_based_se(self):
        var = np.mean(self.update_ ** 2)
        self.se_ = np.sqrt(var/self.n_)
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