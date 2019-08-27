## import required libraries
import numpy as np
from scipy.stats import norm
import measures_of_predictiveness as mp


class vimp:

    ## define initialization values
    """
    @param y the outcome
    @param x the feature data
    @param f the fitted values from a regression of Y on X
    @param m the fitted values from a regression of f (or Y) on X(-s)
    @param s the set of features for importance
    @param type the type of importance (for now, "r_squared", "deviance", "accuracy", or "auc")

    @return an object of class vimp
    """
    def __init__(self, y, x, f, m, s, type):
        self.y_ = y
        self.x_ = x
        self.n_ = y.shape[0]
        self.p_ = []
        self.f_ = f
        self.m_ = m
        self.s_ = s
        self.type_ = type
        self.update_ = []
        self.vimp_ = []
        self.se_ = []
        self.ci_ = []
        self.full_predictiveness_ = []
        self.redu_predictiveness_ = []
        self.full_predictiveness_ic_ = []
        self.redu_predictiveness_ic_ = []
        self.full_predictiveness_ci_ = []
        self.redu_predictiveness_ci_ = []

    ## get the measure function, IC function
    def get_measure_function(self):
        if self.type_ == "r_squared":
            measure = mp.r_squared
        elif self.type_ == "accuracy":
            measure = mp.accuracy
        elif self.type_ == "auc":
            measure = mp.auc
        else:
            measure = mp.deviance
        return measure

    def get_measure_ic(self):
        if self.type_ == "r_squared":
            measure = mp.r_squared_ic
        elif self.type_ == "accuracy":
            measure = mp.accuracy_ic
        elif self.type_ == "auc":
            measure = mp.auc_ic
        else:
            measure = mp.deviance_ic
        return measure

    ## calculate the variable importance estimate
    def get_vimp(self):
        measure = self.get_measure_function()
        ## calculate predictiveness
        self.full_predictiveness_ = measure(self.y_, self.f_)
        self.redu_predictiveness_ = measure(self.y_, self.m_)
        self.vimp_ = self.full_predictiveness_ - self.redu_predictiveness_
        return self

    ## calculate the ci based on the estimate and the
    ## influence curve-based standard error
    def get_ci(self, level = 0.95):
        ## compute the standard error
        ic = self.get_measure_ic()
        self.full_predictiveness_ic_ = ic(self.y_, self.f_)
        self.redu_predictiveness_ic_ = ic(self.y_, self.m_)
        var = np.mean((self.full_predictiveness_ic_ - self.redu_predictiveness_ic_) ** 2)
        self.se_ = np.sqrt(var / self.n_)
        ## get alpha from the level
        a = (1 - level) / 2.
        a = np.array([a, 1 - a])
        ## calculate the quantiles
        fac = norm.ppf(a)
        ## create it
        self.ci_ = self.vimp_ + np.outer((self.se_), fac)
        return self
