## Python class for cross-validated estimates of variable importance
## compute estimates and confidence intervals, do hypothesis testing

## import required libraries
import numpy as np
from scipy.stats import norm
from .predictiveness_measures import cv_predictiveness, cv_predictiveness_precomputed
from .vimpy_utils import get_measure_function


class cv_vim:

    ## define initialization values
    """
    @param y the outcome
    @param x the feature data
    @param s the feature group of interest
    @param measure_type the predictiveness measure to use (for now, one of "r_squared", "auc", "accuracy", "deviance")
    @param V the number of cross-fitting folds (defaults to 5)
    @param pred_func the function that predicts outcome given features
    @param ensemble is pred_func an ensemble (True) or a single function (False, default)
    @param f fitted values from regression of outcome on all features (only used if pred_func is not specified)
    @param r fitted values from regression of outcome on reduced set of features (only used if pred_func is not specified)
    @param folds a list of length 3: outer folds, for hypothesis testing; inner folds based on the outer folds == 1 (for cross-fitting); inner folds based on outer folds == 0 (for cross-fitting)
    @param na_rm remove NAs prior to computing predictiveness? (defaults to False)
    """
    def __init__(self, y, x, s, measure_type, V = 5, pred_func = None, ensemble = False, f = None, r = None, folds = None, na_rm = False):
        self.y_ = y
        self.x_ = x
        self.s_ = s
        self.n_ = y.shape[0]
        self.p_ = x.shape[1]
        self.pred_func_ = pred_func
        self.f_ = f
        self.r_ = r
        assert (pred_func is not None or (f is not None and r is not None))
        self.V_ = V
        self.measure_type_ = measure_type
        self.measure_ = get_measure_function(measure_type)
        self.vimp_ = []
        self.se_ = []
        self.ci_ = []
        self.hyp_test_ = []
        self.test_statistic_ = []
        self.p_value_ = []
        self.v_full_ = []
        self.preds_full_ = []
        self.v_redu_ = []
        self.preds_redu_ = []
        self.se_full_ = []
        self.se_redu_ = []
        self.ci_full_ = []
        self.ci_redu_ = []
        ## set up outer folds for hypothesis testing
        ## set up outer folds for hypothesis testing
        if folds is None:
            self.folds_outer_ = np.random.choice(a = np.arange(2), size = self.n_, replace = True, p = np.array([0.5, 0.5]))
            self.folds_inner_1 = []
            self.folds_inner_0 = []
        else:
            assert (f is not None)
            self.folds_outer_ = folds[0]
            self.folds_inner_1 = folds[1]
            self.folds_inner_0 = folds[2]
        self.ic_ = np.zeros((max(np.sum(self.folds_outer_ == 0), np.sum(self.folds_outer_ == 1))))
        self.ic_full_ = np.zeros((max(np.sum(self.folds_outer_ == 0), np.sum(self.folds_outer_ == 1))))
        self.ic_redu_ = np.zeros((max(np.sum(self.folds_outer_ == 0), np.sum(self.folds_outer_ == 1))))
        ## if only two unique values in y, assume binary
        self.binary_ = (np.unique(y).shape[0] == 2)
        self.na_rm_ = na_rm
        self.ensemble_ = ensemble

    ## calculate the plug-in estimator
    def get_point_est(self):
        if self.pred_func_ is not None:
            predictiveness_func = cv_predictiveness
            this_full_func = self.pred_func_
            this_redu_func = self.pred_func_
            folds_1 = None
            folds_0 = None
        else:
            predictiveness_func = cv_predictiveness_precomputed
            this_full_func = self.f_
            this_redu_func = self.r_
            folds_1 = self.folds_inner_1
            folds_0 = self.folds_inner_0
        self.v_full_, self.preds_full_, ic_full, self.folds_inner_1, self.cc_1 = predictiveness_func(self.x_[self.folds_outer_ == 1, :], self.y_[self.folds_outer_ == 1], np.arange(self.p_), self.measure_, this_full_func, V = self.V_, stratified = self.binary_, na_rm = self.na_rm_, folds = folds_1, ensemble = self.ensemble_)
        self.v_redu_, self.preds_redu_, ic_redu, self.folds_inner_0, self.cc_0 = predictiveness_func(self.x_[self.folds_outer_ == 0, :], self.y_[self.folds_outer_ == 0], np.delete(np.arange(self.p_), self.s_), self.measure_, this_redu_func, V = self.V_, stratified = self.binary_, na_rm = self.na_rm_, folds = folds_0, ensemble = self.ensemble_)
        self.vimp_ = self.v_full_ - self.v_redu_
        self.ic_full_[:ic_full.shape[0]] = ic_full
        self.ic_redu_[:ic_redu.shape[0]] = ic_redu
        return self

    ## calculate the influence function
    def get_influence_function(self):
        self.ic_ = self.ic_full_ - self.ic_redu_
        return self

    ## calculate the standard error
    def get_se(self):
        self.se_full_ = np.sqrt(np.mean(self.ic_full_ ** 2)) / np.sqrt(self.ic_full_.shape[0])
        self.se_redu_ = np.sqrt(np.mean(self.ic_redu_ ** 2)) / np.sqrt(self.ic_redu_.shape[0])
        self.se_ = np.sqrt(np.mean(self.ic_ ** 2)) / np.sqrt(self.ic_.shape[0])
        return self

    ## calculate the ci based on the estimate and the standard error
    def get_ci(self, level = 0.95):
        ## get alpha from the level
        a = (1 - level) / 2.
        a = np.array([a, 1 - a])
        ## calculate the quantiles
        fac = norm.ppf(a)
        ## create cis for vimp, predictiveness
        self.ci_ = self.vimp_ + np.outer((self.se_), fac)
        self.ci_full_ = self.v_full_ + np.outer((self.se_full_), fac)
        self.ci_redu_ = self.v_redu_ + np.outer((self.se_redu_), fac)
        return self

    ## do a hypothesis test
    def hypothesis_test(self, alpha = 0.05, delta = 0):
        self.test_statistic_ = (self.v_full_ - self.v_redu_ - delta) / np.sqrt(self.se_full_ ** 2 + self.se_redu_ ** 2)
        self.p_value_ = 1 - norm.cdf(self.test_statistic_)
        self.hyp_test_ = self.p_value_ < alpha
        return(self)
