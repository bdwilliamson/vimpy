## Python class for estimates of Shapley population variable importance
## compute estimates and confidence intervals, do hypothesis testing

## import required libraries
import numpy as np
from scipy.stats import norm
from .predictiveness_measures import cv_predictiveness, compute_ic
from .spvim_ic import shapley_influence_function, shapley_se
from .vimpy_utils import get_measure_function, choose, shapley_hyp_test


class spvim:

    ## define initialization values
    """
    @param y the outcome
    @param x the feature data
    @param measure_type the predictiveness measure to use (a function)
    @param V the number of cross-validation folds
    @param pred_func the function that predicts outcome given features
    @param ensemble is pred_func an ensemble (True) or a single function (False, default)
    @param na_rm remove NAs prior to computing predictiveness? (defaults to False)
    """
    def __init__(self, y, x, measure_type, V, pred_func, ensemble = False, na_rm = False):
        self.y_ = y
        self.x_ = x
        self.n_ = y.shape[0]
        self.p_ = x.shape[0]
        self.pred_func_ = pred_func
        self.V_ = V
        self.measure_type_ = measure_type
        self.measure_ = get_measure_function(measure_type)
        self.ics_ = []
        self.vimp_ = []
        self.lambdas_ = []
        self.ses_ = []
        self.cis_ = []
        self.na_rm_ = na_rm
        self.Z_ = []
        self.v_ = []
        self.v_ics_ = []
        self.W_ = []
        self.gamma_ = []
        self.test_statistics_ = []
        self.p_values_ = []
        self.hyp_tests_ = []
        self.G_ = np.vstack((np.append(1, np.zeros(self.p_)), np.ones(self.p_ + 1) - np.append(1, np.zeros(self.p_))))
        ## set up outer folds for hypothesis testing
        self.folds_outer_ = np.random.choice(a = np.arange(2), size = self.n_, replace = True, p = np.array([0.25, 0.75]))
        self.folds_inner_ = []
        ## if only two unique values in y, assume binary
        self.binary_ = (np.unique(y).shape[0] == 2)
        self.ensemble_ = ensemble
        self.cc_ = []

    def _get_kkt_matrix(self):
        # kkt matrix for constrained wls
        A_W = np.sqrt(self.W_).dot(self.Z_)
        kkt_matrix_11 = 2 * A_W.transpose().dot(A_W)
        kkt_matrix_12 = self.G_.transpose()
        kkt_matrix_21 = self.G_
        kkt_matrix_22 = np.zeros((kkt_matrix_21.shape[0], kkt_matrix_12.shape[1]))
        kkt_matrix = np.vstack((np.hstack((kkt_matrix_11, kkt_matrix_12)), np.hstack((kkt_matrix_21, kkt_matrix_22))))
        return(kkt_matrix)

    def _get_ls_matrix(self, c_n):
        A_W = np.sqrt(self.W_).dot(self.Z_)
        v_W = np.sqrt(self.W_).dot(self.v_)
        ls_matrix = np.vstack((2 * A_W.transpose().dot(v_W.reshape((len(v_W), 1))), c_n.reshape((c_n.shape[0], 1))))
        return(ls_matrix)

    ## calculate the point estimates
    def get_point_est(self, gamma = 1):
        self.gamma_ = gamma
        ## sample subsets, set up Z
        max_subset = np.array(list(range(self.p_)))
        sampling_weights = np.append(np.append(1, [choose(self.p_ - 2, s - 1) ** (-1) for s in range(1, self.p_)]), 1)
        subset_sizes = np.random.choice(np.arange(0, self.p_ + 1), p = sampling_weights / sum(sampling_weights), size = self.gamma_ * self.x_.shape[0], replace = True)
        S_lst_all = [np.sort(np.random.choice(np.arange(0, self.p_), subset_size, replace = False)) for subset_size in list(subset_sizes)]
        ## only need to continue with the unique subsets S
        Z_lst_all = [np.in1d(max_subset, S).astype(np.float64) for S in S_lst_all]
        Z, z_counts = np.unique(np.array(Z_lst_all), axis = 0, return_counts = True)
        Z_lst = list(Z)
        Z_aug_lst = [np.append(1, z) for z in Z_lst]
        S_lst = [max_subset[z.astype(bool).tolist()] for z in Z_lst]
        ## get v, preds, ic for null set
        preds_none = np.repeat(np.mean(self.y_[self.folds_outer_ == 1]), np.sum(self.folds_outer_ == 1))
        v_none = self.measure_(self.y_[self.folds_outer_ == 1], preds_none)
        ic_none = compute_ic(self.y_[self.folds_outer_ == 1], preds_none, self.measure_.__name__)
        ## get v, preds, ic for remaining non-null groups in S
        v_lst, preds_lst, ic_lst, self.folds_inner_, self.cc_ = zip(*(cv_predictiveness(self.x_[self.folds_outer_ == 1, :], self.y_[self.folds_outer_ == 1], s, self.measure_, self.pred_func_, V = self.V_, stratified = self.binary_, na_rm = self.na_rm_) for s in S_lst[1:]))
        ## set up full lists
        v_lst_all = [v_none] + list(v_lst)
        ic_lst_all = [ic_none] + list(ic_lst)
        self.Z_ = np.array(Z_aug_lst)
        self.W_ = np.diag(z_counts / np.sum(z_counts))
        self.v_ = np.array(v_lst_all)
        self.v_ics_ = ic_lst_all
        c_n = np.array([v_none, v_lst_all[len(v_lst)] - v_none])
        kkt_matrix = self._get_kkt_matrix()
        ls_matrix = self._get_ls_matrix(c_n)
        ls_solution = np.linalg.inv(kkt_matrix).dot(ls_matrix)
        self.vimp_ = ls_solution[0:(self.p_ + 1), :]
        self.lambdas_ = ls_solution[(self.p_ + 1):ls_solution.shape[0], :]
        return(self)

    ## calculate the influence function
    def get_influence_functions(self):
        c_n = np.array([self.v_[0], self.v_[self.v.shape[0]] - self.v_[0]])
        self.ics_ = shapley_influence_function(self.Z_, self.z_counts_, self.W_, self.v_, self.vimp_, self.G_, c_n, np.array(self.v_ics_), self.measure_.__name__)
        return self

    ## calculate standard errors
    def get_ses(self):
        ses = [shapley_se(self.ics_, idx, self.gamma_) for idx in range(self.p_)]
        self.ses_ = np.array(ses)
        return self

    ## calculate the ci based on the estimate and the standard error
    def get_cis(self, level = 0.95):
        ## get alpha from the level
        a = (1 - level) / 2.
        a = np.array([a, 1 - a])
        ## calculate the quantiles
        fac = norm.ppf(a)
        ## create it
        self.ci_ = self.vimp_ + np.outer((self.ses_), fac)
        return self

    ## do a hypothesis test
    def hypothesis_test(self, alpha = 0.05, delta = 0):
        ## null predictiveness
        preds_none_0 = np.repeat(np.mean(self.y_[self.folds_outer_ == 0]), np.sum(self.folds_outer_ == 0))
        v_none_0 = self.measure_(self.y_[self.folds_outer_ == 0], preds_none_0)
        ic_none_0 = compute_ic(self.y_[self.folds_outer_ == 0], preds_none_0, self.measure_.__name__)
        sigma_none_0 = np.sqrt(np.mean((ic_none_0) ** 2)) / np.sqrt(np.sum(self.folds_outer_ == 0))
        ## get shapley values + null predictiveness on first split
        shapley_vals_plus = self.vimp_ + self.vimp_[0]
        sigmas_one = np.sqrt(self.ses_ ** 2 + sigma_none_0 ** 2)
        self.test_statistics_, self.p_values_, self.hyp_tests_ = shapley_hyp_test(shapley_vals_plus[1:], v_none_0, sigmas_one, sigma_none_0, level = alpha, delta = delta, p = self.p_)
        return self
