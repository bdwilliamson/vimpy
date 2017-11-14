###########################################################################################################
##
## FILE:    variable_importance_tmle.py
##
## CREATED: 27 October 2017
##
## AUTHOR:  Brian Williamson
##
## PURPOSE: calculate TMLE for the parameter of interest in variable importance
## UPDATES
## DDMMYY   INIT   COMMENTS
## ------   ----   --------
############################################################################################################

## relies on variableImportanceIC to calculate the influence curve

## FUNCTION: variableImportance
## ARGS:    full - the model fit to the full data
##       reduced - the model fit to the reduced data
##             y - the outcome
##             x - the covariates
##             s - the features to remove
##           lib - the library for Super Learner
##      libnames - the names for Super Learner
##           tol - the tolerance level for convergence to zero
## RETURNS: the naive estimate and one-step estimate
# def variableImportanceTMLE(full = None, reduced = None, y = None, x = None, s = None, lib = None, libnames = None, tol = 10e-16):
def variableImportanceTMLE(full = None, reduced = None, y = None, x = None, s = None, grid = None, tol = 10e-16, V = 5, max_iter = 500):
    import numpy as np
    from variable_importance_se import variableImportanceSE
    from variable_importance_ci import variableImportanceCI
    import statsmodels.api as sm
    # from supylearner import *
    from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

    def expit(x):
        return np.exp(x)/(1. + np.exp(x))
    def logit(x):
        return np.log(x/(1. - x))

    ## set up small x
    x_small = np.delete(x, s, 1)
    ## calculate the covariate
    covar = full - reduced
    off = logit(full)

    ## get initial estimate of epsilon
    eps_init = sm.GLM(endog = y, exog = covar, family = sm.families.Binomial(), offset = off).fit().params

    ## update
    new_f = expit(logit(full) + eps_init*covar)
    # new_r = SuperLearner(lib, libnames, loss = "L2").fit(x[:,-s], new_f).predict(x[-s])
    ## CV to get new reduced model
    cv_r = GridSearchCV(GradientBoostingRegressor(loss = 'ls', max_depth = 1), param_grid = grid, cv = V)
    cv_r.fit(x_small, new_f)
    ntree_r = cv_r.best_params_['n_estimators']
    lr_r = cv_r.best_params_['learning_rate']
    ## fit reduced model
    small_mod = GradientBoostingRegressor(loss = 'ls', learning_rate = lr_r, max_depth = 1, n_estimators = ntree_r)
    small_mod.fit(x_small, new_f)
    ## get fitted values
    new_r = small_mod.predict(x_small)

    ## now repeat until convergence
    epss = np.zeros((1,max_iter))
    epss[0] = eps_init
    if eps_init == 0:
        f = new_f
        r = new_r
        eps = eps_init
    else:
        f = new_f
        r = new_r
        eps = eps_init
        k = 1
        while abs(eps) > tol:
            ## get the covariate
            covar = f - r
            off = logit(f)
            ## update epsilon
            eps = sm.GLM(endog = y, exog = covar, family = sm.families.Binomial(), offset = off).fit().params
            ## update fitted values
            f = expit(logit(f) + eps*covar)
            # r = SuperLearner(lib, libnames, loss = "L2").fit(x[:,-s], f).predict(x[-s])
            cv_r = GridSearchCV(GradientBoostingRegressor(loss = 'ls', max_depth = 1), param_grid = grid, cv = V)
            cv_r.fit(x_small, f)
            ntree_r = cv_r.best_params_['n_estimators']
            lr_r = cv_r.best_params_['learning_rate']
            small_mod = GradientBoostingRegressor(loss = 'ls', learning_rate = lr_r, max_depth = 1, n_estimators = ntree_r)
            small_mod.fit(x_small, f)
            r = small_mod.predict(x_small)
            epss[k] = eps
            k = k+1

    ## variable importance
    est = np.mean((f - r)**2)/np.mean((y - np.mean(y))**2)
    ## standard error
    se = variableImportanceSE(full = f, reduced = r, y = y, n = len(y), standardized = True)
    ## ci
    ci = variableImportanceCI(est = est, se = se, n = len(y), level = 0.95)
    ## return
    ret = {'est':est, 'se':se, 'ci':ci, 'full':f, 'reduced':r, 'eps':eps}
    return ret
  
  

