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
##             s - the features to remove (MUST BE A LIST)
##           lib - the library for Super Learner
##      libnames - the names for Super Learner
##           tol - the tolerance level for convergence to zero
## RETURNS: the naive estimate and one-step estimate
# def variableImportanceTMLE(full = None, reduced = None, y = None, x = None, s = None, lib = None, libnames = None, tol = 10e-16):
def variableImportanceTMLE(full = None, reduced = None, y = None, x = None, s = None, grid = None, tol = 10e-16, V = 5, max_iter = 500):
    import numpy as np
    from variable_importance_se import variableImportanceSE
    from variable_importance_ci import variableImportanceCI
    from variable_importance_ic import variableImportanceIC
    import statsmodels.api as sm
    # from supylearner import *
    from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
    import pdb

    def expit(x):
        return np.exp(x)/(1. + np.exp(x))
    def logit(x):
        return np.log(x/(1. - x))

    ## calculate the covariate and offset
    covar = full.reshape(len(y),1) - reduced
    off = logit(full)

    ## get initial estimate of epsilon
    glm = sm.GLM(endog = y, exog = covar, family = sm.families.Binomial(), offset = off).fit()
    eps = glm.params

    ## update
    new_f = expit(logit(full) + np.dot(covar, eps))

    ## CV to get new reduced model for each s
    new_rs = np.zeros((len(y), len(s)))
    counter = 0
    for i in s:
        ## small data
        x_small = np.delete(x, i, 1)
        cv_r = GridSearchCV(GradientBoostingRegressor(loss = 'ls', max_depth = 1), param_grid = grid, cv = V)
        cv_r.fit(x_small, new_f)
        ntree_r = cv_r.best_params_['n_estimators']
        lr_r = cv_r.best_params_['learning_rate']
        ## fit reduced model
        small_mod = GradientBoostingRegressor(loss = 'ls', learning_rate = lr_r, max_depth = 1, n_estimators = ntree_r)
        small_mod.fit(x_small, new_f)
        ## get fitted values
        new_r = small_mod.predict(x_small)
        new_rs[:, counter] = new_r[:]
        counter = counter + 1

    ## debug location
    pdb.set_trace()
    ## now compute the empirical average
    avg = np.apply_along_axis(np.mean, 1, np.apply_along_axis(variableImportanceIC, 1, new_rs, new_f, y = y))
    ## now repeat until convergence
    avgs = np.zeros((len(s), max_iter))
    avgs[:,0] = avg
    if max(abs(avg)) < tol:
        f = new_f
        r = new_r
        avg = avg
    else:
        f = new_f
        r = new_r
        k = 0
        avgs[:, k] = avg[:]
        k = 1
        while (max(abs(avg)) > tol) & (k < max_iter):
            ## get the covariate
            covar = f - r
            off = logit(f)
            ## update epsilon
            glm = sm.GLM(endog = y, exog = covar, family = sm.families.Binomial(), offset = off).fit()
            eps = glm.params
            ## update fitted values
            f = expit(logit(f) + np.dot(covar, eps))
            ## CV to get new reduced model for each s
            new_rs = np.zeros((len(y), len(s)))
            counter = 0
            for i in s:
                ## small data
                x_small = np.delete(x, i, 1)
                cv_r = GridSearchCV(GradientBoostingRegressor(loss = 'ls', max_depth = 1), param_grid = grid, cv = V)
                cv_r.fit(x_small, f)
                ntree_r = cv_r.best_params_['n_estimators']
                lr_r = cv_r.best_params_['learning_rate']
                ## fit reduced model
                small_mod = GradientBoostingRegressor(loss = 'ls', learning_rate = lr_r, max_depth = 1, n_estimators = ntree_r)
                small_mod.fit(x_small, f)
                ## get fitted values
                new_r = small_mod.predict(x_small)
                new_rs[:, counter] = new_r[:]
                counter = counter + 1

            ## get the average
            avg = np.apply_along_axis(np.mean, 1, np.apply_along_axis(variableImportanceIC, 1, new_rs, f, y = y))
            k = k+1
            avgs[:,k] = avg
            

    ## variable importance
    est = np.array([np.mean((f - r)**2)/np.mean((y - np.mean(y))**2)])
    ## standard error
    se = variableImportanceSE(full = f, reduced = r, y = y, n = len(y), standardized = True)
    ## ci
    ci = variableImportanceCI(est = est, se = se, n = len(y), level = 0.95)
    ## return
    ret = {'est':est, 'se':se, 'ci':ci, 'full':f, 'reduced':r, 'eps':eps}
    return ret
  
  

