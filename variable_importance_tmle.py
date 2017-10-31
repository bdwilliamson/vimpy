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
def variableImportance(full = None, reduced = None, y = None, x = None, s = None, lib = None, libnames = None, tol = 10e-16):
    import numpy as np
    import statsmodels.api as sm
    from supylearner import *
    from sklearn import 

    def expit(x):
        return np.exp(x)/(1. + np.exp(x))
    def logit(x):
        return np.log(x/(1. - x))

    ## calculate the covariate
    covar = full - reduced

    ## get initial estimate of epsilon
    eps_init = sm.GLM(endog = y, exog = covar, family = sm.family.Binomial(), offset = full).fit().params

    ## update
    new_f = expit(logit(full) + eps_init*covar)
    new_r = SuperLearner(lib, libnames, loss = "L2").fit(x[:,-s], new_f).predict(x[-s])

    ## now repeat until convergence
    if eps_init == 0:
        f = new_f
        r = new_r
        eps = eps_init
    else:
        f = new_f
        r = new_r
        eps = eps_init
        while abs(eps) > tol:
            ## get the covariate
            covar = f - r
            ## update epsilon
            eps = sm.GLM(endog = y, exog = covar, family = sm.family.Binomial(), offset = f).fit().params
            ## update fitted values
            f = expit(logit(f) + eps*covar)
            r = SuperLearner(lib, libnames, loss = "L2").fit(x[:,-s], f).predict(x[-s])
    ## variable importance
    est = np.mean((f - r)**2)/np.mean((y - np.mean(y))**2)
    ## return
    ret = {'est':est, 'full':f, 'reduced':r, 'eps':eps}
    return ret
  
  

