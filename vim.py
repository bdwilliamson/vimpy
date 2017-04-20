#######################################################################################################
##
## FILE: vim.py
##
## CREATED: 4 April 2017 by Brian Williamson
##
## PURPOSE: Calculate variable importance given j, fitted values
##
## INPUTS/OUTPUTS: None
##
## UPDATES:
## DDMMYY INIT COMMENTS
## ------ ---- --------
## 200417 BDW  Added actual code
#######################################################################################################

## FUNCTION: vim
## ARGS: full - the model fit to the full data
##       reduced - the model fit to the reduced data
##             y - the outcome
##             n - the sample size
##  standardized - whether or not to compute the standardized estimator
##         level - confidence level (default 0.95) for the CI
## RETURNS: the one-step estimate, se, and CI for one-step
def vim(full = None, reduced = None, y = None, n = None, standardized = True, level = 0.95):
    ## import required functions and packages
    from variable_importance import variableImportance
    from variable_importance_se import variableImportanceSE
    from variable_importance_ci import variableImportanceCI
    import numpy as np
    
    ## get variable importance estimate
    est = variableImportance(full, reduced, y, n, standardized)[1]
    se = variableImportanceSE(full, reduced, y, n, standardized)
    ci = variableImportanceCI(est, se, n, level)
    
    ## return a dictionary with elements
    ret = {'onestep':est, 'se':se, 'ci':ci}
    return ret