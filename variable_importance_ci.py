#######################################################################################################
##
## FILE: variable_importance_ci.py
##
## CREATED: 20 October 2016 by Brian Williamson
##
## PURPOSE: Calculate the CI for the one-step
##
## INPUTS/OUTPUTS: None
##
## UPDATES:
## DDMMYY INIT COMMENTS
## ------ ---- --------
## 111116 BDW  Updated that we calculate SE correctly now
#######################################################################################################

## FUNCTION: variableImportanceCI
## ARGS: est - the estimate
##        se - the standard error
##         n - the sample size
##     level - the level of the CI
def variableImportanceCI(est = None, se = None, n = None, level = 0.95):
    import numpy as np
    import scipy as sp
    from scipy.stats import norm
    
    ## get alpha
    a = (1 - level)/2
    a = np.array([a, 1 - a])
    ## calculate the quantiles
    fac = norm.ppf(a)
    ## set up the ci array
    ci = np.zeros((est.shape[0], 2))
    ## create it
    # ci = est + np.outer((se/np.sqrt(n)), fac)
    ci = est + np.outer((se), fac)
    return ci[0]