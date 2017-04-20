###########################################################################################################
##
## FILE:    variable_importance.py
##
## CREATED: 31 January 2017
##
## AUTHOR:  Brian Williamson
##
## PURPOSE: calculate naive and one-step estimators for both variable importance parameters, now for new POI
## UPDATES
## DDMMYY   INIT   COMMENTS
## ------   ----   --------
############################################################################################################

## relies on variableImportanceIC to calculate the influence curve

## FUNCTION: variableImportance
## ARGS:    full - the model fit to the full data
##       reduced - the model fit to the reduced data
##             y - the outcome
##             n - the sample size
##  standardized - whether or not to compute the standardized estimator
## RETURNS: the naive estimate and one-step estimate
def variableImportance(full = None, reduced = None, y = None, n = None, standardized = True):
    import variable_importance_ic as ic
    import numpy as np
    
    ## calculate naive
    if(standardized):
        naive = np.mean((full - reduced) ** 2)/np.mean((y - np.mean(y)) ** 2)
    else:
        naive = np.mean((full - reduced) ** 2)
    
    ## now add on mean of ic
    onestep = naive + np.mean(ic.variableImportanceIC(full, reduced, y, standardized))
    
    ## return as an array
    ret = np.array([naive, onestep])
    return ret
  
  

