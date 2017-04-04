#######################################################################################################
##
## FILE: variable_importance_se.py
##
## CREATED: 31 January 2017 by Brian Williamson
##
## PURPOSE: Calculate the SE for the one-step, with new POI
##
## INPUTS/OUTPUTS: None
##
## UPDATES:
## DDMMYY INIT COMMENTS
## ------ ---- --------
#######################################################################################################

## relies on variableImportanceIC to calculate the estimate
## FUNCTION: variableImportanceSE
## ARGS:   full - the model fit to the full data
##      reduced - the model fit to the reduced data
##            y - the outcome
##            n - the sample size
## standardized - SE for the standardized or non-standardized parameter?
## RETURNS: the SE of the one-step
def variableImportanceSE(full = None, reduced = None, y = None, n = None, standardized = True):
    import variable_importance_ic as ic
    import numpy as np
    ## calculate the IC
    grad = ic.variableImportanceIC(full, reduced, y, standardized)
    
    ## calculate the variance
    var = np.mean(grad ** 2)
    
    ## calculate and return the SE
    se = np.sqrt(var/n)
    return se
