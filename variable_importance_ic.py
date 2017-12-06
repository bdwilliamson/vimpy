#######################################################################################################
##
## FILE: variable_importance_ic.py
##
## CREATED: 31 January 2017 by Brian Williamson
##
## PURPOSE: Calculate the IC for the variable importance parameters, new POI
##
## INPUTS/OUTPUTS: None
##
## UPDATES:
## DDMMYY INIT COMMENTS
## ------ ---- --------
#######################################################################################################
## FUNCTION: variableImportanceIC
## ARGS:   full - model fit to the full data
##      reduced - model fit to the reduced data
##            y - outcome
## standardized - standardized or non-standardized parameter
##        logit - logit scale?
## RETURNS: influence curve fit to the data
def variableImportanceIC(full = None, reduced = None, y = None, standardized = True):
    import numpy as np
    import pdb
    pdb.set_trace()
    ## calculate naive estimates
    naive_j = np.mean((full - reduced) ** 2)
    naive_var = np.mean((y - np.mean(y)) ** 2)
    
    ## now calculate ic
    if(standardized):
        ret = (2*(y - full)*(full - reduced) + (full - reduced) ** 2 - naive_j)/naive_var - ((y - np.mean(y)) ** 2 - naive_var)*naive_j/(naive_var ** 2)
        
    else:
        ret = 2*(y - full)*(full - reduced) + (full - reduced) ** 2 - naive_j
    
    return ret