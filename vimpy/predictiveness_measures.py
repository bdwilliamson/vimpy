## Compute predictiveness measures and their corresponding influence functions


# general cv predictiveness
def cv_predictiveness(x, y, S, measure, pred_func, V = 5, stratified = True, na_rm = False, folds = None, ensemble = False):
    """
    Compute a cross-validated measure of predictiveness based on the data and the chosen measure

    @param x: the features
    @param y: the outcome
    @param S: the covariates to fit
    @param measure: measure of predictiveness
    @param pred_func: function that fits to the data
    @param V: the number of CV folds
    @param stratified: should the folds be stratified?
    @param na_rm: should we do a complete-case analysis (True) or not (False)
    @param folds (dummy)
    @param ensemble is this an ensemble (True) or not (False)

    @return cross-validated measure of predictiveness, along with preds and ics
    """
    import numpy as np
    from .vimpy_utils import make_folds
    ## if na_rm = True, do a complete-case analysis
    if na_rm:
        xs = x[:, S]
        cc = np.sum(np.isnan(xs), axis = 1) == 0
        newx = x[cc, :]
        newy = y[cc]
    else:
        cc = np.repeat(True, x.shape[0])
        newx = x
        newy = y
    ## set up CV folds
    folds = make_folds(newy, V, stratified = stratified)
    ## do CV
    preds = np.empty((y.shape[0],))
    preds.fill(np.nan)
    ics = np.empty((y.shape[0],))
    ics.fill(np.nan)
    vs = np.empty((V,))
    cc_cond = np.flatnonzero(cc)
    if V == 1:
        x_train, y_train = newx, newy
        pred_func.fit(x_train[:, S], np.ravel(y_train))
        if ensemble:
            preds_v = np.mean(pred_func.transform(x_train[:, S]))
        else:
            if measure.__name__ in ["r_squared"]:
                preds_v = pred_func.predict(x_train[:, S])
            else:
                preds_v = pred_func.predict_proba(x_train[:, S])[:, 1]
        preds[cc_cond] = preds_v
        vs[0] = measure(y_train, preds_v)
        ics[cc_cond] = compute_ic(y_train, preds_v, measure.__name__)
    else:
        for v in range(V):
            fold_cond = np.flatnonzero(folds == v)
            x_train, y_train = newx[folds != v, :], newy[folds != v]
            x_test, y_test = newx[folds == v, :], newy[folds == v]
            pred_func.fit(x_train[:, S], np.ravel(y_train))
            if ensemble:
                preds_v = np.mean(pred_func.transform(x_test[:, S]))
            else:
                if measure.__name__ in ["r_squared"]:
                    preds_v = pred_func.predict(x_test[:, S])
                else:
                    preds_v = pred_func.predict_proba(x_test[:, S])[:, 1]
            preds[cc_cond[fold_cond]] = preds_v
            vs[v] = measure(y_test, preds_v)
            ics[cc_cond[fold_cond]] = compute_ic(y_test, preds_v, measure.__name__)
    return np.mean(vs), preds, ics, folds, cc


# general predictiveness based on precomputed fits
def cv_predictiveness_precomputed(x, y, S, measure, f, V = 5, stratified = True, folds = None, na_rm = False, ensemble = False):
    """
    Compute a cross-validated measure of predictiveness based on the data, the chosen measure, and the sets of fitted values f and r

    @param x: the features
    @param y: the outcome
    @param S: the covariates to fit
    @param measure: measure of predictiveness
    @param f: fitted values based on S
    @param V: the number of CV folds
    @param stratified: should the folds be stratified?
    @param folds: the CV folds
    @param na_rm: should we do a complete-case analysis (True) or not (False)
    @param ensemble: is this an ensemble or not (dummy)

    @return cross-validated measure of predictiveness, along with preds and ics
    """
    import numpy as np
    from .vimpy_utils import make_folds
    ## if na_rm = True, do a complete-case analysis
    if na_rm:
        xs = x[:, S]
        cc = np.sum(np.isnan(xs), axis = 1) == 0
        newy = y[cc]
    else:
        cc = np.repeat(True, x.shape[0])
        newy = y
    ## set up CV folds
    if folds is None:
        folds = make_folds(newy, V, stratified = stratified)
    ## do CV
    preds = np.empty((y.shape[0],))
    preds.fill(np.nan)
    ics = np.empty((y.shape[0],))
    ics.fill(np.nan)
    vs = np.empty((V,))
    cc_cond = np.flatnonzero(cc)
    if V == 1:
        y_train = newy
        preds_v = f
        preds[cc_cond] = preds_v[cc_cond]
        vs[0] = measure(y_train, preds_v)
        ics[cc_cond] = compute_ic(y_train, preds_v, measure.__name__)
    else:
        for v in range(V):
            fold_cond = np.flatnonzero(folds == v)
            y_test = newy[folds == v]
            preds_v = f[folds == v]
            preds[cc_cond[fold_cond]] = preds_v
            vs[v] = measure(y_test, preds_v)
            ics[cc_cond[fold_cond]] = compute_ic(y_test, preds_v, measure.__name__)
    return np.mean(vs), preds, ics, folds, cc


def accuracy(y, preds):
    """
    Compute accuracy for a given set of predictions and outcomes

    @param y: the outcome
    @param preds: the predictions based on a subset of features

    @return the accuracy
    """
    import sklearn.metrics as skm

    if len(preds.shape) == 2:
        if preds.shape[1] > 1:
            return [1. - skm.zero_one_loss(y_true = y, y_pred = preds[:, i], normalize = True) for i in range(preds.shape[1])]
        else:
            return 1. - skm.zero_one_loss(y_true = y, y_pred = preds, normalize = True)
    else:
        return 1. - skm.zero_one_loss(y_true = y, y_pred = preds, normalize = True)


def auc(y, preds, *args, **kwargs):
    """
    Compute AUC for a given set of predictions and outcomes

    @param y: the outcome
    @param preds: the predictions based on a given subset of features

    @return the AUC
    """
    import sklearn.metrics as skm

    if len(preds.shape) == 2:
        if preds.shape[1] > 1:
            return [skm.roc_auc_score(y_true = y, y_score = preds[:, i], average = "micro") for i in range(preds.shape[1])]
        else:
            return skm.roc_auc_score(y_true = y, y_score = preds, average = "micro")
    else:
        return skm.roc_auc_score(y_true = y, y_score = preds, average = "micro")


def cross_entropy(y, preds):
    """
    Compute cross-entropy for a given set of predictions and outcomes

    @param y: the outcome
    @param preds: the predictions based on a subset of features

    @return the cross-entropy
    """
    import sklearn.metrics as skm

    if len(preds.shape) == 2:
        if preds.shape[1] > 1:
            return [(-2) * skm.log_loss(y_true = y, y_pred = preds[:, i], normalize = True) for i in range(preds.shape[1])]
        else:
            return (-2) * skm.log_loss(y_true = y, y_pred = preds, normalize = True)
    else:
        return (-2) * skm.log_loss(y_true = y, y_pred = preds, normalize = True)


def deviance(y, preds):
    """
    Compute deviance for a given set of predictions and outcomes

    @param y: the outcome
    @param preds: the predictions based on a subset of features

    @return the deviance
    """
    import sklearn.metrics as skm
    import numpy as np
    denom = (-1) * np.sum(np.log(np.mean(y, axis = 0)))

    if len(preds.shape) == 2:
        if preds.shape[1] > 1:
            return [(-2) * skm.log_loss(y_true = y, y_pred = preds[:, i], normalize = True) / denom for i in range(preds.shape[1])]
        else:
            return (-2) * skm.log_loss(y_true = y, y_pred = preds, normalize = True) / denom
    else:
        return (-2) * skm.log_loss(y_true = y, y_pred = preds, normalize = True) / denom


def r_squared(y, preds):
    """
    Compute R^s for a given set of predictions and outcomes

    @param y: the outcome
    @param preds: the predictions based on a given subset of features

    @return the R^2
    """
    import sklearn.metrics as skm
    import numpy as np
    var = np.mean((y - np.mean(y)) ** 2)

    if len(preds.shape) == 2:
        if preds.shape[1] > 1:
            return [1. - skm.mean_squared_error(y_true = y, y_pred = preds[:, i]) / var for i in range(preds.shape[1])]
        else:
            return 1. - skm.mean_squared_error(y_true = y, y_pred = preds) / var
    else:
        return 1. - skm.mean_squared_error(y_true = y, y_pred = preds) / var


## ------------------------------------------------------------------
## influence functions
## ------------------------------------------------------------------
def compute_ic(y, preds, measure):
    """
    Compute IC based on the given measure

    @param y: the outcome
    @param preds: the predictions based on the current subset of features
    @param measure: the predictiveness measure

    @return an n-vector of the IC for the given predictiveness measure
    """

    ## do the correct thing
    if measure == "accuracy":
        return accuracy_ic(y, preds)
    elif measure == "auc":
        return auc_ic(y, preds)
    elif measure == "cross_entropy":
        return cross_entropy_ic(y, preds)
    elif measure == "deviance":
        return deviance_ic(y, preds)
    elif measure == "r_squared":
        return r_squared_ic(y, preds)
    else:
        raise ValueError("We do not currently support the entered predictiveness measure. Please provide a different predictiveness measure.")


def accuracy_ic(y, preds):
    """
    Compute the IC for accuracy

    @param y: the outcome
    @param preds: the predictions based on a given subset of features

    @return the IC for accuracy
    """
    import numpy as np
    if len(preds.shape) == 2:
        if preds.shape[1] > 1:
            return np.array([one_accuracy_ic(y, preds[:, m]) for m in range(preds.shape[1])])
        else:
            return np.array([one_accuracy_ic(y, preds)])
    else:
        return np.array([one_accuracy_ic(y, preds)])


def one_accuracy_ic(y, preds):
    """
    Compute the IC for one accuracy

    @param y: the outcome
    @param preds: the predictions based on a given subset of features

    @return the IC for accuracy
    """
    import sklearn.metrics as skm

    misclassification = skm.zero_one_loss(y_true = y, y_pred = preds, normalize = True)
    return (-1) * (((preds > 1. / 2) != y) - misclassification)


def auc_ic(y, preds):
    """
    Compute the IC for AUC

    @param y: the outcome
    @param preds: the predictions based on a given subset of features

    @return the IC for AUC
    """
    import numpy as np
    if len(preds.shape) == 2:
        if preds.shape[1] > 1:
            return np.array([one_auc_ic(y, preds[:, m]) for m in range(preds.shape[1])])
        else:
            return np.array([one_auc_ic(y, preds)])
    else:
        return np.array([one_auc_ic(y, preds)])


def one_auc_ic(y, preds):
    """
    Compute the IC for one AUC

    @param y: the outcome
    @param preds: the predictions based on a given subset of features

    @return the IC for AUC
    """
    import numpy as np
    import sklearn.metrics as skm

    p_1 = np.mean(y)
    p_0 = 1 - p_1

    sens = np.array([np.mean(preds[(y == 0).reshape(preds.shape)] < x) for x in preds])
    spec = np.array([np.mean(preds[(y == 1).reshape(preds.shape)] > x) for x in preds])

    contrib_1 = (y == 1).reshape(preds.shape) / p_1 * sens
    contrib_0 = (y == 0).reshape(preds.shape) / p_0 * spec

    auc = skm.roc_auc_score(y_true = y, y_score = preds, average = "micro")
    return contrib_1 + contrib_0 - ((y == 0).reshape(preds.shape) / p_0 + (y == 1).reshape(preds.shape) / p_1) * auc


def cross_entropy_ic(y, preds):
    """
    Compute the IC for cross-entropy

    @param y: the outcome
    @param preds: the predictions based on a given subset of features

    @return the IC for cross-entropy
    """
    import numpy as np
    if len(preds.shape) == 2:
        if preds.shape[1] > 1:
            return np.array([one_cross_entropy_ic(y, preds[:, m]) for m in range(preds.shape[1])])
        else:
            return np.array([one_cross_entropy_ic(y, preds)])
    else:
        return np.array([one_cross_entropy_ic(y, preds)])


def one_cross_entropy_ic(y, preds):
    """
    Compute the IC for one cross-entropy

    @param y: the outcome
    @param preds: the predictions based on a given subset of features

    @return the IC for cross-entropy
    """
    import sklearn.metrics as skm
    import numpy as np
    cross_entropy = (-2) * skm.log_loss(y_true = y, y_pred = preds, normalize = True)
    ic_cross_entropy = (-2) * np.sum(y * np.log(preds), axis = 1) - cross_entropy
    return ic_cross_entropy


def deviance_ic(y, preds):
    """
    Compute the IC for deviance

    @param y: the outcome
    @param preds: the predictions based on a given subset of features

    @return the IC for deviance
    """
    import numpy as np
    if len(preds.shape) == 2:
        if preds.shape[1] > 1:
            return np.array([one_deviance_ic(y, preds[:, m]) for m in range(preds.shape[1])])
        else:
            return np.array([one_deviance_ic(y, preds)])
    else:
        return np.array([one_deviance_ic(y, preds)])


def one_deviance_ic(y, preds):
    """
    Compute the IC for one deviance

    @param y: the outcome
    @param preds: the predictions based on a given subset of features

    @return the IC for deviance
    """
    import sklearn.metrics as skm
    import numpy as np
    cross_entropy = (-2) * skm.log_loss(y_true = y, y_pred = preds, normalize = True)
    p = np.mean(y, axis = 0)
    denom = (-1) * np.sum(np.log(p))
    ic_cross_entropy = (-2) * np.sum(y * np.log(preds), axis = 1) - cross_entropy
    ic_denom = ((-1.) / p) * ((y == 1) - p)
    grad = np.array([1. / denom, (-1) * cross_entropy / (denom ** 2)])
    return np.dot(grad, np.stack((ic_cross_entropy, ic_denom)))


def r_squared_ic(y, preds):
    """
    Compute the IC for R-squared

    @param y: the outcome
    @param preds: the predictions based on a given subset of features

    @return the IC for R-squared
    """
    import numpy as np
    if len(preds.shape) == 2:
        if preds.shape[1] > 1:
            return np.array([one_r2_ic(y, preds[:, m]) for m in range(preds.shape[1])])
        else:
            return np.array([one_r2_ic(y, preds)])
    else:
        return np.array([one_r2_ic(y, preds)])


def one_r2_ic(y, preds):
    """
    Compute the IC for one R-squared

    @param y: the outcome
    @param preds: the predictions based on a given subset of features

    @return the IC for R-squared
    """
    import sklearn.metrics as skm
    import numpy as np
    mse = skm.mean_squared_error(y_true = y, y_pred = preds)
    var = np.mean((y - np.mean(y)) ** 2)
    ic_mse = (y.reshape(preds.shape) - preds) ** 2 - mse
    ic_var = (y - np.mean(y)) ** 2 - var
    grad = np.array([1. / var, (-1) * mse / (var ** 2)])
    return np.dot(grad, np.stack((ic_mse, ic_var)))
