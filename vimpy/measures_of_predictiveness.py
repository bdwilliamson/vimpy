## measures of predictiveness


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


def r_squared(y, preds):
    """
    Compute R^2 for a given set of predictions and outcomes

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
    return np.dot(grad, np.transpose(np.hstack(ic_mse, ic_var)))


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


def deviance(y, preds):
    """
    Compute cross-entropy for a given set of predictions and outcomes

    @param y: the outcome
    @param preds: the predictions based on a subset of features

    @return the cross-entropy
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


def deviance_ic(y, preds):
    """
    Compute the IC for cross-entropy

    @param y: the outcome
    @param preds: the predictions based on a given subset of features

    @return the IC for cross-entropy
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
    Compute the IC for one cross-entropy

    @param y: the outcome
    @param preds: the predictions based on a given subset of features

    @return the IC for cross-entropy
    """
    import sklearn.metrics as skm
    import numpy as np
    cross_entropy = (-2) * skm.log_loss(y_true = y, y_pred = preds, normalize = True)
    p = np.mean(y, axis = 0)
    denom = (-1) * np.sum(np.log(p))
    ic_cross_entropy = (-2) * np.sum(y * np.log(preds), axis = 1) - cross_entropy
    ic_denom = ((-1.) / p) * ((y == 1) - p)
    grad = np.array([1. / denom, (-1) * cross_entropy / (denom ** 2)])
    return np.dot(grad, np.transpose(np.hstack(ic_cross_entropy, ic_denom)))
