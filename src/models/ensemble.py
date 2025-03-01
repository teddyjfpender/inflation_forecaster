import numpy as np

def ensemble_predictions(preds_list, errors_list):
    """
    Combine model predictions using a weighted average where weights are the inverse of the error.

    :param preds_list: List of numpy arrays of predictions.
    :param errors_list: List of error metrics (e.g. RMSE) for each model.
    :return: (ensemble prediction, weights used)
    """
    errors = np.array(errors_list)
    weights = 1 / errors
    weights = weights / np.sum(weights)
    ensemble_pred = np.sum([w * preds for w, preds in zip(weights, preds_list)], axis=0)
    return ensemble_pred, weights
