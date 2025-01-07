import numpy as np
from sklearn.metrics import make_scorer

def brier_score(y_true, y_pred, threshold_range=None):
    """
    A custom metric function.
    
    Parameters:
    - y_true: array-like of shape (n_samples,)
      Ground truth (correct) labels.
    - y_pred: array-like of shape (n_samples,)
      Predicted labels, as returned by a classifier.
    - **kwargs: Additional parameters.

    Returns:
    - score: float
      The computed metric.
    """
    if threshold_range is not None:
      y_pred = np.clip(y_pred, threshold_range[0], threshold_range[1])
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

brier_score_scorer = make_scorer(brier_score, greater_is_better=True)