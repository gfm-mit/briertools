import numpy as np
from sklearn.metrics import make_scorer
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

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
    #nope!  we should instead clip the prediction towards the right answer
    #calculate the whole brier for that
    #and subtract that off
    #because we want only the integral of the part within the window
    score = np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
    if threshold_range is not None:
      y_bound = np.array(threshold_range)[y_true]
      baseline = np.mean((np.array(y_true) - np.array(y_bound)) ** 2)
      return score - baseline
    return score

def brier_curve(y_true, y_pred, label=None, threshold_range=None):
    """
    Calculates the Brier score for different thresholds.

    Parameters:
    - y_true: array-like of shape (n_samples,)
      Ground truth (correct) labels.
    - y_pred: array-like of shape (n_samples,)
      Predicted labels, as returned by a classifier.
    - threshold_range: tuple of floats, optional
      The range to clip the true values to.

    Returns:
    - thresholds: array of floats
      The thresholds used to calculate the Brier score.
    - brier_scores: array of floats
      The Brier scores for each threshold.
    """
    assert plt is not None, "matplotlib is required to plot the Brier curve"
    if threshold_range is None:
        threshold_range = (0, 1)
    thresholds = np.linspace(*threshold_range, 100)
    idx = np.argsort(y_pred)
    insertion_indices = np.searchsorted(y_pred[idx], thresholds)
    false_neg = np.cumsum(y_true[idx])[insertion_indices]
    false_neg[-1] = np.sum(1-y_true[idx])
    true_neg = insertion_indices - false_neg
    true_neg[-1] = np.sum(y_true[idx])
    false_pos = np.sum(y_true[idx]) - true_neg
    costs = thresholds * false_pos + (1 - thresholds) * false_neg
    costs /= y_true.shape[0]
    plt.plot(thresholds, costs, label=label)
    plt.plot(thresholds, np.minimum(thresholds, 1-thresholds), color="lightgray", linestyle="--", zorder=-10)
    plt.xlabel("C/L")
    plt.ylabel("Regret")
    plt.title("Brier Curve")
    plt.show()

brier_score_scorer = make_scorer(brier_score, greater_is_better=True)