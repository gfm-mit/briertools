import numpy as np
import scipy
from sklearn.metrics import make_scorer
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

def log_loss(y_true, y_pred, threshold_range=None):
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
      y_true = np.clip(y_true, threshold_range[0], threshold_range[1])
    return np.mean(np.log(1 - np.abs(np.array(y_true) - np.array(y_pred))))

def get_logit_ticks(min_val, max_val):
    """Generate tick marks for logit-scaled plots using append/prepend operations."""
    assert 0 <= min_val < max_val <= 1
        
    ticks = [0.5] if min_val <= 0.5 <= max_val else []

    bound = np.log10(min(min_val, 1-max_val))
    bound = 2-int(bound)
        
    for power in range(1, bound):
        val = 10.0 ** -power
        
        if min_val <= val <= max_val:
            ticks.insert(0, val)
        if min_val <= 1 - val <= max_val:
            ticks.insert(0, 1 - val)
    ticks = np.round(ticks, 16)
        
    return ticks

def log_loss_curve(y_true, y_pred, label=None, threshold_range=None):
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
        threshold_range = [0.01, 0.99]
    zscore = np.linspace(*scipy.special.logit(threshold_range), 100)
    expit = scipy.special.expit(zscore)
    idx = np.argsort(y_pred)
    insertion_indices = np.searchsorted(y_pred[idx], expit)
    false_neg = np.cumsum(y_true[idx])[insertion_indices]
    false_neg[-1] = np.sum(1-y_true[idx])
    true_neg = insertion_indices - false_neg
    true_neg[-1] = np.sum(y_true[idx])
    false_pos = np.sum(y_true[idx]) - true_neg
    costs = expit * false_pos + (1 - expit) * false_neg
    costs /= y_true.shape[0]
    plt.plot(zscore, costs, label=label)
    plt.plot(zscore, np.minimum(expit, 1-expit), color="lightgray", linestyle="--", zorder=-10)
    ticks = [0.01, 0.1, 0.5, 0.9, 0.99]
    if threshold_range is not None:
      ticks = get_logit_ticks(threshold_range[0], threshold_range[1])
    plt.xticks(scipy.special.logit(ticks), ticks)
    plt.xlabel("C/L")
    plt.ylabel("Regret")
    plt.title("Brier Curve (Log Loss Version)")
    plt.show()

log_loss_scorer = make_scorer(log_loss, greater_is_better=True)