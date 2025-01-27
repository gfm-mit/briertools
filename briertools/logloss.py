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
    score = np.mean(np.log(1 - np.abs(np.array(y_true) - np.array(y_pred))))
    if threshold_range is not None:
      y_bound = np.array(threshold_range)[y_true]
      baseline = np.mean(np.log(1 - np.abs(np.array(y_true) - np.array(y_bound))))
      return score - baseline
    return score

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

def log_loss_curve(y_true, y_pred, label=None, threshold_range=None, fill_range=None, ticks=None, hatch="/////"):
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
    if threshold_range is None:
      false_neg[-1] = np.sum(1-y_true[idx])
    true_neg = insertion_indices - false_neg
    if threshold_range is None:
      true_neg[-1] = np.sum(y_true[idx])
    false_pos = np.sum(y_true[idx]) - true_neg
    costs = expit * false_pos + (1 - expit) * false_neg
    costs /= y_true.shape[0]

    color = plt.plot(zscore, costs, label=label)[0].get_color()
    #plt.plot(zscore, np.minimum(expit, 1-expit), color="lightgray", linestyle="--", zorder=-10)

    if fill_range is not None:
      low, high = scipy.special.logit(fill_range)
      #print(low, high)
      fill_idx = (low < zscore) & (zscore < high)
      #print(fill_idx)
      plt.fill_between(
         zscore[fill_idx], costs[fill_idx], costs[fill_idx] * 0,
         color=color, alpha=0.3)

    if ticks is not None:
      tick_labels = np.round(np.where(np.array(ticks) <= 0.5, 1. / np.array(ticks) - 1, 1 - 1. / (1-np.array(ticks))))
      def format_tick(tick):
        if tick == 0.5:
          return "(1:1)\nAccuracy"
        if tick > 0.5:
          odds = 1. / (1-tick) - 1
          return f"{odds:.0f}:1"
        else:
          odds = 1. / tick - 1
          return f"1:{odds:.0f}"
      tick_labels = map(format_tick, ticks)
    elif threshold_range is not None:
      ticks = get_logit_ticks(threshold_range[0], threshold_range[1])
      tick_labels = ticks
    else:
      ticks = [0.01, 0.1, 0.5, 0.9, 0.99]
      tick_labels = ticks
    plt.xticks(scipy.special.logit(ticks), tick_labels)
    plt.xlabel("C/L")
    plt.ylabel("Regret")
    plt.title("Brier Curve (Log Loss Version)")

log_loss_scorer = make_scorer(log_loss, greater_is_better=True)