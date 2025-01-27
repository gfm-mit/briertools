import numpy as np
from sklearn.metrics import make_scorer

from briertools.utils import assert_valid, clip_loss, pointwise_l1_loss, l1_to_total_l2_loss
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
    assert_valid(y_true, y_pred)
    if threshold_range is None:
      return l1_to_total_l2_loss(pointwise_l1_loss(y_true, y_pred))
    loss_near, loss_clip = clip_loss(y_true, y_pred, threshold_range)
    near_score = l1_to_total_l2_loss(loss_near)
    far_score = l1_to_total_l2_loss(loss_clip)
    return far_score - near_score

def brier_curve(y_true, y_pred, threshold_range=None, fill_range=None, ticks=None):
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
    if threshold_range is None:
      false_neg[-1] = np.sum(1-y_true[idx])
    true_neg = insertion_indices - false_neg
    if threshold_range is None:
      true_neg[-1] = np.sum(y_true[idx])
    false_pos = np.sum(y_true[idx]) - true_neg
    costs = thresholds * false_pos + (1 - thresholds) * false_neg
    costs /= y_true.shape[0]

    loss = brier_score(y_true, y_pred, threshold_range)
    integral = np.trapz(costs, thresholds) * 2

    plt.plot(thresholds, costs, label=f"MSE: {loss:.2f} | $\mathbb{{E}}$ R(f): {integral:.2f}")
    plt.xlabel("C/L")
    plt.ylabel("Regret")
    plt.title("Brier Curve")

brier_score_scorer = make_scorer(brier_score, greater_is_better=True)