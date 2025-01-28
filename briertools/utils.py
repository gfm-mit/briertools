import numpy as np
from sklearn.isotonic import IsotonicRegression

def pointwise_l1_loss(y_true, y_pred):
  return np.abs(np.array(y_true) - np.array(y_pred))

def l1_to_total_log_loss(l1_loss):
  return np.mean(-np.log(1 - l1_loss))

def l1_to_total_l2_loss(l1_loss):
  return np.mean(l1_loss ** 2)

def clip_loss(y_true, y_pred, threshold_range):
  y_near = np.array(threshold_range)[np.array(y_true, dtype=int)]
  y_far = np.array(threshold_range)[1-np.array(y_true, dtype=int)]

  loss_near = pointwise_l1_loss(y_true, y_near)
  loss_far = pointwise_l1_loss(y_true, y_far)
  loss_pred = pointwise_l1_loss(y_true, y_pred)
  assert np.all(loss_near <= loss_far)

  loss_clip = np.clip(loss_pred, loss_near, loss_far)
  return loss_near, loss_clip

def assert_valid(y_true, y_pred):
  assert np.min(y_true) >= 0
  assert np.min(y_pred) >= 0
  assert np.max(y_true) <= 1
  assert np.max(y_pred) <= 1

def get_regret(y_true, y_pred, thresholds):
    if not isinstance(y_true, np.ndarray):
      y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
      y_pred = np.array(y_pred)

    idx = np.argsort(y_pred)
    insertion_indices = np.searchsorted(y_pred[idx], thresholds)
    false_neg = np.concatenate([[0], np.cumsum(y_true[idx])])[insertion_indices]
    false_pos = np.sum(1-y_true[idx]) - np.concatenate([[0], np.cumsum(1-y_true[idx])])[insertion_indices]

    costs = thresholds * false_pos + (1 - thresholds) * false_neg
    costs /= y_true.shape[0]
    return costs

def partition_loss(y_true, y_pred, loss_fn, thresholds=None):
  print("thresholds", thresholds)
  loss = loss_fn(y_true, y_pred, thresholds)
  ir = IsotonicRegression()
  y_pred_iso = ir.fit_transform(y_pred, y_true)
  discrimination_loss = loss_fn(y_true, y_pred_iso, thresholds)
  calibration_loss = loss - discrimination_loss
  return calibration_loss, discrimination_loss