import numpy as np

def pointwise_l1_loss(y_true, y_pred):
  return np.abs(np.array(y_true) - np.array(y_pred))

def l1_to_total_log_loss(l1_loss):
  return np.mean(-np.log(1 - l1_loss))

def l1_to_total_l2_loss(l1_loss):
  return np.mean(l1_loss ** 2)

def clip_loss(y_true, y_pred, threshold_range):
  y_near = np.array(threshold_range)[np.array(y_true, dtype=int)]
  y_far = np.array(threshold_range)[np.array(1-y_true, dtype=int)]

  loss_near = pointwise_l1_loss(y_true, y_near)
  loss_far = pointwise_l1_loss(y_true, y_far)
  loss_pred = pointwise_l1_loss(y_true, y_pred)
  assert np.all(loss_near <= loss_far)

  loss_clip = np.clip(loss_pred, loss_near, loss_far)
  return loss_near, loss_clip

def assert_valid(y_true, y_pred):
  assert np.min(y_true) >= 0
  assert np.min(y_pred) > 0
  assert np.max(y_true) <= 1
  assert np.max(y_pred) < 1