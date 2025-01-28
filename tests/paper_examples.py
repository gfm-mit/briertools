from matplotlib import pyplot as plt
import numpy as np
import scipy
from briertools.logloss import log_loss_curve, log_loss
from briertools.brier import brier_curve
from sklearn.metrics import roc_auc_score, roc_curve

from briertools.utils import partition_loss

def simulate_binormal(loc, scale=1, scale_neg=1, loc_neg=None, n=3000, fix=True):
  if loc_neg is None:
    loc_neg = -loc
  neg = np.random.normal(loc=loc_neg, scale=scale_neg, size=n)
  pos = np.random.normal(loc=loc, scale=scale, size=n)
  if fix:
    pos, neg = neg, pos
  y_pred = scipy.special.expit(np.concatenate([pos, neg]))
  y_true = np.concatenate([pos * 0 + 1, neg * 0])

  return y_pred, y_true

def draw_curve(y_true, y_pred, **kwargs):
  return log_loss_curve(y_true, y_pred, **kwargs)

def roc():
  y_pred, y_true = simulate_binormal(1, 1, fix=False)
  fig, axs = plt.subplots(3, 1, figsize=(4, 6))
  plt.sca(axs[0])
  draw_curve(y_true, y_pred, ticks=[1./11, 1./3, 1./2])
  plt.sca(axs[1])
  fpr, tpr, _ = roc_curve(y_true, y_pred)
  auc = 1-roc_auc_score(y_true, y_pred)
  plt.plot(fpr, tpr, label=f"AUC: {auc:.2f}")
  plt.xlabel('FPR')
  plt.ylabel('TPR')

  plt.sca(axs[2])
  calibration_loss, discrimination_loss = partition_loss(y_true, y_pred, log_loss)
  plt.scatter(calibration_loss, discrimination_loss)
  plt.xlabel("Calibration Loss")
  plt.ylabel("Discrimination Loss")

  y_pred, y_true = simulate_binormal(3, .5, loc_neg=1, fix=False)
  plt.sca(axs[0])
  draw_curve(y_true, y_pred, ticks=[1./11, 1./3, 1./2])
  plt.sca(axs[1])
  fpr, tpr, _ = roc_curve(y_true, y_pred)
  auc = 1-roc_auc_score(y_true, y_pred)
  plt.plot(fpr, tpr, label=f"AUC: {auc:.2f}")
  plt.xlabel('FPR')
  plt.ylabel('TPR')

  plt.sca(axs[2])
  calibration_loss, discrimination_loss = partition_loss(y_true, y_pred, log_loss)
  plt.scatter(calibration_loss, discrimination_loss)

  plt.sca(axs[0])
  plt.legend()
  plt.title("Log Loss")
  plt.sca(axs[1])
  plt.legend()
  plt.title("ROC")
  plt.sca(axs[2])
  plt.title("Decomposition")
  plt.tight_layout()
  plt.show()

#jail()
#cancer()
#fraud()
#weights()
roc()