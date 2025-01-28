from matplotlib import pyplot as plt
import numpy as np
import scipy
from briertools.logloss import log_loss_curve, log_loss
from briertools.brier import brier_curve
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve

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
  y_pred, y_true = simulate_binormal(.8, 1, fix=False, n=300)
  fig, axs = plt.subplots(1, 3, figsize=(7, 2.5))
  plt.sca(axs[0])
  draw_curve(y_true, y_pred, ticks=[1./101, 1./2])
  plt.sca(axs[1])
  precision, recall, _ = precision_recall_curve(y_true, y_pred)
  auc = average_precision_score(y_true, y_pred)
  plt.plot(recall, precision, label=f"{auc:.2f}")
  plt.xlabel('Recall')
  plt.ylabel('Precision')

  plt.sca(axs[2])
  calibration_loss, discrimination_loss = partition_loss(y_true, y_pred, log_loss)
  plt.scatter(calibration_loss, discrimination_loss)
  plt.xlabel("Calibration Loss")
  plt.ylabel("Discrimination Loss")

  y_pred, y_true = simulate_binormal(3, .5, loc_neg=1, fix=False, n=300)
  plt.sca(axs[0])
  draw_curve(y_true, y_pred, ticks=[1./101, 1./2])
  plt.sca(axs[1])
  precision, recall, _ = precision_recall_curve(y_true, y_pred)
  auc = average_precision_score(y_true, y_pred)
  plt.plot(recall, precision, label=f"{auc:.2f}")
  plt.xlabel('Recall')
  plt.ylabel('Precision')

  plt.sca(axs[2])
  calibration_loss, discrimination_loss = partition_loss(y_true, y_pred, log_loss)
  plt.scatter(calibration_loss, discrimination_loss)

  plt.sca(axs[0])
  plt.legend()
  plt.title("Log Loss")
  plt.sca(axs[1])
  plt.legend()
  plt.title("AUC-PR")
  plt.sca(axs[2])
  plt.title("Log Loss\nDecomposition")
  plt.xlim([0, .55])
  plt.ylim([0, .55])
  plt.tight_layout()
  plt.show()

#jail()
#cancer()
#fraud()
#weights()
roc()