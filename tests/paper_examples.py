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

def jail():
  plt.figure(figsize=(4,2))
  y_pred, y_true = simulate_binormal(1, 2, scale_neg=2)
  draw_curve(y_true, y_pred, draw_range=(0.003, 0.55), fill_range=(1./101, 1./6), ticks=[1./101, 1./6, 1./2])

  y_pred, y_true = simulate_binormal(1, 1)
  draw_curve(y_true, y_pred, draw_range=(0.003, 0.55), fill_range=(1./101, 1./6), ticks=[1./101, 1./6, 1./2])

  plt.legend()
  plt.tight_layout()
  plt.show()

def fraud():
  plt.figure(figsize=(4,2))
  y_pred, y_true = simulate_binormal(1, 1.5)
  draw_curve(y_true, y_pred, draw_range=(0.333, 0.9995), fill_range=(100./101, 1000./1001), ticks=[1./2, 100./101, 1000./1001])

  y_pred, y_true = simulate_binormal(1, 5)
  draw_curve(y_true, y_pred, draw_range=(0.333, 0.9995), fill_range=(100./101, 1000./1001), ticks=[1./2, 100./101, 1000./1001])

  plt.legend()
  plt.tight_layout()
  plt.show()

def cancer():
  plt.figure(figsize=(4,2))
  y_pred, y_true = simulate_binormal(1, 1, scale_neg=2)
  draw_curve(y_true, y_pred, draw_range=(0.03, 0.66), fill_range=(1./11, 1./3), ticks=[1./11, 1./3, 1./2])

  y_pred, y_true = simulate_binormal(1, 1)
  draw_curve(y_true, y_pred, draw_range=(0.03, 0.66), fill_range=(1./11, 1./3), ticks=[1./11, 1./3, 1./2])

  plt.legend()
  plt.tight_layout()
  plt.show()

def weights():
  z = np.linspace(-5, 5, 100)
  x = scipy.special.expit(z)
  one = x * 0 + 1
  beta00 = one/x/(1-x)/40
  lower = np.minimum(one, beta00)
  fig, axs = plt.subplots(1, 2, figsize=(8, 3))
  plt.sca(axs[0])

  color1 = plt.plot(x, one, label="Brier Score")[0].get_color()
  color2 = plt.plot(x, beta00, label="Log Loss")[0].get_color()
  plt.fill_between(x, one, lower, color=color1, alpha=0.2, zorder=-10)
  plt.fill_between(x, beta00, lower, color=color2, alpha=0.2, zorder=-10)
  plt.fill_between(x, one * 0, lower, color="black", alpha=0.2, zorder=-10)
  color3 = plt.plot([0, 0.5, 0.5, 0.5, 1], [0.01, 0.01, 2, 0.01, 0.01], label="Accuracy")[0].get_color()
  plt.ylim([0,1.5])
  plt.xlabel("C/L")
  plt.xticks([1./101, 1./11, 1./4., 1./2, 3./4, 10./11, 100./101],
             "\n1:100 1:10 1:3 1:1 3:1 10:1 \n100:1".split(" "))
  plt.legend(loc="lower right")

  plt.sca(axs[1])
  w = x * (1-x)
  one = w
  beta00 = x * 0 + .1
  lower = np.minimum(one, beta00)
  plt.plot(z, one, color=color1, label="Brier Score")
  plt.plot(z, beta00, color=color2, label="Log Loss")
  plt.fill_between(z, one, lower, color=color1, alpha=0.2, zorder=-10)
  plt.fill_between(z, beta00, lower, color=color2, alpha=0.2, zorder=-10)
  plt.fill_between(z, one * 0, lower, color="black", alpha=0.2, zorder=-10)
  plt.plot([-5, 0, 0, 0, 5], [0.01, 0.01, 2, 0.01, 0.01], color=color3, label="Accuracy")
  plt.ylim([0,.5])
  plt.xlabel("log odds C / L")
  plt.xticks(scipy.special.logit([1./101, 1./11, 1./4., 1./2, 3./4, 10./11, 100./101]),
             "1:100 1:10 1:3 1:1 3:1 10:1 100:1".split(" "))
  plt.legend(loc="upper right")
  for ax in axs:
    plt.sca(ax)
    plt.ylabel("Averaging Weight")
    plt.title("Priors over Thresholds")
  plt.tight_layout()
  plt.show()

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