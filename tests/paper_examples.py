from matplotlib import pyplot as plt
import numpy as np
from briertools.logloss import log_loss_curve

def simulate_binormal(loc, scale=1, n=300):
  pos = np.random.normal(loc=0, scale=1, size=300)
  neg = np.random.normal(loc=loc, scale=scale, size=300)
  y_pred = np.concatenate([pos, neg])
  y_true = np.concatenate([pos * 0 + 1, neg * 0])
  return y_pred, y_true

def jail():
  y_pred, y_true = simulate_binormal(1, 1)
  log_loss_curve(y_true, y_pred, threshold_range=(0.003, 0.66), fill_range=(1./101, 1./6), ticks=[1./101, 1./6, 1./2])

  y_pred, y_true = simulate_binormal(3, 1)
  log_loss_curve(y_true, y_pred, threshold_range=(0.003, 0.66), fill_range=(1./101, 1./6), ticks=[1./101, 1./6, 1./2])

  plt.show()
  plt.tight_layout()

def fraud():
  y_pred, y_true = simulate_binormal(1, 1)
  log_loss_curve(y_true, y_pred, threshold_range=(0.333, 0.9995), fill_range=(100./101, 1000./1001), ticks=[1./2, 100./101, 1000./1001])

  y_pred, y_true = simulate_binormal(1, 2)
  log_loss_curve(y_true, y_pred, threshold_range=(0.333, 0.9995), fill_range=(100./101, 1000./1001), ticks=[1./2, 100./101, 1000./1001])

  plt.show()
  plt.tight_layout()

def cancer():
  y_pred, y_true = simulate_binormal(1, 1)
  log_loss_curve(y_true, y_pred, threshold_range=(0.03, 0.66), fill_range=(1./11, 1./3), ticks=[1./11, 1./3, 1./2])

  y_pred, y_true = simulate_binormal(2, 1)
  log_loss_curve(y_true, y_pred, threshold_range=(0.03, 0.66), fill_range=(1./11, 1./3), ticks=[1./11, 1./3, 1./2])

  plt.show()
  plt.tight_layout()

cancer()