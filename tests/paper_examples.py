from matplotlib import pyplot as plt
import numpy as np
from ..briertools.logloss import log_loss_curve

def jail():
  pos = np.random.normal(loc=0, scale=1, size=300)
  neg = np.random.normal(loc=1, scale=1, size=300)
  y_pred = np.concatenate([pos, neg])
  y_true = np.concatenate([pos * 0 + 1, neg * 0])
  log_loss_curve(y_true, y_pred, threshold_range=(0.003, 0.66), fill_range=(1./101, 1./6), ticks=[1./101, 1./6, 1./2])

  pos = np.random.normal(loc=0, scale=1, size=300)
  neg = np.random.normal(loc=3, scale=1, size=300)
  y_pred = np.concatenate([pos, neg])
  y_true = np.concatenate([pos * 0 + 1, neg * 0])
  log_loss_curve(y_true, y_pred, threshold_range=(0.003, 0.66), fill_range=(1./101, 1./6), ticks=[1./101, 1./6, 1./2])
  plt.show()

def fraud():
  pos = np.random.normal(loc=0, scale=1, size=300)
  neg = np.random.normal(loc=1, scale=1, size=300)
  y_pred = np.concatenate([pos, neg])
  y_true = np.concatenate([pos * 0 + 1, neg * 0])
  log_loss_curve(y_true, y_pred, threshold_range=(0.333, 0.9995), fill_range=(100./101, 1000./1001), ticks=[1./2, 100./101, 1000./1001])

  pos = np.random.normal(loc=0, scale=1, size=300)
  neg = np.random.normal(loc=1, scale=2, size=300)
  y_pred = np.concatenate([pos, neg])
  y_true = np.concatenate([pos * 0 + 1, neg * 0])
  log_loss_curve(y_true, y_pred, threshold_range=(0.333, 0.9995), fill_range=(100./101, 1000./1001), ticks=[1./2, 100./101, 1000./1001])
  plt.show()

def cancer():
  pos = np.random.normal(loc=0, scale=1, size=300)
  neg = np.random.normal(loc=1, scale=1, size=300)
  y_pred = np.concatenate([pos, neg])
  y_true = np.concatenate([pos * 0 + 1, neg * 0])
  log_loss_curve(y_true, y_pred, threshold_range=(0.03, 0.66), fill_range=(1./11, 1./3), ticks=[1./11, 1./3, 1./2])

  pos = np.random.normal(loc=0, scale=1, size=300)
  neg = np.random.normal(loc=2, scale=1, size=300)
  y_pred = np.concatenate([pos, neg])
  y_true = np.concatenate([pos * 0 + 1, neg * 0])
  log_loss_curve(y_true, y_pred, threshold_range=(0.03, 0.66), fill_range=(1./11, 1./3), ticks=[1./11, 1./3, 1./2])
  plt.show()

cancer()