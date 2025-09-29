from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss, log_loss
from briertools.scorers import BrierScorer, LogLossScorer, DCAScorer
import pandas as pd

def draw_curve(y_true, y_pred, scorer, **kwargs):
    """
    Wrapper function to draw the log loss curve using a scorer object.
    """
    ax = plt.gca()
    threshold_range = kwargs.pop('draw_range', None)
    if 'fill_range' in kwargs and not isinstance(kwargs['fill_range'], tuple):
        fill_value = kwargs['fill_range']
        kwargs['fill_range'] = (0.01, fill_value)
    scorer.plot_curve(
        ax, 
        y_true, 
        y_pred,
        threshold_range=threshold_range,
        fill_range=kwargs.get('fill_range'),
        ticks=kwargs.get('ticks'),
        alpha=kwargs.get('alpha', 0.3),
        label=kwargs.get('label'),
        use_data_label=False
    )
    return ax

def plot_predictions(fn, label_col='GroundTruth', BLOCKLIST='GroundTruth,KNeighborsClassifier,DecisionTreeClassifier'.split(','), cost_range=(2./100, 10./100), draw_range=(1./100, 1./30)):
  assert isinstance(BLOCKLIST, list) and not isinstance(BLOCKLIST, str)
  assert isinstance(cost_range, tuple) and len(cost_range) == 2
  assert isinstance(draw_range, tuple) and len(draw_range) == 2
  ticks = [2.0/1000., 1.6666 / 100, 2.0 / 100, 3.0/ 100, 4./100]

  cost_range = tuple([x / (x+1.0) for x in cost_range])
  draw_range = tuple([x / (x+1.0) for x in draw_range])
  ticks = np.array([x / (x+1.0) for x in ticks])
  ticks = ticks[(draw_range[0] < ticks) & (ticks < draw_range[1])]
  df = pd.read_csv(fn)
  print(",".join(list(df.columns)))
  models = [x for x in df.columns if x not in [label_col] + BLOCKLIST]

  # Calculate baseline (prevalence-only) Brier score
  prevalence = np.mean(df[label_col])
  baseline_brier = np.mean((df[label_col] - prevalence) ** 2)
  print(f"Baseline Brier Score (always guessing prevalence {prevalence:.4f}): {baseline_brier:.4f}\n")

  results = []
  model_names = {
    "logistic_minimal": "Logistic w/o Density",
    "logistic_+_density": "Logistic w/ Density",
    "xgboost": "XGBoost",
    "xgboost_2pct": "XGBoost (2% threshold)",
  }
  plt.subplots(figsize=(6, 3))
  for model in models:
    df2 = df[[label_col, model]].dropna()
    draw_curve(
      df2[label_col],
      df2[model],
      scorer=BrierScorer(),
      draw_range=draw_range,
      fill_range=cost_range,
      ticks=ticks,
      label=model_names[model]
      )
    auc = roc_auc_score(df2[label_col], df2[model])
    brier = brier_score_loss(df2[label_col], df2[model])
    log_loss_score = log_loss(df2[label_col], df2[model])
    
    # Store results in a list
    results.append({
      'Model': model,
      'AUC': auc,
      'Brier': brier,
      'Log_Loss': log_loss_score
    })

  # Convert results to pandas DataFrame and print
  results_df = pd.DataFrame(results)
  print("\nModel Performance Results:")
  print(results_df)

plot_predictions(
   'demo/data/bcsc.csv',
   label_col='cancer',
   BLOCKLIST='cancer,climatological,random_forest,xgboost_5pct,xgboost_8pct,xgboost_10pct,xgboost_15pct'.split(','),
   cost_range=(1.66/100, 3./100),
   draw_range=(1./1000, 4./100))
plt.show()