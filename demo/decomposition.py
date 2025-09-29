import argparse
from matplotlib import pyplot as plt
import numpy as np
import scipy
import pandas as pd
from briertools.scorers import LogLossScorer, BrierScorer, DCAScorer
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)
import demo.formatter


def draw_curve(y_true, y_pred, scorer, **kwargs):
    """
    Wrapper function for drawing curves using the new scorer objects.
    """
    ax = plt.gca()
    # Get user label but don't pass it to plot_curve
    user_label = kwargs.pop('label', None)
    
    # Ensure fill_range is a tuple if provided
    if 'fill_range' in kwargs and not isinstance(kwargs['fill_range'], tuple):
        fill_value = kwargs['fill_range']
        kwargs['fill_range'] = (0.01, fill_value)
    
    scorer.plot_curve(
        ax, 
        y_true, 
        y_pred,
        threshold_range=kwargs.get('draw_range'),
        fill_range=kwargs.get('fill_range'),
        ticks=kwargs.get('ticks'),
        alpha=kwargs.get('alpha', 0.3)
    )
    
    # Manually add the legend entry if a label was provided
    if user_label:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            handles[-1].set_label(user_label)
            ax.legend()
    
    return ax


def roc():
    df = pd.read_csv('demo/data/bcsc.csv')
    y_true = df.cancer
    
    # Use severely miscalibrated model
    y_pred_miscalibrated = df.logistic_minimal
    
    # Use high specificity test model
    y_pred_high_spec = df.xgboost
    
    # Create scorers
    brier_scorer = LogLossScorer()
    
    # Changed to 1 row, 4 columns with specified width ratios
    fig, axs = plt.subplots(1, 3, figsize=(7, 2.5), gridspec_kw={'width_ratios': [3, 3, 3]})
    
    # Plot log loss curves
    plt.sca(axs[1])
    
    # Add panel label C
    axs[2].text(0.85, 0.95, 'C', transform=axs[2].transAxes, 
                fontsize=12, fontweight='bold', va='top')
    
    draw_curve(y_true, y_pred_miscalibrated, scorer=brier_scorer, ticks=[1.0 / 101, 100. / 101])
    draw_curve(y_true, y_pred_high_spec, scorer=brier_scorer, ticks=[1.0 / 101, 100./101])
    
    # Plot ROC curves
    plt.sca(axs[0])
    
    # Add panel label B
    axs[1].text(0.85, 0.95, 'B', transform=axs[1].transAxes, 
                fontsize=12, fontweight='bold', va='top')
    
    # Severly miscalibrated model
    fpr, tpr, _ = roc_curve(y_true, y_pred_miscalibrated)
    auc = roc_auc_score(y_true, y_pred_miscalibrated)
    plt.plot(fpr, tpr, label=f"Logistic\nModel\n(AUC: {auc:.2f})")

    # High specificity test
    fpr, tpr, _ = roc_curve(y_true, y_pred_high_spec)
    auc = roc_auc_score(y_true, y_pred_high_spec)
    plt.plot(fpr, tpr, label=f"XGBoost\n(AUC: {auc:.2f})")
    
    # Plot loss decomposition
    plt.sca(axs[2])
    
    # Add panel label A
    axs[0].text(0.05, 0.95, 'A', transform=axs[0].transAxes, 
                fontsize=12, fontweight='bold', va='top')
    
    # Severly miscalibrated model
    calibration_loss, discrimination_loss = brier_scorer._partition_loss(y_true, y_pred_miscalibrated, brier_scorer.score)
    plt.scatter(calibration_loss, discrimination_loss)

    # High specificity test
    calibration_loss, discrimination_loss = brier_scorer._partition_loss(y_true, y_pred_high_spec, brier_scorer.score)
    plt.scatter(calibration_loss, discrimination_loss)

    plt.sca(axs[1])
    plt.ylabel("Regret")
    plt.title("Regret vs Threshold")
    plt.sca(axs[0])
    plt.ylabel("TPR")
    plt.xlabel("FPR")
    plt.title("ROC")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.sca(axs[2])
    plt.xlabel("Miscalibration")
    plt.ylabel("Discrimination")
    plt.title("Log Loss\nDecomposition")
    plt.xlim([0.033, 0.037])
    plt.ylim([0.119, 0.121])
    plt.yticks([0.119, 0.121])
    plt.xticks([0.033, 0.035, 0.037])
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Remove legends from the first three subplots and add to the fourth
    for ax in axs[:3]:
        if ax.get_legend():
            ax.get_legend().remove()
    
    # Get handles and labels from plot 1 (ROC plot)
    handles, labels = axs[1].get_legend_handles_labels()
    labels = ["Logistic", "XGBoost"]
    axs[0].legend(handles, labels, loc='lower right', fontsize=7)

    plt.suptitle("Cancer Detection: Logistic vs XGBoost")
    plt.tight_layout()
    return plt.gca()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", type=str)
  args = parser.parse_args()
  ax = roc()
  if args.out:
    ax.figure.savefig(args.out)
  else:
    plt.show()
