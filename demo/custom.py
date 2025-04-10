import argparse
from matplotlib import pyplot as plt
import numpy as np
import scipy
from briertools.scorers import LogLossScorer, BrierScorer, DCAScorer
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
import demo.formatter
from assel.simulation import ClinicalPredictionModel, generate_disease_status


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
    # Generate true disease status with 20% prevalence
    n_patients = 600
    prevalence = 0.20
    y_true = generate_disease_status(n_patients=n_patients, prevalence=prevalence)
    
    # Use severely miscalibrated model
    y_pred_miscalibrated = ClinicalPredictionModel.severe_risk_underestimation_model(y_true)
    
    # Use high specificity test model
    y_pred_high_spec = ClinicalPredictionModel.calibrated_binary(y_true)
    
    # Create scorers
    brier_scorer = LogLossScorer()
    
    # Changed to 1 row, 4 columns with specified width ratios
    fig, axs = plt.subplots(1, 4, figsize=(7, 2.5), gridspec_kw={'width_ratios': [3, 3, 3, 1]})
    
    # Plot log loss curves
    plt.sca(axs[2])
    
    # Add panel label C
    axs[2].text(0.8, 0.95, 'C', transform=axs[2].transAxes, 
                fontsize=12, fontweight='bold', va='top')
    
    draw_curve(y_true, y_pred_miscalibrated, scorer=brier_scorer, ticks=[1.0 / 101, 100./101])
    draw_curve(y_true, y_pred_high_spec, scorer=brier_scorer, ticks=[1.0 / 101, 100./101])
    
    # Plot ROC curves
    plt.sca(axs[1])
    
    # Add panel label B
    axs[1].text(0.1, 0.95, 'B', transform=axs[1].transAxes, 
                fontsize=12, fontweight='bold', va='top')
    
    # Severly miscalibrated model
    fpr, tpr, _ = roc_curve(y_true, y_pred_miscalibrated)
    auc = roc_auc_score(y_true, y_pred_miscalibrated)
    plt.plot(fpr, tpr, label=f"Severely\nMiscalibrated\n(AUC: {auc:.2f})")

    # High specificity test
    fpr, tpr, _ = roc_curve(y_true, y_pred_high_spec)
    auc = roc_auc_score(y_true, y_pred_high_spec)
    plt.plot(fpr, tpr, label=f"High\nSpecificity\n(AUC: {auc:.2f})")
    
    # Plot loss decomposition
    plt.sca(axs[0])
    
    # Add panel label A
    axs[0].text(0.8, 0.95, 'A', transform=axs[0].transAxes, 
                fontsize=12, fontweight='bold', va='top')
    
    # Severly miscalibrated model
    calibration_loss, discrimination_loss = brier_scorer._partition_loss(y_true, y_pred_miscalibrated, brier_scorer.score)
    plt.scatter(calibration_loss, discrimination_loss)

    # High specificity test
    calibration_loss, discrimination_loss = brier_scorer._partition_loss(y_true, y_pred_high_spec, brier_scorer.score)
    plt.scatter(calibration_loss, discrimination_loss)

    plt.sca(axs[2])
    plt.ylabel("Regret")
    plt.title("Log Loss")
    plt.sca(axs[1])
    plt.ylabel("TPR")
    plt.xlabel("FPR")
    plt.title("AUC-ROC")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.sca(axs[0])
    plt.xlabel("Calibration Loss")
    plt.ylabel("Discrimination Loss")
    plt.title("Decomposition")
    plt.xlim([0, 0.45])
    plt.ylim([0, 0.45])
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Remove legends from the first three subplots and add to the fourth
    for ax in axs[:3]:
        if ax.get_legend():
            ax.get_legend().remove()
    
    # Get handles and labels from plot 1 (ROC plot)
    handles, labels = axs[1].get_legend_handles_labels()
    axs[3].legend(handles, labels, loc='center', fontsize=8, frameon=False)
    axs[3].axis('off')  # Turn off axes for the fourth subplot

    plt.suptitle("Cancer Detection Performance: Calibration vs Discrimination")
    plt.tight_layout()
    return plt.gca()


def dca():
    # Generate true disease status with 20% prevalence
    n_patients = 2000
    prevalence = 0.20
    y_true = generate_disease_status(n_patients=n_patients, prevalence=prevalence)
    
    # Get predictions from models
    y_pred_high_spec = ClinicalPredictionModel.high_specificity_test(y_true)
    y_pred_well_calibrated = ClinicalPredictionModel.severe_risk_underestimation_model(y_true)
    
    # Create scorers
    dca_scorer = DCAScorer()
    brier_scorer = BrierScorer()
    log_loss_scorer = LogLossScorer()
    
    # Changed to 1 row, 4 columns with specified width ratios
    fig, axs = plt.subplots(1, 4, figsize=(8, 2.5), sharey=True, gridspec_kw={'width_ratios': [2, 2, 2, 1]})
    axs[2].set_xscale("log")
    demo.formatter.scale_x_one_minus_one_minus_x_2(axs[1])
    
    # Plot original DCA
    plt.sca(axs[0])
    
    # Add panel label A
    axs[0].text(0.9, 0.95, 'A', transform=axs[0].transAxes, 
                fontsize=12, fontweight='bold', va='top')
    
    draw_curve(
        y_true,
        y_pred_well_calibrated,
        scorer=dca_scorer,
        ticks=[1.0 / 101, 1.0 / 2],
        draw_range=[1e-2, 1 - 1e-2],
        fill_range=(0.05, 0.20),
        label="Well Calibrated"
    )
    draw_curve(
        y_true,
        y_pred_high_spec,
        scorer=dca_scorer,
        ticks=[1.0 / 101, 1.0 / 2],
        draw_range=[1e-2, 1 - 1e-2],
        fill_range=(0.05, 0.20),
        label="High Specificity"
    )
    plt.axhline(y=0.2, color="black", linestyle="--", lw=0.5, zorder=-10)
    
    # Plot Brier Score
    plt.sca(axs[1])
    
    # Add panel label B
    axs[1].text(0.9, 0.95, 'B', transform=axs[1].transAxes, 
                fontsize=12, fontweight='bold', va='top')
    
    draw_curve(
        y_true,
        y_pred_well_calibrated,
        scorer=dca_scorer,
        ticks=[1.0 / 101, 1.0 / 2],
        draw_range=[1e-2, 1 - 1e-2],
        fill_range=(0.05, 0.20),
        label="Well Calibrated"
    )
    draw_curve(
        y_true,
        y_pred_high_spec,
        scorer=dca_scorer,
        ticks=[1.0 / 101, 1.0 / 2],
        draw_range=[1e-2, 1 - 1e-2],
        fill_range=(0.05, 0.20),
        label="High Specificity"
    )
    plt.axhline(y=0.2, color="black", linestyle="--", lw=0.5, zorder=-10)
    
    # Plot Log Loss
    plt.sca(axs[2])
    
    # Add panel label C
    axs[2].text(0.9, 0.95, 'C', transform=axs[2].transAxes, 
                fontsize=12, fontweight='bold', va='top')
    
    draw_curve(
        y_true,
        y_pred_well_calibrated,
        scorer=dca_scorer,
        ticks=[1.0 / 101, 1.0 / 2],
        draw_range=[1e-2, 1 - 1e-2],
        fill_range=(0.05, 0.20),
        label="Well Calibrated"
    )
    draw_curve(
        y_true,
        y_pred_high_spec,
        scorer=dca_scorer,
        ticks=[1.0 / 101, 1.0 / 2],
        draw_range=[1e-2, 1 - 1e-2],
        fill_range=(0.05, 0.20),
        label="High Specificity"
    )
    plt.axhline(y=0.2, color="black", linestyle="--", lw=0.5, zorder=-10)
    
    axs[0].set_ylabel("Net Benefit\n(units of True Positives)")
    axs[1].set_ylabel("")
    axs[2].set_ylabel("")
    axs[0].set_title("Decision Curve")
    axs[1].set_title("Brier Curve")
    axs[2].set_title("Log Loss Curve")
    axs[0].set_xlabel("C/L\n(Linear Scale)")
    axs[1].set_xlabel("C/L\n(Quadratic Scale)")
    axs[2].set_xlabel("C/L\n(Log Scale)")
    axs[1].set_xticks(
        [0.01, 0.1, 0.25, 0.5, 0.75],
        r"$\frac{1}{100}$ $\frac{1}{10}$ $\frac{1}{4}$ $\frac{1}{2}$ $\frac{3}{4}$".split(),
    )
    axs[2].set_xticks(
        [0.01, 0.1, 0.25, 0.5, 0.75],
        r"$\frac{1}{100}$ $\frac{1}{10}$ $\frac{1}{4}$ $\frac{1}{2}$ $\frac{3}{4}$".split(),
    )
    for ax in axs[:3]:
        ax.get_legend().remove()
    handles, labels = axs[0].get_legend_handles_labels()
    axs[3].legend(handles, ["Well\nCalibrated", "High\nSpecificity"], loc='center', fontsize=8, frameon=False)
    axs[3].axis('off')

    plt.ylim([-.02, .24])
    plt.yticks([0, 0.2], "0 $\pi$".split())
    plt.suptitle("Cancer Detection Performance\nDecisions Curves as H-Measures")
    plt.tight_layout()
    return plt.gca()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--flag", action="store_true")
  parser.add_argument("--out", type=str)
  args = parser.parse_args()
  if args.flag:
    ax = roc()
  else:
    ax = dca()
  if args.out:
    ax.figure.savefig(args.out)
  else:
    ax.figure.show()
