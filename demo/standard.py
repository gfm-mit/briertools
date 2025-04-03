from matplotlib import pyplot as plt
import numpy as np
import scipy
import argparse
from briertools.scorers import BrierScorer, LogLossScorer
from assel.simulation import ClinicalPredictionModel, generate_disease_status

def draw_curve(y_true, y_pred, scorer, **kwargs):
    """
    Wrapper function to draw the log loss curve using LogLossScorer.
    Similar to the log_loss_curve function used in the original code.
    """
    ax = plt.gca()
    
    # Convert parameters if needed
    threshold_range = kwargs.pop('draw_range', None)
    
    # Pass only parameters that LogLossScorer.plot_curve accepts
    scorer.plot_curve(
        ax, 
        y_true, 
        y_pred,
        threshold_range=threshold_range,
        fill_range=kwargs.get('fill_range'),
        ticks=kwargs.get('ticks'),
        alpha=kwargs.get('alpha', 0.3),
        label=kwargs.get('label')
    )
    return ax

def curve_comparison(model1, model2, scorer, title, label1, label2, fill_min=1./21, draw_max=0.97):
    """
    Recreate the jail() function using simulation.py for binary vs continuous model comparison.
    This implements the comparison between a continuous prediction model and binary tests
    as mentioned in comparisons.md.
    """
    plt.figure(figsize=(4, 2))
    
    # Generate true disease status with 20% prevalence
    n_patients = 6000
    prevalence = 0.20
    y_true = generate_disease_status(n_patients=n_patients, prevalence=prevalence)
    
    # Use the high sensitivity test (binary model)
    # "Binary test with high sensitivity would be clinically preferable in scenarios where sensitivity is critical"
    y_pred_binary = model1(y_true)
    
    ticks = np.array([1.0 / 101, 1.0/21, 1.0 / 6 ,1.0 / 2])
    ticks = ticks[ticks < draw_max]
    
    # Draw the curve for high sensitivity binary test
    draw_curve(
        y_true,
        y_pred_binary,
        label=label1,
        scorer=scorer,
        draw_range=(0.003, draw_max),
        fill_range=(fill_min, 1.0 / 6),
        ticks=ticks
    )
    
    # Use the well calibrated model (continuous model)
    # "Continuous prediction model (AUC 0.75) compared with binary tests (each with AUC 0.725)"
    y_pred_continuous = model2(y_true)
    
    # Draw the curve for well calibrated continuous model
    draw_curve(
        y_true,
        y_pred_continuous,
        label=label2,
        scorer=scorer,
        draw_range=(0.003, draw_max),
        fill_range=(fill_min, 1.0 / 6),
        ticks=ticks
    )
    
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    return plt.gca()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--flag", action="store_true")
  parser.add_argument("--out", type=str)
  args = parser.parse_args()
  if args.flag:
    ax = curve_comparison(
        model1=ClinicalPredictionModel.high_sensitivity_test,
        label1="High Sensitivity",
        model2=ClinicalPredictionModel.high_specificity_test,
        label2="High Specificity",
        scorer=LogLossScorer(),
        title="Average Sentencing Performance\nWith Bounded Tradeoffs",
        fill_min=1./101,
    ) 
  else:
    ax = curve_comparison(
        model1=ClinicalPredictionModel.well_calibrated_model,
        label1="Continuous",
        model2=ClinicalPredictionModel.high_specificity_test,
        label2="High Specificity",
        scorer=BrierScorer(),
        title="Cancer Detection Performance\nUnder Cost Heterogeneity",
        draw_max=0.25
    ) 
  if args.out:
    ax.figure.savefig(args.out)
  else:
    ax.figure.show()