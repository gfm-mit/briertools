import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from assel.simulation import ClinicalPredictionModel, generate_disease_status
from assel.metrics import brier_score_binary_predictions, calculate_clinical_net_benefit, calculate_diagnostic_test_performance
from briertools.scorers import BrierScorer

PAPER_VALUES = [
  {
    "test": "Assume all negative",
    "specificity": "100%",
    "sensitivity": "0%",
    "auc": 0.5,
    "brier_score": 0.2000,
    "brier_score_method_1": np.nan,
    "brier_score_method_2": np.nan,
    "net_benefit_thresh_5_percent": 0.0000,
    "net_benefit_thresh_10_percent": 0.0000,
    "net_benefit_thresh_20_percent": 0.0000
  },
  {
    "test": "Assume all positive",
    "specificity": "0%",
    "sensitivity": "100%",
    "auc": 0.5,
    "brier_score": 0.8000,
    "brier_score_method_1": np.nan,
    "brier_score_method_2": np.nan,
    "net_benefit_thresh_5_percent": 0.1579,
    "net_benefit_thresh_10_percent": 0.1111,
    "net_benefit_thresh_20_percent": 0.0
  },
  {
    "test": "Highly specific",
    "specificity": "95%",
    "sensitivity": "50%",
    "auc": 0.725,
    "brier_score": np.nan,
    "brier_score_method_1": 0.1400,
    "brier_score_method_2": 0.1169,
    "net_benefit_thresh_5_percent": 0.0979,
    "net_benefit_thresh_10_percent": 0.0956,
    "net_benefit_thresh_20_percent": 0.0900
  },
  {
    "test": "Highly sensitive",
    "specificity": "50%",
    "sensitivity": "95%",
    "auc": 0.725,
    "brier_score": np.nan,
    "brier_score_method_1": 0.4100,
    "brier_score_method_2": 0.1386,
    "net_benefit_thresh_5_percent": 0.1689,
    "net_benefit_thresh_10_percent": 0.1456,
    "net_benefit_thresh_20_percent": 0.0900
  },
  {
    "test": "Well calibrated",
    "specificity": "–",
    "sensitivity": "–",
    "auc": 0.75,
    "brier_score": 0.1386,
    "brier_score_method_1": np.nan,
    "brier_score_method_2": np.nan,
    "net_benefit_thresh_5_percent": 0.1595,
    "net_benefit_thresh_10_percent": 0.1236,
    "net_benefit_thresh_20_percent": 0.0716
  },
  {
    "test": "Overestimating risk",
    "specificity": "–",
    "sensitivity": "–",
    "auc": 0.75,
    "brier_score": 0.1708,
    "brier_score_method_1": np.nan,
    "brier_score_method_2": np.nan,
    "net_benefit_thresh_5_percent": 0.1583,
    "net_benefit_thresh_10_percent": 0.1160,
    "net_benefit_thresh_20_percent": 0.0423
  },
  {
    "test": "Underestimating risk",
    "specificity": "–",
    "sensitivity": "–",
    "auc": 0.75,
    "brier_score": 0.1540,
    "brier_score_method_1": np.nan,
    "brier_score_method_2": np.nan,
    "net_benefit_thresh_5_percent": 0.1483,
    "net_benefit_thresh_10_percent": 0.0986,
    "net_benefit_thresh_20_percent": 0.0413
  },
  {
    "test": "Severely underestimating risk",
    "specificity": "–",
    "sensitivity": "–",
    "auc": 0.75,
    "brier_score": 0.1760,
    "brier_score_method_1": np.nan,
    "brier_score_method_2": np.nan,
    "net_benefit_thresh_5_percent": 0.0921,
    "net_benefit_thresh_10_percent": 0.0372,
    "net_benefit_thresh_20_percent": 0.0076
  }
]

# Create a mapping from old model names to new model names for result comparison
MODEL_NAME_MAPPING = {
    'Treat none (default negative)': 'Assume all negative',
    'Treat all (default positive)': 'Assume all positive',
    'High specificity test': 'Highly specific',
    'High sensitivity test': 'Highly sensitive',
    'Well calibrated model': 'Well calibrated',
    'Risk overestimation model': 'Overestimating risk',
    'Risk underestimation model': 'Underestimating risk',
    'Severe risk underestimation model': 'Severely underestimating risk'
}

def generate_simulation_results(add_brier=False):
    """Generate simulation results using the ClinicalPredictionModel API.
    
    Returns:
        pd.DataFrame: DataFrame containing calculated metrics for all models.
    """
    # 1. Simulate True Disease Status
    y_true = generate_disease_status(n_patients=1_000_000, prevalence=0.20, seed=ClinicalPredictionModel.SEED)
    print("Disease status generated.")
    
    # --- Calculate Metrics from Simulation ---
    print("\n--- Calculating Metrics from Simulated Data ---")
    results_list = []
    thresholds = [0.05, 0.10, 0.20]
    
    # Get all available models
    models = ClinicalPredictionModel.get_all_models()
    
    # Process each model
    for model_name, model_func in models.items():
        if model_name == "Calibrated binary (not in paper)":
            continue
        print(f"Processing model: {model_name}")
        
        # Get the original model name for result comparison
        orig_model_name = MODEL_NAME_MAPPING.get(model_name, model_name)
        
        # Generate predictions
        y_pred = model_func(y_true)
        
        # Prepare result dictionary
        result = {'test': orig_model_name}
        
        # Calculate metrics using calculate_diagnostic_test_performance for all models
        sens, spec, auc, brier_m1, brier_m2 = calculate_diagnostic_test_performance(y_true, y_pred)
        
        # Format sensitivity and specificity according to model type
        if orig_model_name in ['Assume all negative', 'Assume all positive', 'Highly specific', 'Highly sensitive']:
            result['specificity'] = f"{spec*100:.1f}%"
            result['sensitivity'] = f"{sens*100:.1f}%"
        else:
            # For continuous models, keep the dash notation
            result['specificity'] = '–'
            result['sensitivity'] = '–'
        
        result['auc'] = auc
        
        # Assign appropriate Brier scores based on model type
        if orig_model_name in ['Highly specific', 'Highly sensitive']:
            result['brier_score'] = None
            result['brier_score_method_1'] = brier_m1
            result['brier_score_method_2'] = brier_m2
        else:
            # For other models, use the standard Brier score
            result['brier_score'] = brier_m1
            result['brier_score_method_1'] = None
            result['brier_score_method_2'] = None
        
        # Calculate Net Benefit for all thresholds
        for threshold in thresholds:
            threshold_key = f'net_benefit_thresh_{int(threshold*100)}_percent'
            net_benefit = calculate_clinical_net_benefit(y_true, y_pred, threshold)
            result[threshold_key] = net_benefit

        if add_brier:
            result['brier_5_20'] = BrierScorer().score(y_true, y_pred, threshold_range=(0.05, 0.20)) / (0.20 - 0.05)
        
        results_list.append(result)
    
    # Create DataFrame from calculated results
    results_df = pd.DataFrame(results_list)
    results_df.set_index('test', inplace=True)
    print("Metrics calculated.")
    
    return results_df

def run_simulation():
    """Run the simulation and compare results with target values."""
    
    # Load target data
    target_df = pd.DataFrame(PAPER_VALUES)
    target_df.set_index('test', inplace=True)
    print("Target metrics loaded.")
    
    # Generate simulation results
    results_df = generate_simulation_results()
    
    # --- Compare Results ---
    print("\n--- Comparison: Target vs. Calculated Metrics ---")
    
    # Select and order columns for comparison (numeric ones first)
    cols_to_compare_numeric = [
        'auc', 'brier_score', 'brier_score_method_1', 'brier_score_method_2',
        'net_benefit_thresh_5_percent', 'net_benefit_thresh_10_percent',
        'net_benefit_thresh_20_percent'
    ]
    cols_other = ['specificity', 'sensitivity'] # Keep these as strings for display
    
    # Convert target columns to numeric where possible
    target_comp = target_df.copy()
    for col in cols_to_compare_numeric:
        target_comp[col] = pd.to_numeric(target_comp[col], errors='coerce')
    
    # Select and format calculated results
    results_comp = results_df[cols_to_compare_numeric + cols_other].copy()
    
    # Combine for side-by-side view
    comparison_df = pd.concat([target_comp, results_comp], axis=1, keys=['Target', 'Calculated'])
    
    # Reorder columns for better readability: Metric (Target), Metric (Calculated)
    comparison_df = comparison_df.swaplevel(0, 1, axis=1).sort_index(axis=1)
    
    # Display comparison - adjust display options if needed
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print(comparison_df.round(4)) # Round numeric values for display
    
    # Optional: Calculate and display absolute differences for numeric columns
    print("\n--- Absolute Differences (Calculated - Target) ---")
    # Ensure results_comp only contains numeric columns before subtraction
    results_numeric = results_comp[cols_to_compare_numeric].copy()
    for col in cols_to_compare_numeric: # Ensure numeric type
         results_numeric[col] = pd.to_numeric(results_numeric[col], errors='coerce')
    
    diff_df = results_numeric - target_comp[cols_to_compare_numeric]
    print(diff_df.round(4))
    
    return comparison_df, diff_df

def paper_table():
    results_df = generate_simulation_results(add_brier=True)
    
    # Format the DataFrame for LaTeX output
    latex_df = results_df.copy()
    
    # Round numeric columns to appropriate precision
    numeric_columns = ['auc', 'brier_score', 'brier_score_method_1', 'brier_score_method_2', 
                       'net_benefit_thresh_5_percent', 'net_benefit_thresh_10_percent', 
                       'net_benefit_thresh_20_percent', 'brier_5_20']
    
    for col in numeric_columns:
        if col in latex_df.columns:
            # Convert to numeric, handling NaN values
            latex_df[col] = pd.to_numeric(latex_df[col], errors='coerce')
            # Format with 4 decimal places
            latex_df[col] = latex_df[col].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "—")
    
    # Escape percent signs in sensitivity and specificity columns
    for col in ['sensitivity', 'specificity']:
        if col in latex_df.columns:
            latex_df[col] = latex_df[col].astype(str).str.replace('%', '\\%')
    
    # Rename columns to escape percent signs and underscores for LaTeX
    column_mapping = {
        'net_benefit_thresh_5_percent': 'NB\\_5\\%',
        'net_benefit_thresh_10_percent': 'NB\\_10\\%',
        'net_benefit_thresh_20_percent': 'NB\\_20\\%',
        'brier_score': 'brier\\_score',
        'brier_score_method_1': 'brier\\_1',
        'brier_score_method_2': 'brier\\_2',
        'brier_5_20': 'brier\\_5\\_20'
    }
    latex_df = latex_df.rename(columns=column_mapping)
    latex_df = latex_df.sort_values(by='NB\\_5\\%', ascending=False)

    # Generate LaTeX table
    latex_code = latex_df.to_latex(
        index=True,
        columns=['auc', 'brier\\_score', 
                 'brier\\_1', 'brier\\_2', 'brier\\_5\\_20',
                 'NB\\_5\\%', 'NB\\_10\\%', 
                 'NB\\_20\\%'],
        bold_rows=True,
        float_format="%.4f",
        na_rep="—",
        escape=False  # Don't escape our manually escaped characters
    )
    
    # Print LaTeX code
    print("\n--- LaTeX Table ---\n")
    print(latex_code)
    
    return latex_code

if __name__ == "__main__":
    #run_simulation()
    paper_table()