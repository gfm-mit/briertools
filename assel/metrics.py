import numpy as np
from sklearn.metrics import roc_auc_score # To calculate AUC
from typing import Tuple

def brier_score_binary_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Brier score using direct binary predictions (Method 1).
    
    This method treats binary test results (0/1) directly as probability estimates,
    as described in the appendix Method 1 calculation.
    
    Args:
        y_true: True binary disease status (0/1)
        y_pred: Predicted values (binary 0/1 or continuous probabilities)
        
    Returns:
        Brier score
    """
    return np.mean((y_true - y_pred) ** 2)

def brier_score_predictive_values(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Brier score using PPV and NPV conversion (Method 2).
    
    For binary tests, this replaces test results with the positive and negative
    predictive values as described in the appendix Method 2 calculation.
    
    Args:
        y_true: True binary disease status (0/1)
        y_pred: Binary test results (0/1)
        
    Returns:
        Brier score
    """
    # Convert binary predictions to probabilities using PPV and NPV
    probs = np.zeros_like(y_pred, dtype=float)
    
    # Calculate PPV and NPV
    pos_mask = y_pred == 1
    neg_mask = y_pred == 0
    
    # PPV = P(disease=1 | test=1)
    if np.sum(pos_mask) > 0:
        ppv = np.mean(y_true[pos_mask])
        probs[pos_mask] = ppv
    
    # 1-NPV = 1 - P(disease=0 | test=0) = P(disease=1 | test=0)
    if np.sum(neg_mask) > 0:
        # Direct calculation of P(disease=1 | test=0)
        prob_disease_given_negative = np.mean(y_true[neg_mask])
        probs[neg_mask] = prob_disease_given_negative
    
    # Calculate Brier score using these probabilities
    return np.mean((y_true - probs) ** 2)

def calculate_clinical_net_benefit(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold_pt: float) -> float:
    """
    Calculates Net Benefit at a given probability threshold.
    
    Net benefit measures the clinical utility of a model by incorporating
    the threshold probability, which reflects the relative harm of false
    positives versus false negatives.
    """
    y_true = np.asarray(y_true)
    y_pred_prob = np.asarray(y_pred_prob)
    n = len(y_true)
    if n == 0 or threshold_pt <= 0 or threshold_pt >= 1:
        return 0.0

    # Classify based on threshold
    # Note: For binary tests (0/1), this directly uses the test result if threshold < 1
    # This interpretation aligns with how net benefit is typically calculated for binary tests
    # (treating the test result itself as the indicator for action).
    y_pred_class = (y_pred_prob >= threshold_pt).astype(int)

    tp = np.sum((y_pred_class == 1) & (y_true == 1))
    fp = np.sum((y_pred_class == 1) & (y_true == 0))

    net_benefit = (tp / n) - (fp / n) * (threshold_pt / (1 - threshold_pt))
    return net_benefit

def calculate_diagnostic_test_performance(y_true: np.ndarray, test_result: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Calculates comprehensive performance metrics for a binary diagnostic test.
    
    Returns sensitivity, specificity, AUC, both Brier scores (direct and predictive values),
    and predictive values (PPV, NPV) for clinical interpretation.
    """
    y_true = np.asarray(y_true)
    test_result = np.asarray(test_result) # This is y_pred for binary tests
    n = len(y_true)

    tp = np.sum((test_result == 1) & (y_true == 1))
    fp = np.sum((test_result == 1) & (y_true == 0))
    tn = np.sum((test_result == 0) & (y_true == 0))
    fn = np.sum((test_result == 0) & (y_true == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # Use test_result directly for AUC calculation
    auc = roc_auc_score(y_true, test_result) if len(np.unique(test_result)) > 1 else 0.5

    # Brier Method 1: Use test result (0/1) as prediction
    brier_m1 = brier_score_binary_predictions(y_true, test_result)

    # Brier Method 2: Use PPV / (1-NPV) as prediction
    brier_m2 = brier_score_predictive_values(y_true, test_result)

    return sensitivity, specificity, auc, brier_m1, brier_m2