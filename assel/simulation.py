import numpy as np
from scipy.special import expit # Logistic function (1 / (1 + exp(-x)))
from typing import Literal, Dict

ModelType = Literal[
    'high_sensitivity_test',
    'high_specificity_test',
    'calibrated_binary',
    'well_calibrated',
    'overpredict',
    'underpredict',
    'severe_underpredict'
]

# --- Simulation Functions ---

def generate_disease_status(
    n_patients: int = 1_000_000,
    prevalence: float = 0.20,
    seed: int = 42
) -> np.ndarray:
    """
    Simulates the true disease status for a population with given prevalence.
    
    As described in the appendix, this generates binary disease outcomes
    for a simulated patient cohort.
    """
    np.random.seed(seed)
    disease_status = np.random.binomial(1, prevalence, n_patients)
    return disease_status

def generate_clinical_predictions(
    disease_status: np.ndarray,
    model_type: ModelType,
    sens_high_sens_test_target: float = 0.95,
    spec_high_sens_test_target: float = 0.50,
    sens_high_spec_test_target: float = 0.50,
    spec_high_spec_test_target: float = 0.95,
    mean_z_negative: float = 0.0,
    mean_z_positive: float = 0.95,
    sd_z: float = 1.0,
    beta1: float = 1.0,
    beta0_approx: float = -0.83,
    gamma_overpredict: float = 0.38,
    gamma_underpredict: float = -0.3,
    gamma_severe_underpredict: float = -3.2,
    seed: int = 42
) -> np.ndarray:
    """
    Generates clinical test results or risk predictions based on true disease status.
    
    This function implements the simulation verification approach described in the 
    appendix, generating either binary test results with specific sensitivity/specificity
    characteristics or continuous probabilistic predictions with varying calibration.
    
    Returns:
        np.ndarray: Predicted values (binary test results or risk probabilities)
    """
    np.random.seed(seed + 1)
    n_patients = len(disease_status)
    n_positive = np.sum(disease_status)
    n_negative = n_patients - n_positive

    # Simulate Binary Test Results (needed for high_sensitivity_test/high_specificity_test types)
    # High Sensitivity Test
    high_sens_test_result = np.zeros(n_patients, dtype=int)
    high_sens_test_result[disease_status == 1] = np.random.binomial(1, sens_high_sens_test_target, n_positive)
    high_sens_test_result[disease_status == 0] = np.random.binomial(1, 1 - spec_high_sens_test_target, n_negative)

    # High Specificity Test
    high_spec_test_result = np.zeros(n_patients, dtype=int)
    high_spec_test_result[disease_status == 1] = np.random.binomial(1, sens_high_spec_test_target, n_positive)
    high_spec_test_result[disease_status == 0] = np.random.binomial(1, 1 - spec_high_spec_test_target, n_negative)

    # Simulate Continuous Predictor 'z' (needed for continuous models)
    z = np.zeros(n_patients)
    z[disease_status == 0] = np.random.normal(mean_z_negative, sd_z, n_negative)
    z[disease_status == 1] = np.random.normal(mean_z_positive, sd_z, n_positive)

    # Simulate Continuous Model Predictions (needed for continuous models)
    logit_base = beta0_approx + beta1 * z
    prob_well_calibrated = expit(logit_base)
    prob_overpredict = expit(logit_base + gamma_overpredict)
    prob_underpredict = expit(logit_base + gamma_underpredict)
    prob_severe_underpredict = expit(logit_base + gamma_severe_underpredict)

    # Extra test not in paper: calibrated binary test
    calibrated_binary_spec = .6
    calibrated_binary_sens = .6
    calibrated_binary_result = np.zeros(n_patients, dtype=float)
    calibrated_binary_result[disease_status == 1] = np.random.binomial(1, calibrated_binary_sens, n_positive)
    calibrated_binary_result[disease_status == 0] = np.random.binomial(1, 1 - calibrated_binary_spec, n_negative)
    calibrated_binary_result[calibrated_binary_result == 1] = np.mean(disease_status[calibrated_binary_result == 1])
    calibrated_binary_result[calibrated_binary_result == 0] = np.mean(disease_status[calibrated_binary_result == 0])

    # Select y_pred based on model_type
    if model_type == 'high_sensitivity_test':
        y_pred = high_sens_test_result
    elif model_type == 'high_specificity_test':
        y_pred = high_spec_test_result
    elif model_type == 'calibrated_binary':
        y_pred = calibrated_binary_result
    elif model_type == 'well_calibrated':
        y_pred = prob_well_calibrated
    elif model_type == 'overpredict':
        y_pred = prob_overpredict
    elif model_type == 'underpredict':
        y_pred = prob_underpredict
    elif model_type == 'severe_underpredict':
        y_pred = prob_severe_underpredict
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return y_pred

# --- Clinical Prediction Model API ---

class ClinicalPredictionModel:
    """
    A class that provides an API for generating predictions using various clinical
    prediction models and binary diagnostic tests as described in the Assel et al. paper.
    """
    
    # Default parameters
    SEED = 42
    
    # Binary Test Parameters
    SENS_HIGH_SENS_TEST = 0.95
    SPEC_HIGH_SENS_TEST = 0.50
    SENS_HIGH_SPEC_TEST = 0.50
    SPEC_HIGH_SPEC_TEST = 0.95
    
    # Continuous Model Parameters
    MEAN_Z_NEG = 0.0
    MEAN_Z_POS = 0.95
    SD_Z = 1.0
    BETA1 = 1.0
    BETA0 = -0.9
    
    # Calibration & Miscalibration Parameters
    GAMMA_OVER = 0.361
    GAMMA_UNDER = -0.285
    GAMMA_SEVERE_UNDER = -3.04
    
    @classmethod
    def default_treat_none(cls, y_true: np.ndarray) -> np.ndarray:
        """
        Generates predictions assuming all patients are disease-negative (treat none).
        
        This represents the default strategy of assuming all patients are negative,
        as described in the appendix "Comparison to default strategies" section.
        """
        return np.zeros_like(y_true)
    
    @classmethod
    def default_treat_all(cls, y_true: np.ndarray) -> np.ndarray:
        """
        Generates predictions assuming all patients are disease-positive (treat all).
        
        This represents the default strategy of assuming all patients are positive,
        as described in the appendix "Comparison to default strategies" section.
        """
        return np.ones_like(y_true)
    
    @classmethod
    def high_specificity_test(cls, y_true: np.ndarray) -> np.ndarray:
        """
        Generates predictions for the 'Highly specific' binary test.
        
        This test has high specificity (95%) but low sensitivity (50%),
        as described in the appendix Table 1.
        """
        return generate_clinical_predictions(
            disease_status=y_true,
            model_type='high_specificity_test',
            sens_high_spec_test_target=cls.SENS_HIGH_SPEC_TEST,
            spec_high_spec_test_target=cls.SPEC_HIGH_SPEC_TEST,
            seed=cls.SEED
        )
    
    @classmethod
    def calibrated_binary(cls, y_true: np.ndarray) -> np.ndarray:
        """
        Generates predictions for the 'Highly specific' binary test.
        """
        return generate_clinical_predictions(
            disease_status=y_true,
            model_type='calibrated_binary',
            sens_high_spec_test_target=cls.SENS_HIGH_SPEC_TEST,
            spec_high_spec_test_target=cls.SPEC_HIGH_SPEC_TEST,
            seed=cls.SEED
        )
    
    @classmethod
    def high_sensitivity_test(cls, y_true: np.ndarray) -> np.ndarray:
        """
        Generates predictions for the 'Highly sensitive' binary test.
        
        This test has high sensitivity (95%) but low specificity (50%),
        as described in the appendix Table 1.
        """
        return generate_clinical_predictions(
            disease_status=y_true,
            model_type='high_sensitivity_test',
            sens_high_sens_test_target=cls.SENS_HIGH_SENS_TEST,
            spec_high_sens_test_target=cls.SPEC_HIGH_SENS_TEST,
            seed=cls.SEED
        )
    
    @classmethod
    def well_calibrated_model(cls, y_true: np.ndarray) -> np.ndarray:
        """
        Generates predictions for the 'Well calibrated' continuous model.
        
        This model provides risk estimates that closely match the true probabilities,
        as described in the appendix "Continuous prediction models" section.
        """
        return generate_clinical_predictions(
            disease_status=y_true,
            model_type='well_calibrated',
            mean_z_negative=cls.MEAN_Z_NEG,
            mean_z_positive=cls.MEAN_Z_POS,
            sd_z=cls.SD_Z,
            beta1=cls.BETA1,
            beta0_approx=cls.BETA0,
            seed=cls.SEED
        )
    
    @classmethod
    def risk_overestimation_model(cls, y_true: np.ndarray) -> np.ndarray:
        """
        Generates predictions for the 'Overestimating risk' continuous model.
        
        This model systematically predicts higher probabilities than the true risk,
        as described in the appendix "Brier score and miscalibration" section.
        """
        return generate_clinical_predictions(
            disease_status=y_true,
            model_type='overpredict',
            mean_z_negative=cls.MEAN_Z_NEG,
            mean_z_positive=cls.MEAN_Z_POS,
            sd_z=cls.SD_Z,
            beta1=cls.BETA1,
            beta0_approx=cls.BETA0,
            gamma_overpredict=cls.GAMMA_OVER,
            seed=cls.SEED
        )
    
    @classmethod
    def risk_underestimation_model(cls, y_true: np.ndarray) -> np.ndarray:
        """
        Generates predictions for the 'Underestimating risk' continuous model.
        
        This model systematically predicts lower probabilities than the true risk,
        as described in the appendix "Brier score and miscalibration" section.
        """
        return generate_clinical_predictions(
            disease_status=y_true,
            model_type='underpredict',
            mean_z_negative=cls.MEAN_Z_NEG,
            mean_z_positive=cls.MEAN_Z_POS,
            sd_z=cls.SD_Z,
            beta1=cls.BETA1,
            beta0_approx=cls.BETA0,
            gamma_underpredict=cls.GAMMA_UNDER,
            seed=cls.SEED
        )
    
    @classmethod
    def severe_risk_underestimation_model(cls, y_true: np.ndarray) -> np.ndarray:
        """
        Generates predictions for the 'Severely underestimating risk' continuous model.
        
        This model severely underestimates the true risk, demonstrating an extreme case
        of miscalibration as described in the appendix Table 1.
        """
        return generate_clinical_predictions(
            disease_status=y_true,
            model_type='severe_underpredict',
            mean_z_negative=cls.MEAN_Z_NEG,
            mean_z_positive=cls.MEAN_Z_POS,
            sd_z=cls.SD_Z,
            beta1=cls.BETA1,
            beta0_approx=cls.BETA0,
            gamma_severe_underpredict=cls.GAMMA_SEVERE_UNDER,
            seed=cls.SEED
        )
    
    @classmethod
    def get_all_models(cls) -> Dict[str, callable]:
        """Returns a dictionary of all available clinical prediction models."""
        return {
            'Treat none (default negative)': cls.default_treat_none,
            'Treat all (default positive)': cls.default_treat_all,
            'High specificity test': cls.high_specificity_test,
            'High sensitivity test': cls.high_sensitivity_test,
            'Well calibrated model': cls.well_calibrated_model,
            'Risk overestimation model': cls.risk_overestimation_model,
            'Risk underestimation model': cls.risk_underestimation_model,
            'Severe risk underestimation model': cls.severe_risk_underestimation_model,
            'Calibrated binary (not in paper)': cls.calibrated_binary,
        }
    
    @classmethod
    def get_model(cls, model_name: str) -> callable:
        """Returns a specific clinical prediction model by name."""
        models = cls.get_all_models()
        if model_name not in models:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(models.keys())}")
        return models[model_name]