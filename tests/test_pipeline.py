"""
Overview of Pipeline Tests:

1. test_model_loads:
   - What it covers: Verifies that the final trained model pickle exists and loads successfully.
   - Why it matters: Ensures that the serialized model artifact is valid, compatible, and can be retrieved.

2. test_prediction_returns_probability:
   - What it covers: Passes a standard sample input claim dictionary through predict_claims and checks the output.
   - Why it matters: Validates that the prediction pipeline runs end-to-end, producing a float probability between 0 and 1.

3. test_high_risk_claim_scores_high:
   - What it covers: Modifies a claim to represent an obviously suspicious scenario (high claim ratio/amount, no police report, multiple bodily injuries, major severity) and verifies risk classification.
   - Why it matters: Confirms that the model's sensitivity aligns with basic fraud indicators, scoring high-risk inputs above 0.5.

4. test_feature_engineering_columns:
   - What it covers: Runs feature engineering and asserts that custom features exist.
   - Why it matters: Guarantees that derived metrics like ratios, ages, and intervals are correctly generated for modeling.

5. test_no_nulls_after_preprocessing:
   - What it covers: Checks if preprocessing outputs a dense array with zero NaN/null values.
   - Why it matters: Ensures data completeness and prevents downstream model failures due to missing data.
"""

import os
import joblib
import numpy as np
import pandas as pd
from src.config import MODEL_PATH
from src.model import load_model, predict_claims
from src.data import engineer_features, model_input_frame


def test_model_loads():
    # Assert model file exists on disk
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"
    # Assert model loads without error
    model = load_model()
    assert model is not None


def test_prediction_returns_probability(sample_claim):
    # Pass input dictionary as a single-row DataFrame
    df = pd.DataFrame([sample_claim])
    # Run the prediction pipeline
    result_df = predict_claims(df)
    
    # Assert fraud_probability exists and is between 0 and 1
    assert "fraud_probability" in result_df.columns
    prob = result_df.loc[0, "fraud_probability"]
    assert isinstance(prob, (float, int, np.floating))
    assert 0.0 <= float(prob) <= 1.0


def test_high_risk_claim_scores_high(sample_claim):
    # Create an obviously suspicious claim
    suspicious_claim = sample_claim.copy()
    suspicious_claim["total_claim_amount"] = 95000
    suspicious_claim["injury_claim"] = 25000
    suspicious_claim["property_claim"] = 20000
    suspicious_claim["vehicle_claim"] = 50000
    suspicious_claim["policy_annual_premium"] = 1000 # claim_ratio = 95
    suspicious_claim["police_report_available"] = "NO"
    suspicious_claim["bodily_injuries"] = 2
    suspicious_claim["incident_severity"] = "Major Damage"
    
    df = pd.DataFrame([suspicious_claim])
    result_df = predict_claims(df)
    prob = float(result_df.loc[0, "fraud_probability"])
    
    # Assert fraud risk is scored as high (> 0.5)
    assert prob > 0.5


def test_feature_engineering_columns(sample_claim):
    df = pd.DataFrame([sample_claim])
    # Run custom feature engineering function
    df_engineered = engineer_features(df)
    
    # Assert engineered columns exist in the output DataFrame
    assert "claim_ratio" in df_engineered.columns
    assert "vehicle_age" in df_engineered.columns
    assert "days_between_policy_incident" in df_engineered.columns


def test_no_nulls_after_preprocessing(sample_claim):
    df = pd.DataFrame([sample_claim])
    # Extract columns expected by the model
    X = model_input_frame(df)
    
    model = load_model()
    preprocessor = model.named_steps["preprocess"]
    
    # Transform raw columns through the preprocessing pipeline
    X_trans = preprocessor.transform(X)
    
    # Assert that no missing values exist in preprocessed data
    assert not np.isnan(X_trans).any()
