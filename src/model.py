from __future__ import annotations

import warnings

import joblib
import pandas as pd
import streamlit as st

from src.config import HIGH_RISK_THRESHOLD, LOW_RISK_THRESHOLD, MODEL_PATH
from src.data import model_input_frame


@st.cache_resource(show_spinner=False)
def load_model():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(MODEL_PATH)

    estimator = getattr(model, "named_steps", {}).get("model")
    if estimator is not None and not hasattr(estimator, "multi_class"):
        estimator.multi_class = "auto"
    return model


def risk_label(probability: float) -> tuple[str, str]:
    if probability >= HIGH_RISK_THRESHOLD:
        return "High", "risk-high"
    if probability >= LOW_RISK_THRESHOLD:
        return "Medium", "risk-medium"
    return "Low", "risk-low"


def predict_claims(data: pd.DataFrame, model=None, apply_business_rules: bool = True) -> pd.DataFrame:
    from src.rules import business_rule_adjustment

    model = model or load_model()
    X = model_input_frame(data)
    base_probabilities = model.predict_proba(X)[:, 1]

    result = data.copy()
    result["base_fraud_probability"] = base_probabilities

    if apply_business_rules:
        final_probabilities = []
        for i in range(len(result)):
            row = result.iloc[i]
            base_p = float(base_probabilities[i])
            adj_p, _ = business_rule_adjustment(row, base_p)
            final_probabilities.append(adj_p)
    else:
        final_probabilities = [float(p) for p in base_probabilities]

    predictions = ["Fraud" if p >= 0.5 else "Legitimate" for p in final_probabilities]
    risk_levels = [risk_label(p)[0] for p in final_probabilities]

    result["fraud_probability"] = final_probabilities
    result["prediction"] = predictions
    result["risk_level"] = risk_levels
    return result
