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


def predict_claims(data: pd.DataFrame, model=None) -> pd.DataFrame:
    model = model or load_model()
    X = model_input_frame(data)
    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

    result = data.copy()
    result["fraud_probability"] = probabilities
    result["prediction"] = ["Fraud" if value == 1 else "Legitimate" for value in predictions]
    result["risk_level"] = [risk_label(float(value))[0] for value in probabilities]
    return result
