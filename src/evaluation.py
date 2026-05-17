from __future__ import annotations

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, fbeta_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.config import TARGET_COLUMN
from src.data import load_prepared_data, model_input_frame
from src.model import load_model


@st.cache_data(show_spinner=False)
def get_lr_evaluation() -> dict:
    data = load_prepared_data()
    y = data[TARGET_COLUMN].astype(int)
    X = model_input_frame(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    _ = X_train
    model = load_model()
    y_pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "F2 Score": fbeta_score(y_test, y_pred, beta=2),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }
    return {
        "y_true": pd.Series(y_test).reset_index(drop=True),
        "y_pred": pd.Series(y_pred).reset_index(drop=True),
        "probabilities": pd.Series(probabilities).reset_index(drop=True),
        "metrics": metrics,
    }
