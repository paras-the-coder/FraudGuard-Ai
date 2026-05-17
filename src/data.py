from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    MODEL_INPUT_COLUMNS,
    TARGET_COLUMN,
)


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    return data.drop(columns=["_c39"], errors="ignore")


@st.cache_data(show_spinner=False)
def load_prepared_data() -> pd.DataFrame:
    return engineer_features(load_raw_data())


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    if TARGET_COLUMN in df.columns:
        if df[TARGET_COLUMN].dtype == "object":
            df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip().map({"Y": 1, "N": 0})

    if "total_claim_amount" in df.columns and "policy_annual_premium" in df.columns:
        premium = pd.to_numeric(df["policy_annual_premium"], errors="coerce").replace(0, pd.NA)
        df["claim_ratio"] = pd.to_numeric(df["total_claim_amount"], errors="coerce") / premium

    if "incident_date" in df.columns:
        incident_date = pd.to_datetime(df["incident_date"], errors="coerce", format="mixed")
        df["incident_year"] = incident_date.dt.year
    else:
        incident_date = None

    if "incident_year" in df.columns and "auto_year" in df.columns:
        df["vehicle_age"] = pd.to_numeric(df["incident_year"], errors="coerce") - pd.to_numeric(
            df["auto_year"], errors="coerce"
        )

    if "policy_bind_date" in df.columns and incident_date is not None:
        policy_date = pd.to_datetime(df["policy_bind_date"], errors="coerce", format="mixed")
        df["days_between_policy_incident"] = (incident_date - policy_date).dt.days

    if "policy_csl" in df.columns and (
        "csl_per_person" not in df.columns or "csl_per_accident" not in df.columns
    ):
        csl = df["policy_csl"].astype(str).str.split("/", expand=True)
        if csl.shape[1] >= 2:
            df["csl_per_person"] = pd.to_numeric(csl[0], errors="coerce")
            df["csl_per_accident"] = pd.to_numeric(csl[1], errors="coerce")

    return df


def model_input_frame(data: pd.DataFrame) -> pd.DataFrame:
    df = engineer_features(data)
    missing = [column for column in MODEL_INPUT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns after feature engineering: " + ", ".join(missing)
        )
    return df[MODEL_INPUT_COLUMNS].copy()


@st.cache_data(show_spinner=False)
def get_filter_options() -> dict[str, list[str]]:
    data = load_raw_data()
    return {
        column: sorted(data[column].dropna().astype(str).unique().tolist())
        for column in CATEGORICAL_FEATURES + ["policy_csl", "auto_make", "auto_model"]
        if column in data.columns
    }


def get_example_claim(kind: str) -> pd.Series:
    data = load_prepared_data()
    if kind == "fraud" and TARGET_COLUMN in data.columns:
        candidates = data[data[TARGET_COLUMN] == 1].sort_values("claim_ratio", ascending=False)
    elif TARGET_COLUMN in data.columns:
        candidates = data[data[TARGET_COLUMN] == 0].sort_values("claim_ratio", ascending=True)
    else:
        candidates = data

    if candidates.empty:
        candidates = data
    return candidates.iloc[0]
