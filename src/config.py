from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "insurance_claims.csv"
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
STYLE_PATH = BASE_DIR / "assets" / "styles.css"

TARGET_COLUMN = "fraud_reported"

NUMERIC_FEATURES = [
    "months_as_customer",
    "age",
    "policy_deductable",
    "policy_annual_premium",
    "incident_hour_of_the_day",
    "number_of_vehicles_involved",
    "bodily_injuries",
    "witnesses",
    "auto_year",
    "total_claim_amount",
    "claim_ratio",
    "incident_year",
    "vehicle_age",
    "csl_per_person",
    "csl_per_accident",
    "days_between_policy_incident",
]

CATEGORICAL_FEATURES = [
    "policy_state",
    "insured_sex",
    "insured_education_level",
    "insured_occupation",
    "insured_hobbies",
    "insured_relationship",
    "incident_type",
    "collision_type",
    "incident_severity",
    "authorities_contacted",
    "incident_state",
    "incident_city",
    "property_damage",
    "police_report_available",
]

DATE_FEATURES = ["policy_bind_date", "incident_date"]

MODEL_INPUT_COLUMNS = [
    "months_as_customer",
    "age",
    "policy_bind_date",
    "policy_state",
    "policy_deductable",
    "policy_annual_premium",
    "insured_sex",
    "insured_education_level",
    "insured_occupation",
    "insured_hobbies",
    "insured_relationship",
    "incident_date",
    "incident_type",
    "collision_type",
    "incident_severity",
    "authorities_contacted",
    "incident_state",
    "incident_city",
    "incident_hour_of_the_day",
    "number_of_vehicles_involved",
    "property_damage",
    "bodily_injuries",
    "witnesses",
    "police_report_available",
    "auto_year",
    "total_claim_amount",
    "claim_ratio",
    "incident_year",
    "vehicle_age",
    "csl_per_person",
    "csl_per_accident",
    "days_between_policy_incident",
]

MODEL_COMPARISON = {
    "XGBoost (Balanced)": {
        "Accuracy": 0.8300,
        "Recall": 0.7350,
        "F1 Score": 0.6790,
        "F2 Score": 0.7110,
        "ROC-AUC": 0.7780,
    },
    "Logistic Regression": {
        "Accuracy": 0.7400,
        "Recall": 0.7550,
        "F1 Score": 0.5870,
        "F2 Score": 0.6780,
        "ROC-AUC": 0.7960,
    },
    "Random Forest": {
        "Accuracy": 0.8300,
        "Recall": 0.7350,
        "F1 Score": 0.6790,
        "F2 Score": 0.7110,
        "ROC-AUC": 0.7780,
    },
}

LOW_RISK_THRESHOLD = 0.35
HIGH_RISK_THRESHOLD = 0.65
