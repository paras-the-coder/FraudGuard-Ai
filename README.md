# FraudGuard AI — Insurance Claim Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

🔗 **[Live Demo → fraudguard-ai.streamlit.app](https://fraudguard-ai-ljpybjziicpkymlpuzbsus.streamlit.app/)**

FraudGuard AI is a Machine Learning-powered insurance claim fraud detection system designed to identify suspicious automobile insurance claims and support fraud investigation workflows.

The project combines:

* Machine Learning
* Business-rule intelligence
* Interactive analytics dashboards
* Risk scoring and reporting

---

# Application Preview

<img width="1890" height="828" alt="image" src="https://github.com/user-attachments/assets/6c99c472-563e-446e-88e4-c8c71f6044e5" />

---

# Problem Statement

Insurance fraud causes significant financial losses for insurance companies every year. Fraudulent claims may include:

* fake accidents
* exaggerated injury claims
* inflated repair costs
* staged collisions
* false theft reports

Manually reviewing every claim is expensive and time-consuming.

FraudGuard AI helps identify high-risk claims early, prioritize investigations, and improve fraud screening efficiency.

---

# Insurance Domain

This project focuses on:

```text
Automobile Insurance Claim Fraud Detection
```

The dataset contains automobile accident and insurance claim records including:

* customer information
* policy details
* accident details
* claim financials
* vehicle information
* fraud labels

Target Variable:

```text
Fraudulent Claim vs Legitimate Claim
```

---

#  Features Used

### Claim Financial Features

* total_claim_amount
* injury_claim
* vehicle_claim
* property_claim
* policy_annual_premium
* policy_deductable

### Incident Features

* incident_type
* collision_type
* incident_severity
* incident_hour_of_the_day
* number_of_vehicles_involved
* bodily_injuries
* witnesses
* police_report_available
* property_damage

### Customer & Policy Features

* age
* months_as_customer
* insured_occupation
* insured_education_level
* insured_relationship
* policy_state

### Vehicle & Location Features

* auto_make
* auto_model
* auto_year
* incident_state
* incident_city

---

# Feature Engineering

Additional engineered features were created to improve fraud detection performance:

* claim_ratio
* incident_year
* vehicle_age
* days_between_policy_incident
* csl_per_person
* csl_per_accident

These engineered features help capture suspicious claim behavior and fraud-related patterns.

---

# Model Performance

The final model selected for this project is **XGBoost (Balanced & Tuned)**.

After dropping noisy features like `insured_hobbies`, multiple algorithms were evaluated using 5-fold cross-validation and hyperparameter tuning. **XGBoost (Balanced)** was chosen as the primary production model due to its high overall accuracy, balanced F2 score, and superior handling of non-linear feature interactions.

| Model | Accuracy | Recall | F1 Score | F2 Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| **XGBoost (Balanced)** ✓ | **0.830** | 0.735 | **0.679** | **0.711** | 0.778 |
| Random Forest | **0.830** | 0.735 | **0.679** | **0.711** | 0.778 |
| Logistic Regression | 0.740 | **0.755** | 0.587 | 0.678 | **0.796** |

## Why XGBoost Was Selected

In insurance fraud detection, catching suspicious claims requires balancing accuracy, fraud recall, and feature interaction modeling. **XGBoost (Balanced)** was selected because:

- **Higher Overall Accuracy & F1/F2 Balance:** Achieves 83.0% accuracy with a solid 0.711 F2 score, outperforming Logistic Regression's overall classification precision.
- **Handles Non-Linear Feature Interactions:** Captures complex multi-variable risk signals (such as Major Damage combined with high claim ratios or short tenure) that simple linear models miss.
- **Tree-Based SHAP Explainability:** Pairs directly with `shap.Explainer` (TreeExplainer) to provide exact instance-level feature attributions without linear model assumptions.

The final model is not intended to automatically reject claims. Instead, it acts as a **fraud risk screening tool** that helps prioritize claims for manual or SIU review.

### Model Performance Visuals

<img width="1408" height="555" alt="image" src="https://github.com/user-attachments/assets/4b88c3f6-33ab-41bf-bb0c-31cb31a0a48e" />


---

## Explainability & Risk Interpretation

FraudGuard AI uses **SHAP (SHapley Additive exPlanations)** via `shap.Explainer` (TreeExplainer)
to generate per-prediction, instance-level explanations for the XGBoost model.

Each prediction shows which features pushed the fraud probability up or down 
for that specific claim — providing claim-specific reasoning.

SHAP analysis revealed that `incident_severity_major_damage` is the strongest 
global fraud signal. Dropping non-causal features like `insured_hobbies` removed spurious 
correlations, resulting in clean, domain-backed SHAP explanations.

### Fraud Rate by Incident Severity

| Incident Severity | Total Claims | Fraud Claims | Fraud Rate (%) |
|---|:---:|:---:|:---:|
| **Major Damage** | 276 | 167 | **60.5%** |
| **Total Loss** | 280 | 36 | 12.9% |
| **Minor Damage** | 354 | 38 | **10.7%** |
| **Trivial Damage** | 90 | 6 | 6.7% |

> **Why Major Damage Dominates:** Claims reporting **Major Damage** have a **60.5% fraud rate**, compared to only **10.7%** for Minor Damage. This massive 6x risk contrast justifies why XGBoost assigns its highest tree-split weightings to major incident severity.

Explanations support fraud-risk interpretation workflows but should not be treated 
as legal or causal proof of fraud.

---

# Hybrid Fraud Detection Logic

FraudGuard AI combines:

1. Machine Learning probability scoring
2. Additional business-rule fraud analysis (post-model business-rule adjustment)

Example business-rule signals:

* unusually high injury claims
* missing police reports
* inconsistent claim breakdowns
* suspicious timing patterns
* inflated claim-to-premium ratios

This improves fraud-screening realism and operational interpretability.

---

# Application Features

* Multi-page Streamlit dashboard
* Fraud probability scoring
* Risk classification
* Explainable AI outputs
* Visual risk drivers
* Downloadable HTML investigation reports

### Prediction Demo

<img width="1904" height="826" alt="Screenshot 2026-05-17 132935" src="https://github.com/user-attachments/assets/d02d8d68-4224-420f-a065-b24092f7cc78" />


---

# Tech Stack

- **Programming & Data Processing**: Python,Pandas,NumPy

- **Machine Learning**: Scikit-learn, Imbalanced-learn / SMOTE, Logistic Regression, Random Forest, XGBoost, Joblib for model serialization

- **Data Visualization**: Plotly, Matplotlib, Seaborn

- **Web Application**: Streamlit, Custom CSS

- **Model Explainability & Reporting**: SHAP (TreeExplainer), per-prediction feature contributions, business-rule fraud signals, HTML and PDF investigation report generation

---

### Deployment

* Streamlit Community Cloud

---

# 📁 Project Structure

```text
FraudGuard-AI/
│
├── app.py
├── app_pages/
├── src/
├── models/
├── data/
├── assets/
├── notebooks/
├── tests/           (pytest suite for data pipeline and model validation)
├── requirements.txt
└── README.md
```

---

# System Flow
 
```text
Raw Insurance Claim
        │
        ▼
Feature Engineering
(claim_ratio, vehicle_age, days_between_policy_incident, csl splits)
        │
        ▼
Balanced XGBoost Pipeline
(StandardScaler + OneHotEncoder + scale_pos_weight + XGBoost)
        │
        ▼
Base Fraud Probability Score
        │
        ▼
Business Rule Adjustment
(injury-to-damage ratio, missing police report, short tenure, claim dominance)
        │
        ▼
Final Risk Score + Signals
        │
        ├── Low  (<35%)  → Auto-approve
        ├── Medium (35–65%) → Manual review
        └── High  (>65%)  → Escalate to SIU
```
 
---

# Limitations

FraudGuard AI is a fraud-screening support tool, not a final fraud decision system.

- The predictions are based on probabilities, so the model may sometimes predict fraud incorrectly or miss some fraud cases.
- The dataset is synthetic and small in size. Real-world insurance fraud systems require significantly larger and more diverse datasets.
- Human investigation is still required for final fraud decisions.
- The deployed model prioritizes explainability and recall over maximum predictive performance.
- Some business-rule checks added in the app are manually designed and may not fully represent real insurance company workflows.
- Model performance is based on historical labeled data and may not perform well on completely new or evolving fraud patterns.
- Some input fields are simplified or internally generated for demo purposes.
- The model can produce false positives and false negatives.
- Feature contribution explanations only show which factors influenced the prediction. They should not be treated as legal proof of fraud.
- The app does not include production monitoring, drift detection, or real-time claim system integration.

FraudGuard AI should be viewed as a fraud risk assessment and investigation support tool rather than a fully automated fraud detection system.

---

# Future Improvements

- FastAPI backend for real-time claim scoring API
- Ensemble learning models (stacking LR + XGBoost)
- Real-time API integration with claims management systems
- User authentication and role-based access
- Claim history tracking and trend analysis
- Production-grade monitoring and model drift detection
- Larger, more diverse training dataset to reduce spurious correlations

---

# What I Learned

* Built an end-to-end ML pipeline using XGBoost with `scale_pos_weight` class balancing inside a scikit-learn `Pipeline` to prevent data leakage
* Used `make_scorer(fbeta_score, beta=2)` with `RandomizedSearchCV` so hyperparameter tuning optimized directly for fraud recall and F2 score
* Integrated `shap.Explainer` (TreeExplainer) to generate instance-level feature attributions for non-linear tree models
* Designed business-rule post-processing on top of model probability scores to capture domain fraud signals that statistical features alone cannot express
* Identified and removed spurious feature correlations (`insured_hobbies`) to ensure model explainability relies purely on genuine risk factors

---

# Conclusion

FraudGuard AI demonstrates how Machine Learning and business-rule intelligence can work together to support insurance fraud investigation workflows.

The project focuses not only on prediction accuracy, but also on:

* explainability
* fraud reasoning
* business value
* investigation support

making it closer to a realistic fraud analytics application rather than a simple ML notebook project.
