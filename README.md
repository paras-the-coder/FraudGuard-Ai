# FraudGuard AI — Insurance Claim Fraud Detection System

FraudGuard AI is a Machine Learning-powered insurance claim fraud detection system designed to identify suspicious automobile insurance claims and support fraud investigation workflows.

The project combines:

* Machine Learning
* Business-rule intelligence
* Interactive analytics dashboards
* Risk scoring and reporting

---

# Application Preview

![FraudGuard AI Home Page](assets/screenshots/home.png)

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

The final model selected for this project is **Logistic Regression**.

Although multiple models were tested, including **Random Forest** and **XGBoost**, Logistic Regression was chosen as the best production-ready model because it achieved the strongest balance for fraud detection, especially on **recall**.

| Model | Accuracy | Recall | F1 Score | F2 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.845 | 0.857 | 0.730 | 0.802 | 0.849 |
| Random Forest | 0.830 | 0.755 | 0.685 | 0.726 | 0.833 |
| XGBoost | 0.855 | 0.776 | 0.724 | 0.754 | 0.860 |

## Why Logistic Regression Was Selected

In insurance fraud detection, **missing a fraudulent claim is more costly than sending a legitimate claim for review**. Because of this, the model was selected primarily based on **Recall** and **F2 Score**, not accuracy alone.

Logistic Regression achieved the highest recall, detecting about **86% of fraudulent claims** in the test set. It also had the best F2 Score, which gives more importance to recall than precision. This makes it better suited for a pre-claim fraud screening system where the goal is to catch as many suspicious claims as possible before payout.

XGBoost had slightly higher accuracy and ROC-AUC, but it missed more fraud cases than Logistic Regression. Random Forest also performed well, but its recall was lower. Therefore, Logistic Regression was chosen as the final model because it is:

- Better at catching fraud cases
- Easier to interpret
- Faster and simpler to deploy

The final model is not intended to automatically reject claims. Instead, it acts as a **fraud risk screening tool** that helps prioritize claims for manual or SIU review.

### Model Performance Visuals

![Model Performance Comparison](assets/screenshots/model_comparison.png)

---

## Explainability & Risk Interpretation

FraudGuard AI includes explainable prediction outputs such as visual risk drivers and feature contribution analysis.

These explanations show how the trained Logistic Regression model weighted different features during prediction based on learned statistical patterns in the dataset.

The explanations are intended to support fraud-risk interpretation and investigation workflows, but they should not be treated as direct legal or causal proof of fraud.

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

![Prediction Demo Form](assets/screenshots/prediction-form.png)

---

# Tech Stack

- **Programming & Data Processing**: Python,Pandas,NumPy

- **Machine Learning**: Scikit-learn, Imbalanced-learn / SMOTE, Logistic Regression, Random Forest, XGBoost, Joblib for model serialization

- **Data Visualization**: Plotly, Matplotlib, Seaborn

- **Web Application**: Streamlit, Custom CSS

- **Model Explainability & Reporting**: Logistic Regression coefficient-based feature contributions, Business-rule fraud signals, HTML and PDF report generation

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
Logistic Regression Pipeline
(StandardScaler + OneHotEncoder + SMOTE + LR)
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
- The dataset is limited in size and scope. Real-world insurance fraud systems require significantly larger and more diverse datasets.
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

Possible future enhancements:

* SHAP-based explainability
* Ensemble learning models
* Real-time API integration
* Cloud deployment
* User authentication
* Claim history tracking
* Production-grade monitoring
* The UI and the report generation can be improved further

---

# What I Learned

* Built an end-to-end ML pipeline with SMOTE inside an imbalanced-learn `Pipeline` to prevent data leakage during cross-validation
* Used `make_scorer(fbeta_score, beta=2)` as the `RandomizedSearchCV` scoring function so hyperparameter search optimized directly for fraud recall, not accuracy
* Designed business-rule post-processing on top of model probability scores to capture fraud signals that statistical features alone cannot express
* Implemented coefficient-based feature contribution explanations for logistic regression as a lightweight, production-compatible alternative to SHAP
* Learned why precision-recall tradeoffs matter more than accuracy in imbalanced, high-cost classification problems like insurance fraud

---

# Conclusion

FraudGuard AI demonstrates how Machine Learning and business-rule intelligence can work together to support insurance fraud investigation workflows.

The project focuses not only on prediction accuracy, but also on:

* explainability
* fraud reasoning
* business value
* investigation support

making it closer to a realistic fraud analytics application rather than a simple ML notebook project.
