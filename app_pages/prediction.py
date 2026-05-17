from __future__ import annotations

import pandas as pd
import streamlit as st

from src.charts import contribution_bar, metric_gauge
from src.config import MODEL_INPUT_COLUMNS
from src.data import get_example_claim, get_filter_options, model_input_frame
from src.explain import top_feature_contributions
from src.model import load_model, predict_claims, risk_label
from src.report import build_html_report, build_pdf_report, data_integrity_flags, recommended_action
from src.rules import business_rule_adjustment, generate_internal_dates, validate_claim_before_prediction


def _base_manual_values() -> pd.Series:
    return get_example_claim("legitimate")


def _selectbox(label: str, options: list[str], value: str, key: str):
    index = options.index(value) if value in options else 0
    return st.selectbox(
        label,
        options,
        index=index,
        key=key,
        format_func=lambda option: "Unknown" if str(option).strip() == "?" else option,
    )


def _money_text_input(label: str, value: int | float, key: str) -> str:
    return st.text_input(label, value=f"{float(value):.2f}", key=key)


def _parse_money(value: str, label: str) -> float:
    cleaned = value.replace("$", "").replace(",", "").strip()
    try:
        parsed = float(cleaned)
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid number.") from exc
    if parsed < 0:
        raise ValueError(f"{label} cannot be negative.")
    return parsed


def _hidden_pipeline_defaults(values: pd.Series) -> dict:
    csl = str(values.get("policy_csl", "250/500"))
    csl_parts = csl.split("/")
    csl_per_person = int(csl_parts[0]) if len(csl_parts) > 0 and csl_parts[0].isdigit() else 250
    csl_per_accident = int(csl_parts[1]) if len(csl_parts) > 1 and csl_parts[1].isdigit() else 500
    return {
        "insured_hobbies": str(values.get("insured_hobbies", "reading")),
        "csl_per_person": csl_per_person,
        "csl_per_accident": csl_per_accident,
    }


def _business_explanation(contributors: pd.DataFrame, prediction: str) -> str:
    direction = "Raises fraud risk" if prediction == "Fraud" else "Lowers fraud risk"
    selected = contributors[contributors["Direction"] == direction].head(4)
    if selected.empty:
        selected = contributors.head(4)

    heading = "This claim was flagged because:" if prediction == "Fraud" else "This claim appears lower risk because:"
    bullets = "".join(f"<li>{row.Feature}</li>" for row in selected.itertuples(index=False))
    return f"""
    <div class="reason-card">
        <div class="reason-title">{heading}</div>
        <ul>{bullets}</ul>
    </div>
    """


def _action_card(probability: float) -> str:
    action, level = recommended_action(probability)
    return f"""
    <div class="action-card action-{level}">
        <span>Recommended Action</span>
        <strong>{action}</strong>
    </div>
    """


def _data_flags_card(flags: list[tuple[str, str]]) -> str:
    items = "".join(f"<li class='flag-{level}'>{text}</li>" for level, text in flags)
    return f"""
    <div class="flags-card">
        <span>Data Integrity Flags</span>
        <ul>{items}</ul>
    </div>
    """


def _rule_signals_card(signals: list[str]) -> str:
    items = "".join(f"<li>{signal}</li>" for signal in signals)
    return f"""
    <div class="flags-card">
        <span>Additional Fraud Risk Signals</span>
        <ul>{items}</ul>
    </div>
    """


def manual_form() -> pd.DataFrame | None:
    options = get_filter_options()
    values = _base_manual_values()
    hidden = _hidden_pipeline_defaults(values)
    st.caption("The form is prefilled with a realistic claim profile. Adjust the business fields and run a risk screen.")

    with st.form("manual_prediction_form"):
        st.markdown("#### Claim Intake")
        tabs = st.tabs(["Claim Financials", "Incident Details", "Customer & Policy", "Vehicle & Location"])

        with tabs[0]:
            c1, c2 = st.columns(2)
            with c1:
                total_claim_raw = _money_text_input("Total Claim Amount ($)", values["total_claim_amount"], "total_claim_text")
                injury_claim_raw = _money_text_input("Injury Claim ($)", values.get("injury_claim", 0), "injury_claim_text")
                property_claim_raw = _money_text_input("Property Claim ($)", values.get("property_claim", 0), "property_claim_text")
            with c2:
                vehicle_claim_raw = _money_text_input("Vehicle Claim ($)", values.get("vehicle_claim", 0), "vehicle_claim_text")
                policy_premium_raw = _money_text_input(
                    "Annual Premium ($)",
                    values["policy_annual_premium"],
                    "policy_premium_text",
                )
                deductibles = [500, 1000, 2000]
                policy_deductable = st.selectbox(
                    "Policy Deductible ($)",
                    deductibles,
                    index=deductibles.index(int(values["policy_deductable"])),
                    key="policy_deductable",
                )

        with tabs[1]:
            c3, c4 = st.columns(2)
            with c3:
                severity = _selectbox("Incident Severity", options["incident_severity"], values["incident_severity"], "severity")
                incident_type = _selectbox("Incident Type", options["incident_type"], values["incident_type"], "incident_type")
                collision_type = _selectbox("Collision Type", options["collision_type"], values["collision_type"], "collision_type")
                authorities = _selectbox("Authorities Contacted", options["authorities_contacted"], values["authorities_contacted"], "authorities")
                property_damage = _selectbox("Property Damage", options["property_damage"], values["property_damage"], "property_damage")
            with c4:
                incident_hour = st.slider("Incident Hour", 0, 23, int(values["incident_hour_of_the_day"]), key="incident_hour")
                vehicles = st.slider("Vehicles Involved", 1, 4, int(values["number_of_vehicles_involved"]), key="vehicles")
                bodily_injuries = st.slider("Bodily Injuries", 0, 2, int(values["bodily_injuries"]), key="bodily_injuries")
                witnesses = st.slider("Witnesses", 0, 3, int(values["witnesses"]), key="witnesses")
                police_report = _selectbox(
                    "Police Report Available",
                    options["police_report_available"],
                    values["police_report_available"],
                    "police_report",
                )

        with tabs[2]:
            c5, c6 = st.columns(2)
            with c5:
                age = st.number_input("Customer Age", 18, 90, int(values["age"]), key="age")
                months = st.number_input(
                    "Months as Customer", 0, 520, int(values["months_as_customer"]), key="months_as_customer"
                )
                policy_state = _selectbox("Policy State", options["policy_state"], values["policy_state"], "policy_state")
            with c6:
                insured_sex = _selectbox("Insured Sex", options["insured_sex"], values["insured_sex"], "insured_sex")
                education = _selectbox(
                    "Education", options["insured_education_level"], values["insured_education_level"], "education"
                )
                occupation = _selectbox("Occupation", options["insured_occupation"], values["insured_occupation"], "occupation")
                relationship = _selectbox(
                    "Relationship", options["insured_relationship"], values["insured_relationship"], "relationship"
                )

        with tabs[3]:
            c7, c8 = st.columns(2)
            with c7:
                auto_year = st.number_input("Vehicle Year", 1990, 2026, int(values["auto_year"]), key="auto_year")
                auto_make = _selectbox("Vehicle Make", options["auto_make"], values.get("auto_make", ""), "auto_make")
                auto_model = _selectbox("Vehicle Model", options["auto_model"], values.get("auto_model", ""), "auto_model")
            with c8:
                incident_state = _selectbox("Incident State", options["incident_state"], values["incident_state"], "incident_state")
                incident_city = _selectbox("Incident City", options["incident_city"], values["incident_city"], "incident_city")

        submitted = st.form_submit_button("Predict Claim Risk", type="primary", use_container_width=True)

    if not submitted:
        return None

    try:
        total_claim = _parse_money(total_claim_raw, "Total Claim Amount")
        injury_claim = _parse_money(injury_claim_raw, "Injury Claim")
        property_claim = _parse_money(property_claim_raw, "Property Claim")
        vehicle_claim = _parse_money(vehicle_claim_raw, "Vehicle Claim")
        policy_premium = _parse_money(policy_premium_raw, "Annual Premium")
        if policy_premium == 0:
            raise ValueError("Annual Premium must be greater than zero.")
    except ValueError as exc:
        st.error(str(exc))
        return None

    policy_bind, incident_date = generate_internal_dates(months, auto_year)
    incident_year = incident_date.year
    claim = {
        "policy_number": values.get("policy_number", "Demo claim"),
        "months_as_customer": months,
        "age": age,
        "policy_bind_date": str(policy_bind),
        "policy_state": policy_state,
        "policy_deductable": policy_deductable,
        "policy_annual_premium": policy_premium,
        "insured_sex": insured_sex,
        "insured_education_level": education,
        "insured_occupation": occupation,
        "insured_hobbies": hidden["insured_hobbies"],
        "insured_relationship": relationship,
        "incident_date": str(incident_date),
        "incident_type": incident_type,
        "collision_type": collision_type,
        "incident_severity": severity,
        "authorities_contacted": authorities,
        "incident_state": incident_state,
        "incident_city": incident_city,
        "incident_hour_of_the_day": incident_hour,
        "number_of_vehicles_involved": vehicles,
        "property_damage": property_damage,
        "bodily_injuries": bodily_injuries,
        "witnesses": witnesses,
        "police_report_available": police_report,
        "auto_year": auto_year,
        "auto_make": auto_make,
        "auto_model": auto_model,
        "total_claim_amount": total_claim,
        "injury_claim": injury_claim,
        "property_claim": property_claim,
        "vehicle_claim": vehicle_claim,
        "claim_ratio": total_claim / policy_premium if policy_premium else 0,
        "incident_year": incident_year,
        "vehicle_age": incident_year - auto_year,
        "csl_per_person": hidden["csl_per_person"],
        "csl_per_accident": hidden["csl_per_accident"],
        "days_between_policy_incident": (incident_date - policy_bind).days,
    }
    validation_errors = validate_claim_before_prediction(claim)
    if validation_errors:
        for error in validation_errors:
            st.error(error)
        return None

    frame = pd.DataFrame([claim])
    return frame


def show_prediction(frame: pd.DataFrame) -> None:
    model = load_model()
    prepared = model_input_frame(frame)
    result = predict_claims(prepared, model=model).iloc[0]
    base_probability = float(result["fraud_probability"])
    report_claim = frame.iloc[0]
    probability, rule_signals = business_rule_adjustment(report_claim, base_probability)
    prediction = "Fraud" if probability >= 0.5 else "Legitimate"
    risk, risk_class = risk_label(probability)
    contributors = top_feature_contributions(model, prepared.iloc[[0]])
    action, _ = recommended_action(probability)
    flags = data_integrity_flags(report_claim)

    st.markdown("### Prediction Results")
    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.plotly_chart(metric_gauge(probability), use_container_width=True)
    with c2:
        st.markdown(
            f"""
            <div class="result-card">
                <span>Risk Level</span>
                <strong class="{risk_class}">{risk}</strong>
            </div>
            <div class="result-card">
                <span>Prediction</span>
                <strong>{prediction}</strong>
            </div>
            <div class="result-card">
                <span>Review Recommendation</span>
                <strong>{action}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(_action_card(probability), unsafe_allow_html=True)
    st.markdown(_data_flags_card(flags), unsafe_allow_html=True)
    if rule_signals:
        st.markdown(_rule_signals_card(rule_signals), unsafe_allow_html=True)
    st.markdown(_business_explanation(contributors, prediction), unsafe_allow_html=True)

    st.markdown("#### Visual Risk Drivers")
    st.caption("Coefficient-based contribution chart from the Logistic Regression model. Red bars push the fraud score up; green bars push it down.")
    st.plotly_chart(contribution_bar(contributors), use_container_width=True)

    st.markdown("#### Top Contributing Features")
    st.dataframe(contributors, use_container_width=True, hide_index=True)

    report = build_html_report(report_claim, probability, prediction, risk, contributors, rule_signals)
    pdf_report = build_pdf_report(report_claim, probability, prediction, risk, contributors, rule_signals)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download HTML Report",
            data=report,
            file_name="fraudguard_prediction_report.html",
            mime="text/html",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Download PDF Report",
            data=pdf_report,
            file_name="fraudguard_prediction_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


def batch_upload() -> None:
    uploaded = st.file_uploader("Upload a CSV with raw insurance claim columns", type=["csv"])
    if uploaded is None:
        st.info("Tip: `data/insurance_claims.csv` is a valid upload template.")
        return

    try:
        data = pd.read_csv(uploaded)
        results = predict_claims(data)
    except Exception as exc:
        st.error(f"Unable to score this file: {exc}")
        return

    st.success(f"Scored {len(results):,} claims.")
    visible_columns = [column for column in MODEL_INPUT_COLUMNS if column in results.columns]
    st.dataframe(
        results[visible_columns + ["fraud_probability", "risk_level", "prediction"]].sort_values(
            "fraud_probability", ascending=False
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download Batch Predictions",
        results.to_csv(index=False),
        file_name="fraudguard_batch_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render() -> None:
    st.title("Fraud Prediction Demo")
    st.caption("Score a single claim manually or upload a CSV for batch review.")

    mode = st.segmented_control("Prediction Mode", ["Fill Form", "Upload CSV"], default="Fill Form")
    if mode == "Fill Form":
        frame = manual_form()
        if frame is not None:
            with st.spinner("Analyzing claim signals..."):
                show_prediction(frame)
                st.toast("Prediction complete")
    else:
        batch_upload()
