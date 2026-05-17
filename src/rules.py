from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def _num(value, default: float = 0.0) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return default
    return float(parsed)


def generate_internal_dates(months_as_customer: int | float, auto_year: int | float) -> tuple[date, date]:
    incident_year = max(2015, int(auto_year))
    incident_date = date(incident_year, 1, 31)
    tenure_days = max(int(float(months_as_customer) * 30.4375), 0)
    policy_bind_date = incident_date - timedelta(days=tenure_days)
    return policy_bind_date, incident_date


def validate_claim_before_prediction(claim: dict) -> list[str]:
    errors: list[str] = []
    total_claim = float(claim.get("total_claim_amount", 0))
    injury_claim = float(claim.get("injury_claim", 0))
    property_claim = float(claim.get("property_claim", 0))
    vehicle_claim = float(claim.get("vehicle_claim", 0))
    policy_premium = float(claim.get("policy_annual_premium", 0))
    auto_year = int(claim.get("auto_year", 0))
    incident_year = int(claim.get("incident_year", 0))
    days_between = int(claim.get("days_between_policy_incident", 0))

    if policy_premium <= 0:
        errors.append("Annual Premium must be greater than zero.")
    if total_claim <= 0:
        errors.append("Total Claim Amount must be greater than zero.")
    if any(value < 0 for value in [injury_claim, property_claim, vehicle_claim]):
        errors.append("Claim component amounts cannot be negative.")

    component_total = injury_claim + property_claim + vehicle_claim
    if abs(total_claim - component_total) > 100:
        errors.append(
            "Total Claim Amount should match Injury + Property + Vehicle claim amounts before prediction."
        )

    if auto_year > incident_year:
        errors.append("Vehicle Year cannot be later than the internally generated incident year.")
    if days_between < 0:
        errors.append("Policy bind date cannot be after the incident date.")

    return errors


def business_rule_adjustment(claim: pd.Series, base_probability: float) -> tuple[float, list[str]]:
    adjusted_probability = float(base_probability)
    signals: list[str] = []

    injury_claim = _num(claim.get("injury_claim", 0))
    property_claim = _num(claim.get("property_claim", 0))
    vehicle_claim = _num(claim.get("vehicle_claim", 0))
    total_claim = _num(claim.get("total_claim_amount", 0))
    months = _num(claim.get("months_as_customer", 0))
    severity = str(claim.get("incident_severity", "")).strip().lower()
    police_report = str(claim.get("police_report_available", "")).strip().upper()
    witnesses = int(_num(claim.get("witnesses", 0)))

    physical_damage = property_claim + vehicle_claim
    injury_to_damage_ratio = injury_claim / max(physical_damage, 1)
    injury_share = injury_claim / max(total_claim, 1)
    claim_per_month = total_claim / max(months, 1)

    if severity == "minor damage" and injury_claim >= 10000 and injury_to_damage_ratio >= 1.5:
        adjusted_probability += 0.18
        signals.append("Unusually high injury claim relative to minor damage severity.")

    if police_report in {"NO", "?"} and witnesses == 0 and injury_claim >= 10000:
        adjusted_probability += 0.12
        signals.append("High injury claim has limited supporting police or witness documentation.")

    if months <= 12 and total_claim >= 50000:
        adjusted_probability += 0.10
        signals.append("Very high claim amount for a short-tenure customer.")

    if injury_share >= 0.65 and total_claim >= 25000:
        adjusted_probability += 0.08
        signals.append("Injury claim dominates the total claim amount.")

    adjusted_probability = min(max(adjusted_probability, 0.0), 0.95)
    return adjusted_probability, signals
