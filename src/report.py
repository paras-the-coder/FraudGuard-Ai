from __future__ import annotations

from datetime import datetime
from io import BytesIO
from textwrap import wrap

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from src.config import HIGH_RISK_THRESHOLD, LOW_RISK_THRESHOLD


def recommended_action(probability: float) -> tuple[str, str]:
    if probability < LOW_RISK_THRESHOLD:
        return "Auto-approve for processing.", "low"
    if probability < HIGH_RISK_THRESHOLD:
        return "Flag for manual review by a junior adjuster.", "medium"
    return "Escalate to Special Investigations Unit (SIU) immediately.", "high"


def data_integrity_flags(claim: pd.Series) -> list[tuple[str, str]]:
    flags: list[tuple[str, str]] = []
    days_since_bind = pd.to_numeric(claim.get("days_between_policy_incident"), errors="coerce")
    claim_ratio = pd.to_numeric(claim.get("claim_ratio"), errors="coerce")
    total_claim = pd.to_numeric(claim.get("total_claim_amount"), errors="coerce")
    injury_claim = pd.to_numeric(claim.get("injury_claim"), errors="coerce")
    property_claim = pd.to_numeric(claim.get("property_claim"), errors="coerce")
    vehicle_claim = pd.to_numeric(claim.get("vehicle_claim"), errors="coerce")
    vehicle_age = pd.to_numeric(claim.get("vehicle_age"), errors="coerce")

    if pd.notna(days_since_bind) and days_since_bind < 30:
        flags.append(("high", f"Incident occurred only {int(days_since_bind)} days after policy binding."))
    if pd.notna(claim_ratio) and claim_ratio >= 50:
        flags.append(("high", f"Claim-to-premium ratio is very high at {claim_ratio:.1f}x."))
    elif pd.notna(claim_ratio) and claim_ratio >= 30:
        flags.append(("medium", f"Claim-to-premium ratio is elevated at {claim_ratio:.1f}x."))
    if pd.notna(vehicle_age) and vehicle_age < 0:
        flags.append(("high", "Vehicle year is later than the incident year."))
    if all(pd.notna(value) for value in [total_claim, injury_claim, property_claim, vehicle_claim]):
        component_total = injury_claim + property_claim + vehicle_claim
        if abs(total_claim - component_total) > 100:
            flags.append(("medium", "Claim component amounts do not reconcile with total claim amount."))

    for column, label in [
        ("collision_type", "Collision type"),
        ("property_damage", "Property damage"),
        ("police_report_available", "Police report"),
    ]:
        if str(claim.get(column, "")).strip() == "?":
            flags.append(("medium", f"{label} is unknown in the submitted claim."))

    if not flags:
        flags.append(("low", "No rule-based data integrity red flags were found."))
    return flags


def _html_bars(contributors: pd.DataFrame) -> str:
    max_impact = max(float(contributors["Impact"].max()), 0.001)
    bars = []
    for row in contributors.itertuples(index=False):
        contribution = float(row.Contribution)
        width = max(abs(contribution) / max_impact * 100, 4)
        color = "#ef4444" if contribution > 0 else "#10b981"
        bars.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{row.Feature}</div>
              <div class="bar-track"><span style="width:{width:.1f}%; background:{color};"></span></div>
              <div class="bar-value">{contribution:.3f}</div>
            </div>
            """
        )
    return "\n".join(bars)


def build_html_report(
    claim: pd.Series,
    probability: float,
    prediction: str,
    risk: str,
    contributors: pd.DataFrame,
    rule_signals: list[str] | None = None,
) -> str:
    action, action_level = recommended_action(probability)
    flags = data_integrity_flags(claim)
    rule_signals = rule_signals or []
    rows = "\n".join(
        f"<tr><td>{row.Feature}</td><td>{row.Direction}</td><td>{row.Contribution}</td></tr>"
        for row in contributors.itertuples(index=False)
    )
    flag_rows = "\n".join(
        f"<li class='flag-{level}'>{text}</li>"
        for level, text in flags
    )
    signal_rows = "\n".join(f"<li>{signal}</li>" for signal in rule_signals)
    signal_section = (
        f"""
        <div class="section">
          <h2>Additional Fraud Risk Signals</h2>
          <ul>{signal_rows}</ul>
        </div>
        """
        if rule_signals
        else ""
    )
    progress_color = "#10b981" if probability < LOW_RISK_THRESHOLD else "#f59e0b" if probability < HIGH_RISK_THRESHOLD else "#ef4444"
    incident_date = claim.get("incident_date", "Unknown")
    policy_number = claim.get("policy_number", "Demo claim")
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>FraudGuard AI Prediction Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #0f172a; color: #e5e7eb; padding: 32px; }}
    .card {{ background: #111c31; border: 1px solid #223454; border-radius: 14px; padding: 24px; max-width: 820px; margin: 0 auto; }}
    h1 {{ color: #f8fafc; margin-bottom: 4px; }}
    .muted {{ color: #94a3b8; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 24px 0; }}
    .metric {{ background: #0b1220; border-radius: 12px; padding: 16px; }}
    .metric strong {{ display: block; color: #10b981; font-size: 24px; }}
    .progress {{ height: 14px; background: #0b1220; border-radius: 999px; overflow: hidden; border: 1px solid #223454; }}
    .progress span {{ display: block; height: 100%; width: {probability * 100:.1f}%; background: {progress_color}; }}
    .section {{ background: #0b1220; border: 1px solid #223454; border-radius: 12px; padding: 16px; margin-top: 18px; }}
    .action {{ border-left: 4px solid {progress_color}; }}
    .bar-row {{ display: grid; grid-template-columns: 220px 1fr 70px; gap: 10px; align-items: center; margin: 10px 0; }}
    .bar-label {{ color: #cbd5e1; font-size: 13px; }}
    .bar-track {{ height: 10px; background: #172554; border-radius: 999px; overflow: hidden; }}
    .bar-track span {{ display: block; height: 100%; border-radius: 999px; }}
    .bar-value {{ color: #e5e7eb; text-align: right; font-size: 13px; }}
    .flag-high {{ color: #fca5a5; }}
    .flag-medium {{ color: #fcd34d; }}
    .flag-low {{ color: #86efac; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
    td, th {{ border-bottom: 1px solid #223454; padding: 10px; text-align: left; }}
    th {{ color: #38bdf8; }}
    @media (max-width: 600px) {{
      body {{ padding: 16px; }}
      .card {{ padding: 16px; max-width: 100%; }}
      .grid {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; gap: 4px; }}
      .bar-value {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <div class="card">
    <h1>FraudGuard AI Prediction Report</h1>
    <p class="muted">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
    <div class="grid">
      <div class="metric"><span>Fraud Probability</span><strong>{probability:.1%}</strong></div>
      <div class="metric"><span>Prediction</span><strong>{prediction}</strong></div>
      <div class="metric"><span>Risk Level</span><strong>{risk}</strong></div>
    </div>
    <div class="progress"><span></span></div>
    <div class="section action"><strong>Recommended Action:</strong> {action}</div>
    <div class="section">
      <h2>Claim Summary</h2>
      <p><strong>Policyholder Number:</strong> {policy_number}</p>
      <p><strong>Date of Incident:</strong> {incident_date}</p>
      <p><strong>Claim Amount:</strong> ${float(claim.get("total_claim_amount", 0)):,.0f}</p>
      <p><strong>Incident Type:</strong> {claim.get("incident_type", "Unknown")}</p>
      <p><strong>Incident Severity:</strong> {claim.get("incident_severity", "Unknown")}</p>
    </div>
    <div class="section">
      <h2>Data Integrity Flags</h2>
      <ul>{flag_rows}</ul>
    </div>
    {signal_section}
    <div class="section">
      <h2>Visual Risk Drivers</h2>
      {_html_bars(contributors)}
    </div>
    <h2>Top Feature Contributors</h2>
    <table>
      <thead><tr><th>Feature</th><th>Direction</th><th>Contribution</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</body>
</html>
"""


def build_pdf_report(
    claim: pd.Series,
    probability: float,
    prediction: str,
    risk: str,
    contributors: pd.DataFrame,
    rule_signals: list[str] | None = None,
) -> bytes:
    action, _ = recommended_action(probability)
    flags = data_integrity_flags(claim)
    rule_signals = rule_signals or []
    buffer = BytesIO()

    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        y = 0.94
        ax.text(0.08, y, "FraudGuard AI Prediction Report", fontsize=20, fontweight="bold", color="#0f172a")
        y -= 0.04
        ax.text(0.08, y, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", fontsize=9, color="#64748b")
        y -= 0.06
        ax.text(0.08, y, f"Fraud Probability: {probability:.1%}", fontsize=15, fontweight="bold", color="#0f172a")
        ax.text(0.50, y, f"Prediction: {prediction}", fontsize=15, fontweight="bold", color="#0f172a")
        y -= 0.035
        ax.text(0.08, y, f"Risk Level: {risk}", fontsize=12, color="#334155")
        y -= 0.04
        pdf_bar_color = "#10b981" if probability < LOW_RISK_THRESHOLD else "#f59e0b" if probability < HIGH_RISK_THRESHOLD else "#ef4444"
        ax.barh([y], [probability], left=[0.08], height=0.015, color=pdf_bar_color)
        ax.barh([y], [1 - probability], left=[0.08 + probability], height=0.015, color="#e2e8f0")
        ax.set_xlim(0, 1)
        y -= 0.06

        summary = [
            f"Recommended Action: {action}",
            f"Policyholder Number: {claim.get('policy_number', 'Demo claim')}",
            f"Date of Incident: {claim.get('incident_date', 'Unknown')}",
            f"Claim Amount: ${float(claim.get('total_claim_amount', 0)):,.0f}",
            f"Incident: {claim.get('incident_type', 'Unknown')} | {claim.get('incident_severity', 'Unknown')}",
        ]
        for line in summary:
            ax.text(0.08, y, line, fontsize=10.5, color="#0f172a")
            y -= 0.028

        y -= 0.02
        ax.text(0.08, y, "Data Integrity Flags", fontsize=13, fontweight="bold", color="#0f172a")
        y -= 0.03
        for _, text in flags:
            for part in wrap(f"- {text}", 95):
                ax.text(0.10, y, part, fontsize=9.5, color="#334155")
                y -= 0.023

        if rule_signals:
            y -= 0.02
            ax.text(0.08, y, "Additional Fraud Risk Signals", fontsize=13, fontweight="bold", color="#0f172a")
            y -= 0.03
            for signal in rule_signals:
                for part in wrap(f"- {signal}", 95):
                    ax.text(0.10, y, part, fontsize=9.5, color="#334155")
                    y -= 0.023

        y -= 0.02
        ax.text(0.08, y, "Top Model Contributors", fontsize=13, fontweight="bold", color="#0f172a")
        y -= 0.04
        sorted_contrib = contributors.sort_values("Contribution")
        max_impact = max(float(sorted_contrib["Impact"].max()), 0.001)
        for row in sorted_contrib.itertuples(index=False):
            width = abs(float(row.Contribution)) / max_impact * 0.34
            color = "#ef4444" if float(row.Contribution) > 0 else "#10b981"
            ax.text(0.08, y, str(row.Feature)[:34], fontsize=8.8, color="#0f172a")
            ax.barh([y + 0.002], [width], left=[0.42], height=0.012, color=color)
            ax.text(0.78, y, f"{float(row.Contribution):.3f}", fontsize=8.8, color="#0f172a")
            y -= 0.028

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    buffer.seek(0)
    return buffer.getvalue()
