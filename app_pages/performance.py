import streamlit as st
from sklearn.metrics import confusion_matrix

from src.charts import (
    confusion_matrix_chart,
    model_comparison_bar,
    model_comparison_table,
    precision_recall_chart,
    roc_curve_chart,
    threshold_chart,
)
from src.evaluation import get_lr_evaluation


def _interpretation_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="interpretation-card">
            <strong>{title}</strong>
            <span>{body}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render() -> None:
    st.title("Model Performance")
    st.caption("Production model metrics are recomputed from the saved Logistic Regression pipeline.")

    evaluation = get_lr_evaluation()
    metrics = evaluation["metrics"]
    cm = confusion_matrix(evaluation["y_true"], evaluation["y_pred"])
    tn, fp, fn, tp = cm.ravel()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall", f"{metrics['Recall']:.1%}")
    c2.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
    c3.metric("ROC-AUC", f"{metrics['ROC-AUC']:.3f}")
    c4.metric("F2 Score", f"{metrics['F2 Score']:.3f}")

    st.markdown("### Logistic Regression Diagnostics")
    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(confusion_matrix_chart(evaluation["y_true"], evaluation["y_pred"]), use_container_width=True)
        _interpretation_card(
            "Confusion matrix",
            f"Out of 200 test claims, the model correctly cleared {tn} legitimate claims and caught {tp} fraud claims. The {fn} missed fraud cases are the biggest risk, while {fp} false alarms represent claims that would need extra review.",
        )
    with c6:
        st.plotly_chart(roc_curve_chart(evaluation["y_true"], evaluation["probabilities"]), use_container_width=True)
        _interpretation_card(
            "ROC curve",
            f"This shows how well the model separates fraud from legitimate claims across many cutoffs. The ROC-AUC is {metrics['ROC-AUC']:.3f}, which means fraud claims generally receive higher risk scores than legitimate claims.",
        )

    c7, c8 = st.columns(2)
    with c7:
        st.plotly_chart(precision_recall_chart(evaluation["y_true"], evaluation["probabilities"]), use_container_width=True)
        _interpretation_card(
            "Precision-recall curve",
            "Fraud is the smaller class, so this chart is useful for operations. It shows the tradeoff between catching more fraud and sending more legitimate claims to manual review.",
        )
    with c8:
        st.plotly_chart(threshold_chart(evaluation["y_true"].to_numpy(), evaluation["probabilities"].to_numpy()), use_container_width=True)
        _interpretation_card(
            "Threshold analysis",
            "A lower threshold flags more claims and catches more fraud, but creates more review work. A higher threshold reduces review volume, but increases the chance of missing fraud.",
        )

    st.markdown("### Model Comparison")
    table = model_comparison_table()
    st.dataframe(
        table.style.format({column: "{:.3f}" for column in table.columns if column != "Model"}),
        use_container_width=True,
        hide_index=True,
    )
    st.plotly_chart(model_comparison_bar(), use_container_width=True)

    threshold = st.slider("Operational Review Threshold", 0.05, 0.95, 0.35, 0.05)
    probabilities = evaluation["probabilities"].to_numpy()
    y_true = evaluation["y_true"].to_numpy()
    predicted = (probabilities >= threshold).astype(int)
    flagged = int(predicted.sum())
    caught = int(((predicted == 1) & (y_true == 1)).sum())
    actual_fraud = int((y_true == 1).sum())
    recall = caught / actual_fraud if actual_fraud else 0
    st.info(
        f"At a {threshold:.0%} threshold, the model flags {flagged} of 200 test claims and catches {caught} of {actual_fraud} fraud cases ({recall:.1%} recall)."
    )
