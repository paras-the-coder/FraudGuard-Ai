import streamlit as st


def render() -> None:
    st.markdown(
        """
        <section class="hero">
            <div class="hero-kicker">Insurance Claim Fraud Detection</div>
            <h1>FraudGuard AI - Pre-Claim Insurance Fraud Detection</h1>
            <p>FraudGuard AI screens insurance claims before payout, estimates fraud risk, and helps investigation teams focus on the claims that deserve a closer look.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="insight-strip">
            <div><strong>What it does</strong><span>Turns claim, incident, policy, customer, and vehicle details into a clear fraud probability.</span></div>
            <div><strong>Why it helps</strong><span>Claims teams can review risky cases before money leaves the business instead of reacting after payout.</span></div>
            <div><strong>How to use it</strong><span>Submit a claim in the demo, review the risk score, and inspect the model's top contributing signals.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="home-panel">
            <div>
                <span class="panel-kicker">Business workflow</span>
                <h3>From claim intake to investigation priority</h3>
                <p>FraudGuard AI helps insurance teams check claims faster. It does not replace human investigators. Instead, it highlights suspicious claims first so investigators can review them earlier, while low-risk claims can keep moving.</p>
            </div>
            <div class="workflow-steps">
                <span>1. Enter claim details</span>
                <span>2. Score fraud probability</span>
                <span>3. Review risk drivers</span>
                <span>4. Route to SIU if needed</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Make Prediction", type="primary", use_container_width=True):
        st.session_state["page"] = "Prediction Demo"
        st.rerun()
