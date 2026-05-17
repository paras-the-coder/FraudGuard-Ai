from __future__ import annotations

import streamlit as st

from app_pages import home, performance, prediction
from src.config import STYLE_PATH


st.set_page_config(
    page_title="FraudGuard AI",
    page_icon=":material/security:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css() -> None:
    if STYLE_PATH.exists():
        st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def sidebar_navigation() -> str:
    pages = ["Home", "Prediction Demo", "Model Performance"]
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"
    if st.session_state["page"] not in pages:
        st.session_state["page"] = "Home"

    with st.sidebar:
        st.markdown("<div class='brand'><span>FraudGuard</span><strong>AI</strong></div>", unsafe_allow_html=True)
        selected = st.radio(
            "Navigation",
            pages,
            index=pages.index(st.session_state["page"]),
            label_visibility="collapsed",
        )
        st.session_state["page"] = selected
        st.markdown(
            """
            <div class="sidebar-note">
                <strong>Production Lens</strong>
                <span>Recall-optimized screening for pre-claim fraud detection.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return st.session_state["page"]


def main() -> None:
    load_css()
    page = sidebar_navigation()

    if page == "Home":
        home.render()
    elif page == "Prediction Demo":
        prediction.render()
    else:
        performance.render()


if __name__ == "__main__":
    main()
