"""Main dashboard application."""

import streamlit as st

from src.dashboard.pages import (
    dataset_analysis,
    evaluation,
    failure_analysis,
    inference,
    training,
)

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="BDD100K Dashboard",
    layout="wide",
)

# ==========================================================
# PAGE REGISTRY
# ==========================================================

PAGES = {
    "📈 Training": training.render,
    "📊 Dataset Analysis": dataset_analysis.render,
    "🔍 Inference": inference.render,
    "📋 Evaluation": evaluation.render,
    "🚨 Failure Analysis": failure_analysis.render,
}

# ==========================================================
# SIDEBAR
# ==========================================================

def render_sidebar() -> str:
    """Render sidebar navigation."""

    st.sidebar.title(
        "BDD100K Dashboard"
    )

    return st.sidebar.radio(
        "Navigation",
        list(PAGES.keys()),
    )

# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    """Run dashboard."""

    selected_page = render_sidebar()

    render_page = PAGES[
        selected_page
    ]

    render_page()


if __name__ == "__main__":
    main()