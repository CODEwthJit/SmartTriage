"""
SmartTriage Interactive Web UI (Streamlit).
Allows users, maintainers, and recruiters to interactively submit issue titles and descriptions,
view predicted categories and calibrated confidence, and inspect semantic duplicate warnings.
Supports both Live FastAPI Microservice connectivity and Resilient In-Process Engine execution.
"""

import os
from pathlib import Path
import time
import joblib
import numpy as np
import requests
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="SmartTriage - AI Issue Triaging & Deduplication",
    page_icon="🤖",
    layout="wide",
)

# Robust workspace root resolution (supports running from root or from ui/ directory)
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR if (CURRENT_DIR / "models").exists() else CURRENT_DIR.parent

# Priority heuristic map
CATEGORY_PRIORITY_MAP = {
    "security": "P0-Critical",
    "bug": "P1-High",
    "performance": "P1-High",
    "feature": "P2-Medium",
    "documentation": "P3-Low",
}

# Predefined example scenarios for quick testing
EXAMPLES = {
    "NPE Checkout Bug (Duplicate Test)": {
        "title": "Cart checkout crashes with NullPointerException when basket is empty",
        "body": "When submitting checkout without items, server throws 500 error in CheckoutService.",
    },
    "Critical SQL Injection": {
        "title": "SQL injection vulnerability in user search filter parameter",
        "body": "User filter parameter is concatenated directly into the database query string.",
    },
    "New OAuth2 Feature": {
        "title": "Add support for Google OAuth2 Login with PKCE",
        "body": "Users need to sign in with Google accounts. Requires client id and redirect configuration.",
    },
    "Slow Feed Query": {
        "title": "Activity stream query takes over 4 seconds for active users",
        "body": "Fetching user activity events times out on accounts with >50k events. Missing composite index.",
    },
}


@st.cache_resource(show_spinner="Loading in-process ML models...")
def load_local_engine():
    """
    Fallback loader: loads baseline classification pipeline and vector index directly
    for standalone cloud deployment (e.g. Streamlit Cloud / Hugging Face Spaces).
    """
    try:
        from src.inference.vector_search import DuplicateSearchEngine

        clf_path = PROJECT_ROOT / "models" / "registry" / "baseline_pipeline.joblib"
        idx_path = PROJECT_ROOT / "models" / "registry" / "vector_index.joblib"

        if clf_path.exists() and idx_path.exists():
            clf = joblib.load(clf_path)
            engine = DuplicateSearchEngine()
            engine.load(idx_path)
            return clf, engine
    except Exception as exc:
        print(f"In-process loader note: {exc}")
    return None, None


# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Cloud & Engine")

    default_api = os.getenv("API_URL", "http://127.0.0.1:8000/v1/triage")
    api_endpoint = st.text_input("FastAPI Endpoint URL", value=default_api)

    mode = st.radio(
        "Inference Strategy:",
        ["Auto (API with In-Process Fallback)", "Direct In-Process Only", "FastAPI Only"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📋 Severity Taxonomy")
    st.markdown("- **Security**: `P0-Critical`")
    st.markdown("- **Bug / Performance**: `P1-High`")
    st.markdown("- **Feature Request**: `P2-Medium`")
    st.markdown("- **Documentation**: `P3-Low`")

    st.markdown("---")
    st.caption("SmartTriage v1.0.0 • Production ML Architecture")


def run_in_process_triage(title: str, body: str, clf, engine):
    """Executes inference directly in-process without network overhead."""
    start_time = time.perf_counter()
    full_text = f"{title} {body}".strip()

    # Classification
    probs = clf.predict_proba([full_text])[0]
    classes = clf.classes_
    top_idx = int(np.argmax(probs))
    category = str(classes[top_idx])
    confidence = float(probs[top_idx])
    priority = CATEGORY_PRIORITY_MAP.get(category, "P2-Medium")

    # Duplicate search
    matches = engine.query(title=title, body=body, top_k=3, threshold=0.70)
    duplicate_warning = any(m["is_duplicate_warning"] for m in matches)
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return {
        "category": category,
        "priority": priority,
        "confidence": confidence,
        "duplicate_warning": duplicate_warning,
        "top_duplicates": matches,
        "latency_ms": latency_ms,
        "backend": "In-Process Engine (Local Memory)",
    }


# Header
st.title("🤖 SmartTriage Engine")
st.caption(
    "Production AI/ML Service for Intelligent GitHub Issue Routing & Real-Time Semantic Deduplication"
)
st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📝 Submit an Issue")

    selected_example = st.selectbox(
        "Load sample test case:", ["Custom Input"] + list(EXAMPLES.keys())
    )

    default_title = ""
    default_body = ""
    if selected_example != "Custom Input":
        default_title = EXAMPLES[selected_example]["title"]
        default_body = EXAMPLES[selected_example]["body"]

    input_title = st.text_input(
        "Issue Title",
        value=default_title,
        placeholder="e.g. NullPointerException on checkout",
    )
    input_body = st.text_area(
        "Issue Description / Body",
        value=default_body,
        height=140,
        placeholder="Paste logs, stack traces, or context here...",
    )

    submit_btn = st.button("🚀 Analyze & Triage Issue", type="primary", use_container_width=True)

with col2:
    st.subheader("🎯 Triage & Deduplication Results")

    if submit_btn:
        if len(input_title.strip()) < 3:
            st.error("Please enter an issue title with at least 3 characters.")
        else:
            payload = {"title": input_title.strip(), "body": input_body.strip()}
            res = None

            with st.spinner("Analyzing semantic embeddings and running model inference..."):
                # Strategy 1: FastAPI if selected
                if mode in ["Auto (API with In-Process Fallback)", "FastAPI Only"]:
                    try:
                        resp = requests.post(api_endpoint, json=payload, timeout=6.0)
                        if resp.status_code == 200:
                            res = resp.json()
                            res["backend"] = f"Live FastAPI Gateway ({api_endpoint})"
                    except requests.exceptions.RequestException:
                        if mode == "FastAPI Only":
                            st.error(f"Could not connect to FastAPI server at `{api_endpoint}`.")

                # Strategy 2: In-Process Fallback
                if res is None and mode != "FastAPI Only":
                    local_clf, local_engine = load_local_engine()
                    if local_clf and local_engine:
                        res = run_in_process_triage(
                            input_title.strip(), input_body.strip(), local_clf, local_engine
                        )
                    else:
                        st.error("In-process models could not be loaded from models/registry/.")

            # Render Results
            if res:
                cat = res["category"].upper()
                priority = res["priority"]
                conf = res["confidence"]
                latency = res["latency_ms"]

                st.success(f"**Predicted Category:** `{cat}`")

                metric_col1, metric_col2, metric_col3 = st.columns(3)
                metric_col1.metric("Component", cat)
                metric_col2.metric("Severity Priority", priority)
                metric_col3.metric("Latency", f"{latency:.1f} ms")

                st.write(f"**Model Confidence:** `{conf * 100:.1f}%`")
                st.progress(float(conf))

                st.caption(f"⚡ Served by: `{res.get('backend', 'SmartTriage Engine')}`")
                st.markdown("---")

                # Duplicate Warning Section
                if res["duplicate_warning"]:
                    st.error("🚨 **Potential Duplicate Issue Detected!**")
                    st.write(
                        "A highly similar issue already exists in the repository. Please review before filing:"
                    )
                else:
                    st.info("✅ **No immediate duplicate detected.** Unique issue.")

                # Render duplicate candidates
                st.write("### 🔍 Top Semantic Matches in Repository:")
                for idx, match in enumerate(res.get("top_duplicates", []), 1):
                    sim_pct = match["similarity_score"] * 100
                    status_badge = (
                        "🔴 DUPLICATE ALERT"
                        if match["is_duplicate_warning"]
                        else "⚪ Relevant Match"
                    )

                    with st.expander(
                        f"#{idx} | [{sim_pct:.1f}% Match] Issue #{match['issue_id']} - {match['title']}"
                    ):
                        st.write(f"**Similarity Score:** `{sim_pct:.2f}%` ({status_badge})")
                        st.write(
                            f"**Existing Category:** `{match['category']}` | **Priority:** `{match['priority']}`"
                        )
                        st.write(f"**Issue Title:** {match['title']}")
    else:
        st.write(
            "Enter an issue title on the left or select a sample case, then click **Analyze & Triage Issue**."
        )
