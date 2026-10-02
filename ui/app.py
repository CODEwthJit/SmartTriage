"""
SmartTriage Interactive Web UI (Streamlit).
Allows users and recruiters to interactively submit issue titles and descriptions,
view predicted categories and calibrated confidence, and inspect semantic duplicate warnings.
"""

import requests
import streamlit as st

# Configure page layout
st.set_page_config(page_title="SmartTriage - AI Issue Triaging & Deduplication", page_icon="🤖", layout="wide")

API_URL = "http://127.0.0.1:8000/v1/triage"

# Header
st.title("🤖 SmartTriage Engine")
st.caption("Production AI/ML Service for Intelligent GitHub Issue Routing & Real-Time Semantic Deduplication")
st.markdown("---")

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

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📝 Submit an Issue")

    # Quick example loader
    selected_example = st.selectbox("Load sample test case:", ["Custom Input"] + list(EXAMPLES.keys()))

    default_title = ""
    default_body = ""
    if selected_example != "Custom Input":
        default_title = EXAMPLES[selected_example]["title"]
        default_body = EXAMPLES[selected_example]["body"]

    input_title = st.text_input("Issue Title", value=default_title, placeholder="e.g. NullPointerException on checkout")
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

            with st.spinner("Analyzing semantic embeddings and running model inference..."):
                try:
                    response = requests.post(API_URL, json=payload, timeout=5.0)

                    if response.status_code == 200:
                        res = response.json()

                        # Badges for Category and Priority
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
                                "🔴 DUPLICATE ALERT" if match["is_duplicate_warning"] else "⚪ Relevant Match"
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
                        st.error(f"API Error ({response.status_code}): {response.text}")

                except requests.exceptions.ConnectionError:
                    st.warning("⚠️ Could not connect to FastAPI server at `http://127.0.0.1:8000`.")
                    st.info(
                        "To start the backend API, run in your terminal:\n```powershell\n& 'C:\\ProgramData\\anaconda3\\python.exe' -m uvicorn src.api.main:app --port 8000 --reload\n```"
                    )
    else:
        st.write("Enter an issue title on the left or select a sample case, then click **Analyze & Triage Issue**.")
