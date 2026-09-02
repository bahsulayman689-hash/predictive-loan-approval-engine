import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
import uuid
from datetime import datetime

LOAN_RECORDS_CSV = "loan_records.csv"

# --- CRITICAL STREAMLIT CONFIGURATION LAYER ---
st.set_page_config(
    page_title="MFI RiskRadar - Loan Credit Scorer",
    page_icon="💳",
    layout="wide"
)

# ------------------------- Custom CSS Branded Styling --------------------------------
st.markdown(
    """
    <style>
    /* Dark Emerald Green Background representing corporate finance security */
    [data-testid="stSidebar"] {
        background-color: #064E3B !important;
    }
    /* Enforce pristine white legibility matrix across sidebar elements */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] a,
    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }
    /* Style button for crisp interface execution click flows */
    .stButton>button {
        background-color: #059669 !important;
        color: white !important;
        border-radius: 6px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Fallback Safe File Load Mechanisms ---
@st.cache_resource
def load_assets():
    if os.path.exists("finance_loan_model.pkl") and os.path.exists("finance_scaler.pkl"):
        model = joblib.load("finance_loan_model.pkl")
        scaler = joblib.load("finance_scaler.pkl")
        return model, scaler
    return None, None

model, scaler = load_assets()

# --- Main App Header Matrix ---
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.title("📊 MFI RiskRadar: Predictive Loan Approval Engine")
    st.write("Automating credit assessment and default risk analysis using optimized pipeline models.")
with header_col2:
    if os.path.exists("contract.png"):
        st.image("contract.png", width=130)

st.markdown("---")

if model is None or scaler is None:
    st.error("🚨 **Asset Error:** Production pipeline model files (`finance_loan_model.pkl` / `finance_scaler.pkl`) not found in workspace directory. Please run `train_model.py` first to generate your weights arrays.")
else:
    # --- Input Processing Enclosure Forms ---
    st.subheader("📝 Input Applicant Credit Profile Parameters")

    st.markdown("---")

    # --- Officer / Branch Accountability Fields ---
    acc_col1, acc_col2 = st.columns(2)
    with acc_col1:
        officer_name = st.text_input("Loan Officer Name", value="")
    with acc_col2:
        branch_name = st.text_input("Branch / Sub-Station", value="")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 👤 Demographic Profiles")
        applicant_name = st.text_input("Applicant Full Name", value="")
        no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=2, step=1)
        education = st.selectbox("Education Level", options=[0, 1], format_func=lambda x: "Graduate (0)" if x == 0 else "Not Graduate (1)")
        self_employed = st.selectbox("Self-Employed Status", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes / Independent Entrepreneur (1)")

    with col2:
        st.markdown("### 💰 Financial Terms")
        income_annum = st.number_input("Annual Income Base (GMD Equiv.)", min_value=0, value=500000, step=10000)
        loan_amount = st.number_input("Requested Loan Amount (GMD)", min_value=0, value=250000, step=10000)
        loan_term = st.slider("Loan Repayment Term (Months)", min_value=2, max_value=240, value=12, step=2)
        cibil_score = st.number_input("Credit / CIBIL Score Rating", min_value=300, max_value=900, value=650, step=5)

    with col3:
        st.markdown("### 🏡 Underlying Asset Evaluation")
        residential_assets_value = st.number_input("Residential Asset Value (GMD)", min_value=0, value=400000)
        commercial_assets_value = st.number_input("Commercial Asset Value (GMD)", min_value=0, value=150000)
        luxury_assets_value = st.number_input("Luxury Asset Holdings (GMD)", min_value=0, value=50000)
        bank_asset_value = st.number_input("Liquid Bank Balance / Capital (GMD)", min_value=0, value=80000)

    st.markdown("---")

    # --- Run Compute Execution Loop Trigger ---
    if st.button("Execute Credit Risk Evaluation Assessment", use_container_width=True):
        # Assemble feature pipeline list exactly matching vector orders
        input_data = np.array([[
            no_of_dependents, education, self_employed, income_annum,
            loan_amount, loan_term, cibil_score, residential_assets_value,
            commercial_assets_value, luxury_assets_value, bank_asset_value
        ]])

        # Ingest through pre-fitted standardization parameters
        input_scaled = scaler.transform(input_data)

        # Query classification algorithm prediction matrix
        prediction = model.predict(input_scaled)

        # Generate a unique loan ID for this assessment
        loan_id = "GAWFA-" + uuid.uuid4().hex[:8].upper()

        # Display contextual results arrays based on response outcomes
        if prediction[0] == 1:
            status = "Approved"
            st.success(f"✅ **CRITERIA MATCHED: LOAN APPROVED** — Loan ID `{loan_id}`. The applicant exhibits a low-risk portfolio structure. Profile metrics fall within safe credit underwriting thresholds.")
        else:
            status = "Rejected"
            st.error(f"⚠️ **CRITERIA BLOCKED: APPLICANT HIGH RISK** — Loan ID `{loan_id}`. Credit score parameters, debt ratios, or asset evaluations indicate a higher probability of credit default.")

        # --- Persist applicant record (approved or rejected) to CSV ---
        record = {
            "loan_id": loan_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "applicant_name": applicant_name if applicant_name.strip() else "Unnamed Applicant",
            "officer_name": officer_name if officer_name.strip() else "Unspecified",
            "branch_name": branch_name if branch_name.strip() else "Unspecified",
            "no_of_dependents": no_of_dependents,
            "education": education,
            "self_employed": self_employed,
            "income_annum": income_annum,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "cibil_score": cibil_score,
            "residential_assets_value": residential_assets_value,
            "commercial_assets_value": commercial_assets_value,
            "luxury_assets_value": luxury_assets_value,
            "bank_asset_value": bank_asset_value,
            "loan_status": status,
        }
        record_df = pd.DataFrame([record])

        if os.path.exists(LOAN_RECORDS_CSV):
            record_df.to_csv(LOAN_RECORDS_CSV, mode="a", header=False, index=False)
        else:
            record_df.to_csv(LOAN_RECORDS_CSV, mode="w", header=True, index=False)

        st.info(f"📁 Record `{loan_id}` saved to `{LOAN_RECORDS_CSV}` — status: **{status}**.")

        # Live Calculation Analysis Expander
        with st.expander("🔍 View Risk Engine Calculations"):
            st.write(f"📊 **Monthly Disposable Income:** {income_annum / 12:,.2f} GMD")
            st.write(f"💳 **Estimated Monthly Payment:** {loan_amount / loan_term:,.2f} GMD")
            st.write(f"🏡 **Total Collateral Portfolio Value:** {residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value:,.2f} GMD")

# --- Sidebar Branded Footprints ---
with st.sidebar:
    st.header("🛡️ FinTech RiskHub")
    st.write("This standalone interface acts as a decision support utility tailored for local credit unions and MFI networks.")
    st.markdown("---")

    # 1. Help Guide Expander
    with st.sidebar.expander("💡 Input Help Guide"):
        st.write("**CIBIL Score:** Scores below 550 represent severe credit default risk and will trigger an automatic system rejection.")
        st.write("**Dependents:** High dependent footprints directly restrict monthly repayment capacities.")
        st.write("**Asset Cover:** The requested loan total must be backed by healthy collateral valuations to safeguard capital.")

    # 2. Model Architecture Expander
    with st.sidebar.expander("⚙️ Model Architecture Details"):
        st.write("**Core Algorithm:** Random Forest Classifier")
        st.write("**Tree Estimators:** 100 Decision Trees")
        st.write("**Data Scaler:** Scikit-Learn StandardScaler")
        st.write("**Inference Engine:** Serialized Weight Arrays via Joblib")

    st.markdown("---")

    # 3. Loan Records Viewer + Summary Chart
    with st.sidebar.expander("📁 Loan Records & Portfolio Summary"):
        if os.path.exists(LOAN_RECORDS_CSV):
            records_df = pd.read_csv(LOAN_RECORDS_CSV)
            status_counts = records_df["loan_status"].value_counts()

            st.write(f"**Total Records:** {len(records_df)}")
            st.write(f"✅ Approved: {status_counts.get('Approved', 0)}  |  ⚠️ Rejected: {status_counts.get('Rejected', 0)}")

            st.bar_chart(status_counts)

            st.dataframe(records_df, use_container_width=True, height=220)
            st.download_button(
                "⬇️ Download CSV",
                data=records_df.to_csv(index=False).encode("utf-8"),
                file_name="loan_records.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.write("No loan records saved yet.")

    st.markdown("---")
    st.subheader("👨‍💻 Application Architect")
    if os.path.exists("IMG-20260704-WA0633.jpg"):
        st.image("IMG-20260704-WA0633.jpg", caption="Sulayman Bah", width=180)
    else:
        st.info("💡 Profile image asset not found.")

    st.write("**Lead Engineer:** Sulayman Bah")
    st.write("**Specialization:** Predictive Financial Risk System Integrations")
    st.markdown("---")

    st.subheader("🔗 System Portals")
    st.markdown("[📁 GitHub Profile](https://github.com/bahsulayman689-hash)")
    st.markdown("[💼 LinkedIn Profile](https://www.linkedin.com/in/sulayman-bah-8a7096423)")
    st.markdown("[📧 Email Support](mailto:bahsulayman689@gmail.com)")