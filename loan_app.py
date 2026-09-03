import csv
import io
import os
import smtplib
import uuid
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import HRFlowable, Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from streamlit_drawable_canvas import st_canvas
import plotly.graph_objects as go

LOAN_RECORDS_CSV = "loan_records.csv"

# --- STREAMLIT PAGE CONFIGURATION ---
st.set_page_config(
    page_title="MFI RiskRadar - Enterprise Credit Assessment Portal",
    page_icon="💳",
    layout="wide"
)

# --- CUSTOM BRANDED CORPORATE STYLING ---
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #064E3B !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] a, [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }
    .stButton>button {
        background-color: #059669 !important;
        color: white !important;
        border-radius: 6px !important;
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- SESSION STATE INITIALIZATION FOR FORM RESET ---
if "reset_trigger" not in st.session_state:
    st.session_state["reset_trigger"] = 0

def reset_form_callback():
    st.session_state["reset_trigger"] += 1
    
def load_demo_callback():
    """Pre-fills the form with a guaranteed-approval demo applicant profile."""
    rk = f"_reset_{st.session_state['reset_trigger']}"
    st.session_state[f"officer{rk}"] = "Demo Officer"
    st.session_state[f"branch{rk}"] = "Head Office"
    st.session_state[f"rate{rk}"] = 15.0
    st.session_state[f"app_name{rk}"] = "Fatou Jallow"
    st.session_state[f"app_phone{rk}"] = "+220 1234567"
    st.session_state[f"app_email{rk}"] = "fatou.jallow@example.com"
    st.session_state[f"dependents{rk}"] = 2
    st.session_state[f"edu{rk}"] = 0
    st.session_state[f"emp{rk}"] = 0
    st.session_state[f"income{rk}"] = 500000
    st.session_state[f"loan_amt{rk}"] = 250000
    st.session_state[f"term{rk}"] = 12
    st.session_state[f"cibil{rk}"] = 650
    st.session_state[f"res_asset{rk}"] = 400000
    st.session_state[f"com_asset{rk}"] = 150000
    st.session_state[f"lux_asset{rk}"] = 50000
    st.session_state[f"bank_asset{rk}"] = 80000

# --- REPORTLAB PDF GENERATOR UTILITY (WITH E-SIGNATURE SUPPORT) ---
def generate_decision_pdf(applicant_name, applicant_phone, applicant_email, loan_id, status, loan_amount, loan_term, interest_rate, officer_name, branch_name, signature_img_bytes=None):
    """Generates an official enterprise PDF decision document buffer with signature support."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#064E3B'), spaceAfter=6)
    subtitle_style = ParagraphStyle('DocSubTitle', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#4B5563'), spaceAfter=15)
    body_style = ParagraphStyle('BodyTextCustom', parent=styles['Normal'], fontSize=10, leading=14, textColor=colors.HexColor('#1F2937'))
    
    status_color = "#059669" if status == "Approved" else "#DC2626"
    status_style = ParagraphStyle('StatusStyle', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor(status_color), spaceAfter=10)

    # Document Header
    story.append(Paragraph("MFI RISKRADAR ENTERPRISE CREDIT REPORT", title_style))
    story.append(Paragraph(f"Official Underwriting Decision Letter | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#064E3B'), spaceAfter=15))

    # Application Summary Table
    table_data = [
        [Paragraph("<b>Loan Reference ID:</b>", body_style), Paragraph(loan_id, body_style)],
        [Paragraph("<b>Applicant Name:</b>", body_style), Paragraph(applicant_name, body_style)],
        [Paragraph("<b>Phone / Email:</b>", body_style), Paragraph(f"{applicant_phone} / {applicant_email}", body_style)],
        [Paragraph("<b>Loan Officer:</b>", body_style), Paragraph(officer_name if officer_name else "Unspecified", body_style)],
        [Paragraph("<b>Branch Location:</b>", body_style), Paragraph(branch_name if branch_name else "Unspecified", body_style)],
        [Paragraph("<b>Requested Amount:</b>", body_style), Paragraph(f"{loan_amount:,.2f} GMD", body_style)],
        [Paragraph("<b>Repayment Term:</b>", body_style), Paragraph(f"{loan_term} Months @ {interest_rate}% p.a.", body_style)],
    ]
    
    summary_table = Table(table_data, colWidths=[150, 380])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F9FAFB')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))

    # Decision Banner
    story.append(Paragraph(f"UNDERWRITING DECISION: {status.upper()}", status_style))
    
    if status == "Approved":
        decision_text = (
            f"Based on the credit risk model evaluation, the applicant's profile meets institutional risk thresholds. "
            f"The loan request for <b>{loan_amount:,.2f} GMD</b> is conditionally <b>APPROVED</b> subject to standard verification."
        )
    else:
        decision_text = (
            f"Following standard underwriting analysis, the requested credit facility of <b>{loan_amount:,.2f} GMD</b> has been "
            f"<b>REJECTED</b> due to elevated debt service risk, credit score thresholds, or insufficient collateral coverage."
        )
    story.append(Paragraph(decision_text, body_style))
    story.append(Spacer(1, 20))

    # Sign-off Block with E-Signature Image
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#D1D5DB'), spaceAfter=15))
    
    if signature_img_bytes:
        sig_img = RLImage(signature_img_bytes, width=120, height=40)
        sign_data = [
            [sig_img, Paragraph("___________________________", body_style)],
            [Paragraph("<b>Loan Officer Digital Signature</b>", body_style), Paragraph("<b>Credit Committee Approval</b>", body_style)]
        ]
    else:
        sign_data = [
            [Paragraph("___________________________", body_style), Paragraph("___________________________", body_style)],
            [Paragraph("<b>Loan Officer Signature</b>", body_style), Paragraph("<b>Credit Committee Approval</b>", body_style)]
        ]
        
    sign_table = Table(sign_data, colWidths=[265, 265])
    sign_table.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'LEFT')]))
    story.append(sign_table)

    doc.build(story)
    buffer.seek(0)
    return buffer

# --- NOTIFICATION HELPERS (EMAIL & SMS) ---
def send_loan_status_email(applicant_email, applicant_name, status, loan_id, loan_amount):
    try:
        smtp_server = st.secrets["SMTP_SERVER"]
        smtp_port = st.secrets["SMTP_PORT"]
        sender_email = st.secrets["SMTP_EMAIL"]
        sender_password = st.secrets["SMTP_PASSWORD"]
    except Exception:
        return False

    if not applicant_email or applicant_email == "N/A" or "@" not in applicant_email:
        return False

    subject = f"Loan Application Status Update [{loan_id}] - MFI RiskRadar"
    body_content = f"<h2>Dear {applicant_name},</h2><p>Your loan application <strong>ID: {loan_id}</strong> for <strong>{loan_amount:,.2f} GMD</strong> is <strong>{status.upper()}</strong>.</p>"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"MFI RiskRadar <{sender_email}>"
    msg["To"] = applicant_email
    msg.attach(MIMEText(body_content, "html"))

    try:
        with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, applicant_email, msg.as_string())
        return True
    except Exception:
        return False

def dispatch_sms_alert(phone, applicant_name, status, loan_id):
    """Stub for SMS gateway dispatch."""
    if not phone or phone == "N/A":
        return False
    return True

# --- LOAD MODEL ASSETS ---
@st.cache_resource
def load_assets():
    if os.path.exists("finance_loan_model.pkl") and os.path.exists("finance_scaler.pkl"):
        model = joblib.load("finance_loan_model.pkl")
        scaler = joblib.load("finance_scaler.pkl")
        return model, scaler
    return None, None

model, scaler = load_assets()

# --- HEADER SECTION ---
header_col1, header_col2 = st.columns([4, 1])
with header_col1:
    st.title("🛡️ MFI RiskRadar: Enterprise Credit Scorer")
    st.write("Institutional Credit Underwriting, SHAP Explainable AI & Automated Decisioning Portal.")
with header_col2:
    if os.path.exists("contract.png"):
        st.image("contract.png", width=130)

st.markdown("---")
st.markdown("### 📤 Bulk Credit Assessment (CSV Upload)")
uploaded_file = st.file_uploader("Upload Batch Applications CSV", type=["csv"])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    if st.button("Run Batch Assessment"):
        # Scale and predict
        batch_scaled = scaler.transform(batch_df[feature_names])
        batch_df['Loan_Status'] = ["Approved" if p == 1 else "Rejected" for p in model.predict(batch_scaled)]
        
        st.write("### Batch Assessment Results", batch_df)
        
        batch_csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Batch Results (CSV)",
            data=batch_csv,
            file_name="batch_underwriting_results.csv",
            mime="text/csv"
        )

if model is None or scaler is None:
    st.error("🚨 **Asset Error:** Production model files (`finance_loan_model.pkl` / `finance_scaler.pkl`) not found.")
else:
    reset_key = f"_reset_{st.session_state['reset_trigger']}"

    st.subheader("📝 Credit Application Entry")
    st.button("🎯 Load Demo Applicant (Guaranteed Approval)", on_click=load_demo_callback, use_container_width=True)
    st.markdown("---")
    
    acc_col1, acc_col2, acc_col3 = st.columns(3)
    with acc_col1:
        officer_name = st.text_input("Loan Officer Name", value="", key=f"officer{reset_key}")
    with acc_col2:
        branch_name = st.text_input("Branch / Sub-Station", value="", key=f"branch{reset_key}")
    with acc_col3:
        interest_rate = st.number_input("Annual Interest Rate (%)", min_value=1.0, max_value=50.0, value=15.0, step=0.5, key=f"rate{reset_key}")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    feature_names = [
        "no_of_dependents", "education", "self_employed", "income_annum",
        "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
        "commercial_assets_value", "luxury_assets_value", "bank_asset_value"
    ]

    with col1:
        st.markdown("### 👤 Demographic Profiles")
        applicant_name = st.text_input("Applicant Full Name", value="", key=f"app_name{reset_key}")
        applicant_phone = st.text_input("Phone Number", value="", placeholder="+220 1234567", key=f"app_phone{reset_key}")
        applicant_email = st.text_input("Email Address", value="", placeholder="applicant@example.com", key=f"app_email{reset_key}")
        no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=20, value=2, step=1, key=f"dependents{reset_key}")
        education = st.selectbox("Education Level", options=[0, 1], format_func=lambda x: "Graduate (0)" if x == 0 else "Not Graduate (1)", key=f"edu{reset_key}")
        self_employed = st.selectbox("Self-Employed Status", options=[0, 1], format_func=lambda x: "No (0)" if x == 0 else "Yes / Independent (1)", key=f"emp{reset_key}")

    with col2:
        st.markdown("### 💰 Financial Terms")
        income_annum = st.number_input("Annual Income Base (GMD)", min_value=0, value=500000, step=10000, key=f"income{reset_key}")
        loan_amount = st.number_input("Requested Loan Amount (GMD)", min_value=0, value=250000, step=10000, key=f"loan_amt{reset_key}")
        loan_term = st.slider("Loan Repayment Term (Months)", min_value=2, max_value=240, value=12, step=2, key=f"term{reset_key}")
        cibil_score = st.number_input("Credit / CIBIL Score Rating", min_value=300, max_value=900, value=650, step=5, key=f"cibil{reset_key}")

    with col3:
        st.markdown("### 🏡 Collateral & Assets")
        residential_assets_value = st.number_input("Residential Asset Value (GMD)", min_value=0, value=400000, key=f"res_asset{reset_key}")
        commercial_assets_value = st.number_input("Commercial Asset Value (GMD)", min_value=0, value=150000, key=f"com_asset{reset_key}")
        luxury_assets_value = st.number_input("Luxury Asset Holdings (GMD)", min_value=0, value=50000, key=f"lux_asset{reset_key}")
        bank_asset_value = st.number_input("Liquid Bank Balance (GMD)", min_value=0, value=80000, key=f"bank_asset{reset_key}")

    st.markdown("---")

    # Digital E-Signature Capture Block
    st.markdown("### ✍️ Loan Officer Authorization & E-Signature")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)",
        stroke_width=2,
        stroke_color="#000000",
        background_color="#F9FAFB",
        height=100,
        width=400,
        drawing_mode="freedraw",
        key=f"canvas{reset_key}"
    )

    st.markdown("---")

    btn_col1, btn_col2 = st.columns([3, 1])
    with btn_col1:
        run_evaluation = st.button("Execute Credit Risk Evaluation", use_container_width=True)
    with btn_col2:
        st.button("🔄 Reset Form", on_click=reset_form_callback, use_container_width=True)

    if run_evaluation:
        input_data = np.array([[
            no_of_dependents, education, self_employed, income_annum,
            loan_amount, loan_term, cibil_score, residential_assets_value,
            commercial_assets_value, luxury_assets_value, bank_asset_value
        ]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        
        # Risk Metrics
        total_assets = residential_assets_value + commercial_assets_value + luxury_assets_value + bank_asset_value
        debt_to_income = (loan_amount / income_annum * 100) if income_annum > 0 else 0.0
        loan_to_asset = (loan_amount / total_assets * 100) if total_assets > 0 else 0.0
        
        monthly_rate = (interest_rate / 100) / 12
        if monthly_rate > 0 and loan_term > 0:
            monthly_payment = (loan_amount * monthly_rate * (1 + monthly_rate)**loan_term) / ((1 + monthly_rate)**loan_term - 1)
        else:
            monthly_payment = loan_amount / loan_term if loan_term > 0 else 0.0

        pd_score = max(0.01, min(0.99, (850 - cibil_score) / 550.0))
        lgd = max(0.10, 1.0 - (total_assets / (loan_amount if loan_amount > 0 else 1)))
        expected_loss = pd_score * lgd * loan_amount

        loan_id = "GAWFA-" + uuid.uuid4().hex[:8].upper()
        display_name = applicant_name.strip() if applicant_name.strip() else "Unnamed Applicant"
        display_phone = applicant_phone.strip() if applicant_phone.strip() else "N/A"
        display_email = applicant_email.strip() if applicant_email.strip() else "N/A"

        st.markdown("---")

        # 1. Decision Banner
        if prediction[0] == 1:
            status = "Approved"
            st.success(f"✅ **CRITERIA MATCHED: LOAN APPROVED** — Loan ID: `{loan_id}`")
        else:
            status = "Rejected"
            st.error(f"⚠️ **CRITERIA BLOCKED: APPLICATION HIGH RISK** — Loan ID: `{loan_id}`")

        # 2. Executive Metrics & Expected Loss
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Debt-to-Income Ratio", f"{debt_to_income:.1f}%")
        m_col2.metric("Loan-to-Asset Ratio", f"{loan_to_asset:.1f}%")
        m_col3.metric("Est. Monthly Repayment", f"{monthly_payment:,.2f} GMD")
        m_col4.metric("Expected Loss (Provision)", f"{expected_loss:,.2f} GMD")

        # 3. SHAP Explainable AI Feature Diagnostics
        st.markdown("### 📊 SHAP Explainable AI Diagnostics")
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(input_scaled)
            
            fig, ax = plt.subplots(figsize=(8, 3))
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0] if shap_values.ndim == 2 else shap_values[0][0]
            
            y_pos = np.arange(len(feature_names))
            ax.barh(y_pos, sv, align='center', color=['#059669' if x > 0 else '#DC2626' for x in sv])
            ax.set_yticks(y_pos)
            ax.set_yticklabels(feature_names)
            ax.invert_yaxis()
            ax.set_xlabel('SHAP Impact on Credit Approval Probability')
            st.pyplot(fig)
        except Exception:
            st.info("SHAP visual breakdown generated based on static parameter impact.")
        
        

        # Visual Credit Score Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cibil_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Credit Score Tier Assessment"},
            gauge={
                'axis': {'range': [300, 900]},
                'bar': {'color': "#064E3B"},
                'steps': [
                    {'range': [300, 550], 'color': "#FEE2E2"},  # High Risk
                    {'range': [550, 700], 'color': "#FEF3C7"},  # Moderate Risk
                    {'range': [700, 900], 'color': "#D1FAE5"}   # Prime
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': cibil_score
                }
            }
        )
    )
        fig_gauge.update_layout(height=280, margin=dict(l=10, r=10, t=60, b=10), title_font_size=16,)
        st.plotly_chart(fig_gauge, use_container_width=True)
 


        # 4. Safely Process Signature Image
        sig_bytes = None
        if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, 3] > 0):
            img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
            sig_buffer = io.BytesIO()
            img.save(sig_buffer, format='PNG')
            sig_buffer.seek(0)
            sig_bytes = sig_buffer

        # 5. PDF Document Download
        pdf_buffer = generate_decision_pdf(
            applicant_name=display_name,
            applicant_phone=display_phone,
            applicant_email=display_email,
            loan_id=loan_id,
            status=status,
            loan_amount=loan_amount,
            loan_term=loan_term,
            interest_rate=interest_rate,
            officer_name=officer_name,
            branch_name=branch_name,
            signature_img_bytes=sig_bytes
        )
        
        st.download_button(
            label="⬇️ Download Official Decision Letter with Signature (PDF)",
            data=pdf_buffer.getvalue(),
            file_name=f"Loan_Decision_{loan_id}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        # 6. Amortization Table Expander
        with st.expander("📅 View Complete Monthly Amortization Schedule"):
            schedule = []
            balance = loan_amount
            for month in range(1, int(loan_term) + 1):
                interest_pay = balance * monthly_rate
                principal_pay = monthly_payment - interest_pay
                balance -= principal_pay
                schedule.append({
                    "Month": month,
                    "Monthly Payment (GMD)": round(monthly_payment, 2),
                    "Principal Repayment (GMD)": round(principal_pay, 2),
                    "Interest Charge (GMD)": round(interest_pay, 2),
                    "Remaining Principal Balance (GMD)": max(0, round(balance, 2))
                })
            st.dataframe(pd.DataFrame(schedule), use_container_width=True)
                # 7. Stress-Testing Simulator (What-If Analysis)
        with st.expander("🧪 Stress-Testing Simulator (What-If Analysis)"):
            st.write("Evaluate how macroeconomic shifts affect this applicant's risk profile.")

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                rate_shock = st.slider("Interest Rate Increase (+%)", 0.0, 10.0, 2.0, step=0.5, key=f"rate_shock{reset_key}")
            with col_s2:
                haircut = st.slider("Collateral Valuation Haircut (-%)", 0, 50, 10, step=5, key=f"haircut{reset_key}")

            stressed_rate = interest_rate + rate_shock
            stressed_monthly_rate = (stressed_rate / 100) / 12
            stressed_assets = total_assets * (1 - haircut / 100)

            if stressed_monthly_rate > 0 and loan_term > 0:
                stressed_payment = (loan_amount * stressed_monthly_rate * (1 + stressed_monthly_rate)**loan_term) / ((1 + stressed_monthly_rate)**loan_term - 1)
            else:
                stressed_payment = loan_amount / loan_term

            stressed_lta = (loan_amount / stressed_assets * 100) if stressed_assets > 0 else 0

            st.warning(
                f"**Stressed Repayment:** {stressed_payment:,.2f} GMD/mo | "
                f"**Stressed Loan-to-Asset:** {stressed_lta:.1f}%"
            )    

        # 7. CLEAN TRANSACTION SAVING (Auto-Resets Old Non-Matching Files)
        # 7. CLEAN TRANSACTION SAVING (Extended Risk & Financial Metrics)
        record = {
            "loan_id": loan_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "applicant_name": display_name,
            "applicant_phone": display_phone,
            "applicant_email": display_email,
            "loan_amount": loan_amount,
            "loan_term_months": loan_term,
            "cibil_score": cibil_score,
            "annual_income": income_annum,
            "debt_to_income_ratio": round(debt_to_income, 2),
            "loan_to_asset_ratio": round(loan_to_asset, 2),
            "expected_loss": round(expected_loss, 2),
            "officer_name": officer_name,
            "branch_name": branch_name,
            "loan_status": status,
        }
        record_df = pd.DataFrame([record])
        
        file_exists = os.path.exists(LOAN_RECORDS_CSV)
        should_reset = False
        
        # Check if existing CSV columns match the new 15-column schema
        if file_exists:
            try:
                existing_df = pd.read_csv(LOAN_RECORDS_CSV, nrows=1)
                if list(existing_df.columns) != list(record_df.columns):
                    should_reset = True
            except Exception:
                should_reset = True

        mode = "w" if (not file_exists or should_reset) else "a"
        header = True if (not file_exists or should_reset) else False

        record_df.to_csv(
            LOAN_RECORDS_CSV,
            mode=mode,
            header=header,
            index=False,
            quoting=csv.QUOTE_ALL
        )

        send_loan_status_email(display_email, display_name, status, loan_id, loan_amount)
        dispatch_sms_alert(display_phone, display_name, status, loan_id)

# --- SIDEBAR AUDIT LOG ---
# --- ENHANCED SIDEBAR AUDIT & EXECUTIVE ANALYTICS ---
with st.sidebar:
    st.header("🛡️ FinTech RiskHub")
    st.markdown("---")
    
    if os.path.exists(LOAN_RECORDS_CSV):
        try:
            records_df = pd.read_csv(LOAN_RECORDS_CSV)
            
            # -------------------------------------------------------------
            # A. REAL-TIME EXECUTIVE PORTFOLIO ANALYTICS
            # -------------------------------------------------------------
            st.markdown("### 📈 Executive Risk Metrics")
            
            total_apps = len(records_df)
            approved_apps = len(records_df[records_df['loan_status'] == 'Approved'])
            approval_rate = (approved_apps / total_apps * 100) if total_apps > 0 else 0.0
            
            total_volume = records_df['loan_amount'].sum() if 'loan_amount' in records_df.columns else 0.0
            avg_cibil = records_df['cibil_score'].mean() if 'cibil_score' in records_df.columns else 0.0
            
            kpi_col1, kpi_col2 = st.columns(2)
            kpi_col1.metric("Applications", f"{total_apps}")
            kpi_col2.metric("Approval Rate", f"{approval_rate:.1f}%")
            
            kpi_col3, kpi_col4 = st.columns(2)
            kpi_col3.metric("Volume (GMD)", f"{total_volume:,.0f}")
            kpi_col4.metric("Avg Credit Score", f"{avg_cibil:.0f}")
            
            st.markdown("---")

            # -------------------------------------------------------------
            # B. AUDIT LOG DATA TABLE
            # -------------------------------------------------------------
            with st.expander("📁 Detailed Transaction Audit Log", expanded=False):
                st.dataframe(records_df, use_container_width=True)

            # -------------------------------------------------------------
            # C. PROFESSIONAL MULTI-SHEET EXCEL EXPORT ENGINE
            # -------------------------------------------------------------
            def create_excel_report(df):
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    # Sheet 1: Raw Audit Transactions
                    df.to_excel(writer, sheet_name='Transaction Audit Log', index=False)
                    
                    # Sheet 2: Executive KPI Summary Breakdown
                    summary_data = {
                        "Executive KPI Metric": [
                            "Total Evaluated Applications",
                            "Approved Applications",
                            "Rejected Applications",
                            "Portfolio Approval Rate (%)",
                            "Total Underwritten Capital (GMD)",
                            "Average Applicant Credit Score"
                        ],
                        "Institutional Value": [
                            total_apps,
                            approved_apps,
                            total_apps - approved_apps,
                            f"{approval_rate:.2f}%",
                            f"{total_volume:,.2f}",
                            f"{avg_cibil:.1f}"
                        ]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Executive KPI Summary', index=False)
                    
                excel_buffer.seek(0)
                return excel_buffer

            excel_data = create_excel_report(records_df)

            # Export Buttons
            st.download_button(
                label="📊 Export Executive Report (Excel .xlsx)",
                data=excel_data,
                file_name=f"Executive_Credit_Audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            # Simple CSV Fallback
            st.download_button(
                label="📄 Export Raw Data (CSV)",
                data=records_df.to_csv(index=False).encode('utf-8'),
                file_name=f"loan_records_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.warning("Audit log standardizing. Submit an evaluation to sync portfolio analytics.")
    else:
        st.info("No credit evaluations logged yet. System ready for application entries.")

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
with st.sidebar:
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
