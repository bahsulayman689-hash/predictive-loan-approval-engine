# 📊 MFI RiskRadar: Predictive Loan Approval Engine

An end-to-end FinTech Machine Learning web application designed to automate credit underwriting, evaluate asset leverage, and mitigate default risk for Microfinance Institutions (MFIs) and credit unions. Powered by **Python**, **Scikit-Learn**, **Plotly**, and **Streamlit**.

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![ML Library](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)

---

## 🚀 Live Production Demo
🔗 **Interact with the deployed live application here:**  
👉 **[MFI RiskRadar Streamlit App](https://predictive-loan-approval-engine-bah.streamlit.app/)**

---

## 💡 Business Case & Market Value
In emerging financial ecosystems like The Gambia, small businesses, market vendors, and micro-entrepreneurs frequently apply for capital to scale operations. Traditionally, loan officers manually audit paper records to estimate credit risk—a process that takes days and is highly prone to human error, inconsistency, or unexpected defaults.

**MFI RiskRadar** solves this operational bottleneck. By instantly computing an applicant's demographic profile, annual revenue, credit rating, and underlying asset structures against a trained Random Forest classification model, it reduces credit processing time from **5 days to 5 seconds**. This empowers loan officers to protect institutional capital reserves by flagging high-risk profiles before capital leaves the branch.

---

## ✨ Key Platform Features

* **⚡ Automated Credit Underwriting:** Instant inference engine predicting loan approval vs. rejection with clear decision outputs.
* **📈 Real-Time Executive Portfolio Analytics:** Live sidebar dashboard displaying total applications, approval percentages, underwritten capital volume, and average credit scores.
* **🎯 Interactive Credit Risk Scorecard Gauge:** Plotly-powered visual credit score gauge classifying applicant credit health into clear risk tiers (High Risk, Moderate Risk, Prime).
* **🧪 Macroeconomic Stress-Testing Simulator:** Interactive "What-If" engine allowing underwriters to simulate interest rate hikes and asset valuation haircuts to evaluate loan durability.
* **📊 Dual-Format Data Export:** One-click download options for raw data (`.csv`) or a structured multi-sheet executive report (`.xlsx` via `openpyxl`) complete with KPI breakdowns and transaction audit logs.

---

## 🎛️ System Architecture

1. **Pipeline Data Scaling:** Raw user demographic and financial inputs are securely routed through a standardized pre-fitted scaling vector (`finance_scaler.pkl`) to normalize variance and financial skewness.
2. **Classification Inference Engine:** Normalized matrices are evaluated by a robust ensemble classifier (`finance_loan_model.pkl`) trained on optimized financial indicators.
3. **Responsive UI Frontend:** Built on a custom-styled Streamlit container framework utilizing targeted high-contrast emerald green CSS overrides for corporate fintech presentation.

---

## 📥 Features Evaluated by the Model

The pipeline maps an 11-point multidimensional vector profile to ensure accurate risk classification:

* **Demographics:** Number of dependents, formal graduation/education index, and self-employment status.
* **Financial Terms:** Annual income base, requested loan amount, loan repayment lifecycle terms (months), and historical CIBIL rating score.
* **Asset Foundations:** Liquid bank/capital reserves, residential property valuations, commercial real estate holdings, and luxury asset valuations.

---

## 💻 Local Installation & Setup

Execute the following deployment steps to launch this financial engine locally on your machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
