# 📊 MFI RiskRadar: Predictive Loan Approval Engine

An end-to-end FinTech Machine Learning web application designed to automate credit assessment, evaluate asset leverage, and mitigate default risk for Microfinance Institutions (MFIs) and credit unions. Powered by **Python**, **Scikit-Learn**, and **Streamlit**.

![Python Version](https://shields.io)
![Framework](https://shields.io)
![ML Library](https://shields.io)

---

## 🚀 Live Production Demo
🔗 **Interact with the deployed live application here:** *[INSERT YOUR STREAMLIT SHARE LINK HERE]*

---

## 💡 Business Case & Market Value
In emerging financial ecosystems like The Gambia, small businesses, market vendors, and entrepreneurs frequently apply for microloans. Traditionally, loan officers manually audit paper records to estimate credit risk—a process that takes days and is highly prone to human error or loan defaults.

**MFI RiskRadar** solves this operational bottleneck. By instantly computing an applicant's demographic profile, annual revenue, and underlying asset structures against a trained Random Forest classification model, it reduces credit processing time from **5 days to 5 seconds**. This helps credit unions protect their capital reserves by flagging high-risk profiles before capital leaves the branch.

---

## 🎛️ System Architecture

1. **Pipeline Data Scaling:** Raw user demographic inputs are securely routed through a standardized pre-fitted scaling vector (`finance_scaler.pkl`) to normalize financial skewness and data variance.
2. **Classification Inference Engine:** Normalized matrices are evaluated by a robust ensemble classifier (`finance_loan_model.pkl`) trained on optimized financial indicators.
3. **Responsive UI Frontend:** Built on a custom-styled Streamlit container framework utilizing targeted high-contrast emerald green CSS overrides for corporate fintech presentation.

---

## 📥 Features Evaluated by the Model

The pipeline maps an 11-point multidimensional vector profile to ensure accurate risk classification:

* **Demographics:** Number of dependents, formal graduation/education index, and self-employed tracking.
* **Financial Terms:** Annual income base, requested loan amount, loan repayment lifecycle terms (months), and historical credit/CIBIL rating score.
* **Asset Foundations:** Liquid bank/capital reserves, residential property valuations, commercial real estate holdings, and luxury asset valuations.

---

## 💻 Local Installation & Setup

Execute the following deployment steps to launch this financial engine locally on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com
cd YOUR_REPOSITORY_NAME
```

### 2. Install Lightweight Dependencies
To avoid package version conflicts and ensure stable native compilation, configure your local workspace using the optimized requirements script:
```bash
pip install -r requirements.txt
```

### 3. Generate the Machine Learning Weights File
Run the training pipeline script locally to construct the synthetic data frame, compute the scaling parameters, and save the binary model pipelines:
```bash
python train_model.py
```

### 4. Boot Up the Dashboard Web Server
```bash
streamlit run app_app.py
```

---

## 👨‍💻 Project Developer & System Architect
* **Lead AI Engineer:** Sulayman Bah
* **Domain Focus:** Machine Learning Pipelines, FinTech Solutions, & Automated MLOps Dashboards
* **Contact:** [bahsulayman689@gmail.com](mailto:bahsulayman689@gmail.com)

---
*Disclaimer: This system is built as a portfolio machine learning proof-of-concept and does not constitute formal financial underwriting software.*
