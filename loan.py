import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Generate Synthetic Mirror of the Kaggle Dataset structure
print("🤖 Building financial training matrix columns...")
np.random.seed(42)
n_samples = 2000

data = {
    'no_of_dependents': np.random.randint(0, 6, n_samples),
    'education': np.random.choice([0, 1], n_samples), # 0: Graduate, 1: Not Graduate
    'self_employed': np.random.choice([0, 1], n_samples), # 0: No, 1: Yes
    'income_annum': np.random.randint(300000, 10000000, n_samples), # Value in Dalasis/Base
    'loan_amount': np.random.randint(100000, 7000000, n_samples),
    'loan_term': np.random.choice([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], n_samples),
    'cibil_score': np.random.randint(300, 900, n_samples),
    'residential_assets_value': np.random.randint(0, 5000000, n_samples),
    'commercial_assets_value': np.random.randint(0, 5000000, n_samples),
    'luxury_assets_value': np.random.randint(0, 3000000, n_samples),
    'bank_asset_value': np.random.randint(0, 4000000, n_samples)
}

df = pd.DataFrame(data)

# --- NEW STRICT FINTECH UNDERWRITING MATRIX ---
# 1. Calculate Monthly Disposable Capacity
monthly_income = df['income_annum'] / 12
estimated_monthly_payment = df['loan_amount'] / df['loan_term']

# 2. Total Net Asset Cover Value
total_assets = (df['residential_assets_value'] + 
                df['commercial_assets_value'] + 
                df['luxury_assets_value'] + 
                df['bank_asset_value'])

# 3. Apply Multi-Layer Sanction Rules to find defaults
# Rule A: Default risk if requested loan amount is higher than total asset backing
fail_asset_cover = df['loan_amount'] > total_assets

# Rule B: Default risk if monthly payment takes up more than 50% of monthly income
fail_debt_ratio = estimated_monthly_payment > (monthly_income * 0.5)

# Rule C: Default risk if credit score is deeply subprime (under 550)
fail_credit = df['cibil_score'] < 550

# Target is Approved (1) ONLY if they pass ALL clinical risk hurdles. Otherwise Default (0)
df['loan_status'] = np.where(fail_asset_cover | fail_debt_ratio | fail_credit, 0, 1)

# 2. Split Features & Targets
X = df.drop(columns=['loan_status'])
y = df['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Fit Pipeline Scalar & Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4. Train the Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Serialize Weights Matrix Files securely to Disk
joblib.dump(model, "finance_loan_models.pkl")
joblib.dump(scaler, "finance_scalers.pkl")

print("✅ Success: New strict models saved as 'finance_loan_models.pkl' & 'finance_scalers.pkl'!")
