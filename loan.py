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

# Inject logical financial rules for the targets to mimic Kaggle
# High CIBIL score + high assets = approved (1), otherwise default risk (0)
score_metric = (df['cibil_score'] * 3) + (df['income_annum'] / 10000) + (df['bank_asset_value'] / 10000)
loan_metric = (df['loan_amount'] / 5000)
df['loan_status'] = np.where(score_metric > loan_metric, 1, 0)

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
joblib.dump(model, "finance_loan_model.pkl")
joblib.dump(scaler, "finance_scaler.pkl")

print("✅ Success: 'finance_loan_model.pkl' & 'finance_scaler.pkl' have been saved successfully!")
