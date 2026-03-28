import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def load_and_clean_data(path="./diabetes.csv"):
    df = pd.read_csv(path).drop_duplicates()
    
    cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
    
    df = df.fillna(df.mean())
    return df

df = load_and_clean_data()
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# --- PREDICTION ENGINE ---
def predict_diabetes(data_dict):
    input_df = pd.DataFrame([data_dict])
    input_scaled = scaler.transform(input_df)
    
    prob = model.predict_proba(input_scaled)[0]
    is_diabetic = model.predict(input_scaled)[0]
    
    state = "Diabetic" if is_diabetic == 1 else "Healthy"
    confidence = prob[1] if is_diabetic == 1 else prob[0]
    
    return f"Result: {state} ({confidence:.2%} confidence)"

joblib.dump(model, 'diabetes_model.pkl')
joblib.dump(scaler, 'diabetes_scaler.pkl')

print("Model and scaler saved.")
