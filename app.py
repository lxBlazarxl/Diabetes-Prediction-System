from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Diabetes Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('diabetes_scaler.pkl')

class PatientData(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

@app.post("/predict")
def predict_diabetes(patient: PatientData):
    input_df = pd.DataFrame([patient.model_dump()])
    
    input_scaled = scaler.transform(input_df)
    
    prob = model.predict_proba(input_scaled)[0]
    is_diabetic = model.predict(input_scaled)[0]
    
    state = "Diabetic" if is_diabetic == 1 else "Healthy"
    confidence = prob[1] if is_diabetic == 1 else prob[0]
    
    return {
        "prediction": state,
        "confidence": round(float(confidence), 4)
    }
