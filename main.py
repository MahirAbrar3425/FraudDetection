from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Fraud Detection API")

# Allow frontend


# Load artifacts
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# EXACT features used during training (order matters!)
FEATURES = [
    'Time',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
    'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17',
    'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25',
    'V26', 'V27', 'V28',
    'Amount',
    'Amount_scaled',
    'log_amount',
    'time_diff'
]

THRESHOLD = 0.1


@app.post("/predict")
def predict(transaction: dict):
    """
    Predict fraud probability for a single transaction.
    """

    # Convert request to DataFrame
    df = pd.DataFrame([transaction])

    # ---------- Feature Engineering ----------
    df["Amount_scaled"] = scaler.transform(df[["Amount"]]).ravel()
    df["log_amount"] = np.log1p(df["Amount"])
    df["time_diff"] = 0.0  # for single transaction inference

    # ---------- Enforce schema ----------
    X = df[FEATURES]

    # ---------- Prediction ----------
    prob = model.predict_proba(X)[0, 1]

    return {
        "fraud_probability": float(prob),
        "is_fraud": bool(prob > THRESHOLD),
    }
