"""FastAPI application for NoShowIQ."""

import os
from datetime import datetime, timezone
from fastapi import FastAPI
from pydantic import BaseModel
from pymongo import MongoClient
from noshow_iq.preprocess import clean_data
from noshow_iq.model import load_model, predict, get_recommendation
import pandas as pd


app = FastAPI(title="NoShowIQ", version="0.1.0")

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["noshowiq"]
predictions_col = db["predictions"]
training_runs_col = db["training_runs"]

clf = load_model()


class AppointmentInput(BaseModel):
    PatientId: float
    AppointmentID: int
    Gender: str
    ScheduledDay: str
    AppointmentDay: str
    Age: int
    Neighbourhood: str
    Scholarship: int
    Hipertension: int
    Diabetes: int
    Alcoholism: int
    Handcap: int
    SMS_received: int


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok", "model": "loaded"}


@app.post("/predict")
def predict_noshow(input: AppointmentInput):
    """Predict no-show risk for one appointment."""
    raw = input.dict()
    df = pd.DataFrame([raw])
    df_clean = clean_data(df)
    X, _ = df_clean.drop(columns=["no_show"], errors="ignore"), None
    X = df_clean.drop(columns=["no_show"], errors="ignore")
    pred, prob = predict(clf, X)
    risk = "HIGH" if prob[0] >= 0.5 else "LOW"
    recommendation = get_recommendation(risk)

    doc = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_input": raw,
        "risk_level": risk,
        "probability": round(float(prob[0]), 4),
        "recommendation": recommendation,
    }
    predictions_col.insert_one(doc)

    return {
        "risk_level": risk,
        "probability": round(float(prob[0]), 4),
        "recommendation": recommendation,
    }


@app.get("/history")
def history():
    """Return last 20 predictions."""
    docs = list(
        predictions_col.find({}, {"_id": 0}).sort("timestamp", -1).limit(20)
    )
    return {"predictions": docs}


@app.get("/stats")
def stats():
    """Return aggregated stats using MongoDB aggregation pipeline."""
    pipeline = [
        {
            "$group": {
                "_id": None,
                "total_predictions": {"$sum": 1},
                "high_risk_count": {
                    "$sum": {
                        "$cond": [{"$eq": ["$risk_level", "HIGH"]}, 1, 0]
                    }
                },
                "low_risk_count": {
                    "$sum": {
                        "$cond": [{"$eq": ["$risk_level", "LOW"]}, 1, 0]
                    }
                },
                "average_probability": {"$avg": "$probability"},
            }
        }
    ]
    result = list(predictions_col.aggregate(pipeline))
    last_run = training_runs_col.find_one(
        {}, {"_id": 0}, sort=[("timestamp", -1)]
    )
    last_trained = last_run["timestamp"] if last_run else None
    if result:
        r = result[0]
        return {
            "total_predictions": r["total_predictions"],
            "high_risk_count": r["high_risk_count"],
            "low_risk_count": r["low_risk_count"],
            "average_probability": round(r["average_probability"], 4),
            "last_trained": last_trained,
        }
    return {
        "total_predictions": 0,
        "high_risk_count": 0,
        "low_risk_count": 0,
        "average_probability": 0.0,
        "last_trained": last_trained,
    }
