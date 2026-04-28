from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
from pymongo import MongoClient
import os

from noshow_iq.model import load_model
from noshow_iq.preprocess import clean_data

app = FastAPI()

# MongoDB
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(MONGO_URI)
db = client["noshow_db"]
pred_collection = db["predictions"]

model = load_model()


class Appointment(BaseModel):
    age: int
    scholarship: int
    hipertension: int
    diabetes: int
    alcoholism: int
    handcap: int
    sms_received: int
    scheduledday: str
    appointmentday: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_api(data: Appointment):
    df = pd.DataFrame([data.dict()])
    df = clean_data(df)

    features = df[[
        "age", "scholarship", "hipertension", "diabetes",
        "alcoholism", "handcap", "sms_received",
        "days_in_advance", "appointment_weekday"
    ]]

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][1]

    risk = "High" if pred == 1 else "Low"
    recommendation = "Send reminder" if pred == 1 else "No action"

    record = {
        "timestamp": datetime.utcnow(),
        "input": data.dict(),
        "risk": risk,
        "probability": float(prob),
        "recommendation": recommendation
    }

    pred_collection.insert_one(record)

    return record


@app.get("/history")
def history():
    records = list(pred_collection.find().sort("timestamp", -1).limit(20))
    for r in records:
        r["_id"] = str(r["_id"])
    return records


@app.get("/stats")
def stats():
    pipeline = [
        {
            "$group": {
                "_id": None,
                "total_predictions": {"$sum": 1},
                "high_risk_count": {
                    "$sum": {"$cond": [{"$eq": ["$risk", "High"]}, 1, 0]}
                },
                "low_risk_count": {
                    "$sum": {"$cond": [{"$eq": ["$risk", "Low"]}, 1, 0]}
                },
                "average_probability": {"$avg": "$probability"}
            }
        }
    ]

    result = list(pred_collection.aggregate(pipeline))
    return result[0] if result else {}