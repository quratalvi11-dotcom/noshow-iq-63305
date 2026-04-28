"""Data preprocessing module for NoShowIQ."""

import pandas as pd


def load_data(filepath):
    """Load raw CSV dataset."""
    df = pd.read_csv(filepath)
    return df


def clean_data(df):
    """Clean and preprocess the raw appointment data."""
    df = df.rename(columns={
        "No-show": "no_show",
        "Hipertension": "hypertension",
        "Handcap": "handicap"
    })

    df.columns = [col.strip().lower() for col in df.columns]

    # Drop invalid ages
    df = df[df["age"] >= 0].copy()
    df = df[df["age"] <= 100].copy()

    df["scheduledday"] = pd.to_datetime(df["scheduledday"], utc=True)
    df["appointmentday"] = pd.to_datetime(df["appointmentday"], utc=True)

    # Feature 1: days_in_advance
    df["days_in_advance"] = (
        df["appointmentday"].dt.normalize()
        - df["scheduledday"].dt.normalize()
    ).dt.days.clip(lower=0)

    # Feature 2: appointment_hour
    df["appointment_hour"] = df["scheduledday"].dt.hour

    # Feature 3: day of week
    df["appointment_dayofweek"] = df["appointmentday"].dt.dayofweek

    # Feature 4: month
    df["appointment_month"] = df["appointmentday"].dt.month

    # Feature 5: is weekend
    df["is_weekend"] = (
        df["appointment_dayofweek"] >= 5
    ).astype(int)

    # Feature 6: same day booking
    df["same_day"] = (df["days_in_advance"] == 0).astype(int)

    # Feature 7: neighbourhood risk — encode target first
    df["no_show_binary"] = df["no_show"].map({"No": 0, "Yes": 1})
    df["neighbourhood_risk"] = df.groupby(
        "neighbourhood"
    )["no_show_binary"].transform("mean")

    # Feature 8: total conditions
    df["total_conditions"] = (
        df["hypertension"] + df["diabetes"] + df["alcoholism"]
    )

    df["gender"] = df["gender"].map({"F": 0, "M": 1})
    df["no_show"] = df["no_show_binary"]

    df = df.drop(
        columns=[
            "patientid", "appointmentid",
            "neighbourhood", "scheduledday",
            "appointmentday", "no_show_binary"
        ]
    )

    df = df.dropna()
    return df


def get_features_and_target(df):
    """Split dataframe into features X and target y."""
    X = df.drop(columns=["no_show"])
    y = df["no_show"]
    return X, y