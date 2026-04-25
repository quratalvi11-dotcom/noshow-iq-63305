"""Data preprocessing module for NoShowIQ."""

import pandas as pd


def load_data(filepath):
    """Load raw CSV dataset."""
    df = pd.read_csv(filepath)
    return df


def clean_data(df):
    """Clean and preprocess the raw appointment data."""
    # Fix the problematic target column name (No-show has a hyphen)
    df = df.rename(columns={"No-show": "no_show"})

    # Standardise all column names to lowercase with underscores
    df.columns = [
        col.strip()
        .lower()
        .replace("-", "_")
        .replace("hipertension", "hypertension")
        .replace("handcap", "handicap")
        for col in df.columns
    ]

    # Drop rows with invalid ages (negative ages in dataset)
    df = df[df["age"] >= 0].copy()

    # Convert date columns to datetime
    df["scheduledday"] = pd.to_datetime(df["scheduledday"], utc=True)
    df["appointmentday"] = pd.to_datetime(df["appointmentday"], utc=True)

    # Feature 1: days_in_advance (how early was the appointment booked)
    df["days_in_advance"] = (
        df["appointmentday"] - df["scheduledday"]
    ).dt.days

    # Clip negative values (same-day or data errors)
    df["days_in_advance"] = df["days_in_advance"].clip(lower=0)

    # Feature 2: appointment_hour (what hour was it scheduled)
    df["appointment_hour"] = df["scheduledday"].dt.hour

    # Encode gender: F=0, M=1
    df["gender"] = df["gender"].map({"F": 0, "M": 1})

    # Encode target: No=0 (showed up), Yes=1 (no-show)
    df["no_show"] = df["no_show"].map({"No": 0, "Yes": 1})

    # Drop columns not useful for prediction
    df = df.drop(
        columns=["patientid", "appointmentid",
                 "neighbourhood", "scheduledday", "appointmentday"]
    )

    # Drop any remaining nulls
    df = df.dropna()

    return df


def get_features_and_target(df):
    """Split dataframe into features X and target y."""
    X = df.drop(columns=["no_show"])
    y = df["no_show"]
    return X, y