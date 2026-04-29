"""Data preprocessing module for NoShowIQ."""
import pandas as pd


def load_data(filepath):
    """Load raw CSV dataset."""
    df = pd.read_csv(filepath)
    return df


def clean_data(df):
    """Clean and preprocess the raw appointment data."""
    df = df.copy()

    # Rename columns
    df.columns = [col.strip() for col in df.columns]
    rename_map = {}
    for col in df.columns:
        if col.lower() == "no-show":
            rename_map[col] = "no_show"
        elif col.lower() == "hipertension":
            rename_map[col] = "hipertension"
        elif col.lower() == "handcap":
            rename_map[col] = "handcap"
    df = df.rename(columns=rename_map)
    df.columns = [col.strip().lower() for col in df.columns]

    # Fix age
    df = df[df["age"] >= 0].copy()
    df = df[df["age"] <= 100].copy()

    # Parse dates
    df["scheduledday"] = pd.to_datetime(df["scheduledday"], utc=True)
    df["appointmentday"] = pd.to_datetime(df["appointmentday"], utc=True)

    # Feature 1: days_in_advance
    df["days_in_advance"] = (
        df["appointmentday"].dt.normalize()
        - df["scheduledday"].dt.normalize()
    ).dt.days.clip(lower=0)

    # Feature 2: appointment_weekday
    df["appointment_weekday"] = df["appointmentday"].dt.dayofweek

    # Handle no_show only if present (training mode)
    if "no_show" in df.columns:
        df["no_show_binary"] = df["no_show"].map({"No": 0, "Yes": 1})
    else:
        df["no_show_binary"] = 0

    # Drop unnecessary columns
    cols_to_drop = [c for c in [
        "patientid", "appointmentid", "neighbourhood",
        "scheduledday", "appointmentday",
        "no_show", "no-show", "gender"
    ] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    df = df.dropna()
    return df


def get_features_and_target(df):
    """Split dataframe into features X and target y."""
    y = df["no_show_binary"]
    X = df.drop(columns=["no_show_binary"])
    return X, y
