"""Tests for NoShowIQ package."""

import pandas as pd
from noshow_iq.preprocess import clean_data, get_features_and_target
from noshow_iq.model import get_recommendation


def sample_df():
    """Return a minimal raw dataframe for testing."""
    return pd.DataFrame([{
        "PatientId": 123,
        "AppointmentID": 456,
        "Gender": "F",
        "ScheduledDay": "2016-04-29T10:00:00Z",
        "AppointmentDay": "2016-05-10T00:00:00Z",
        "Age": 30,
        "Neighbourhood": "JARDIM",
        "Scholarship": 0,
        "Hipertension": 0,
        "Diabetes": 0,
        "Alcoholism": 0,
        "Handcap": 0,
        "SMS_received": 1,
        "No-show": "No",
    }])


def test_clean_data_columns():
    """Test that clean_data renames columns correctly."""
    df = clean_data(sample_df())
    assert "no_show" in df.columns
    assert "hypertension" in df.columns
    assert "handicap" in df.columns


def test_clean_data_no_negative_ages():
    """Test that negative ages are dropped."""
    df = sample_df()
    df.loc[1] = df.loc[0]
    df.at[1, "Age"] = -1
    cleaned = clean_data(df)
    assert (cleaned["age"] >= 0).all()


def test_days_in_advance_feature():
    """Test that days_in_advance is engineered correctly."""
    df = clean_data(sample_df())
    assert "days_in_advance" in df.columns
    assert df["days_in_advance"].iloc[0] >= 0


def test_appointment_hour_feature():
    """Test that appointment_hour feature is present."""
    df = clean_data(sample_df())
    assert "appointment_hour" in df.columns


def test_get_features_and_target():
    """Test that X and y are split correctly."""
    df = clean_data(sample_df())
    X, y = get_features_and_target(df)
    assert "no_show" not in X.columns
    assert len(y) == len(X)


def test_get_recommendation_high():
    """Test recommendation for HIGH risk."""
    rec = get_recommendation("HIGH")
    assert "reminder" in rec.lower()


def test_get_recommendation_low():
    """Test recommendation for LOW risk."""
    rec = get_recommendation("LOW")
    assert "reminder" in rec.lower()
