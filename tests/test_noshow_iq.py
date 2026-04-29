"""Tests for NoShowIQ package."""
import pandas as pd
from noshow_iq.preprocess import clean_data, get_features_and_target
from noshow_iq.model import load_model


def make_sample_df():
    return pd.DataFrame([{
        "age": 30,
        "scholarship": 0,
        "hipertension": 0,
        "diabetes": 0,
        "alcoholism": 0,
        "handcap": 0,
        "sms_received": 1,
        "scheduledday": "2016-04-29T00:00:00Z",
        "appointmentday": "2016-05-04T00:00:00Z",
        "No-show": "No",
        "gender": "F",
        "neighbourhood": "JARDIM DA PENHA",
        "patientid": 123,
        "appointmentid": 456
    }])


def test_clean_data_returns_dataframe():
    df = make_sample_df()
    result = clean_data(df)
    assert isinstance(result, pd.DataFrame)


def test_clean_data_has_days_in_advance():
    df = make_sample_df()
    result = clean_data(df)
    assert "days_in_advance" in result.columns


def test_clean_data_has_appointment_weekday():
    df = make_sample_df()
    result = clean_data(df)
    assert "appointment_weekday" in result.columns


def test_clean_data_removes_negative_ages():
    df = make_sample_df()
    df["age"] = -1
    result = clean_data(df)
    assert len(result) == 0


def test_get_features_and_target():
    df = make_sample_df()
    cleaned = clean_data(df)
    X, y = get_features_and_target(cleaned)
    assert len(X) == len(y)


def test_model_loads():
    model = load_model()
    assert model is not None
