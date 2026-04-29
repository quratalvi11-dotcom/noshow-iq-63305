import sys
import requests

URL = sys.argv[1] if len(sys.argv) > 1 else "https://Qurat-09-noshow-iq.hf.space"

def test_health():
    r = requests.get(f"{URL}/health")
    assert r.status_code == 200
    print("PASS /health")

def test_predict():
    r = requests.post(f"{URL}/predict", json={
        "age": 30, "scholarship": 0, "hipertension": 0,
        "diabetes": 0, "alcoholism": 0, "handcap": 0,
        "sms_received": 1,
        "scheduledday": "2016-04-29T00:00:00Z",
        "appointmentday": "2016-05-04T00:00:00Z"
    })
    assert r.status_code == 200
    print("PASS /predict")

def test_stats():
    r = requests.get(f"{URL}/stats")
    assert r.status_code == 200
    print("PASS /stats")

test_health()
test_predict()
test_stats()
print("ALL TESTS PASSED!")
