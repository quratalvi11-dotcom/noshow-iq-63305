"""Model training and prediction module for NoShowIQ."""

import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE


MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")


def train(X, y, model_path=MODEL_PATH):
    """Train a RandomForest classifier with SMOTE for class imbalance."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_res, y_train_res)

    # Save model
    joblib.dump(clf, model_path)

    # Evaluate
    metrics = evaluate(clf, X_test, y_test)
    return clf, metrics


def load_model(model_path=MODEL_PATH):
    """Load trained model from disk."""
    return joblib.load(model_path)


def predict(clf, X):
    """Return predicted class and probability."""
    prob = clf.predict_proba(X)[:, 1]
    pred = clf.predict(X)
    return pred, prob


def evaluate(clf, X_test, y_test):
    """Evaluate model and return precision, recall, F1 per class."""
    y_pred = clf.predict(X_test)
    report = classification_report(
        y_test, y_pred, target_names=["show", "no_show"], output_dict=True
    )
    return report


def get_recommendation(risk_level):
    """Return a recommendation based on risk level."""
    if risk_level == "HIGH":
        return (
            "Send reminder SMS and call patient 24h before appointment."
        )
    return "Send standard appointment reminder SMS."
