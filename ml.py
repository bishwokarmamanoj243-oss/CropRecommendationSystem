"""Training, validation, and prediction for the authoritative crop CSV."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "crop_recommendation.csv"
MODEL_DIR = BASE_DIR / "models"
FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_and_validate(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = FEATURE_ORDER + ["label"]
    if list(df.columns) != required:
        raise ValueError(f"CSV columns must be exactly {required}")
    if df.empty or df[required].isna().any().any():
        raise ValueError("Dataset contains missing values")
    if df.duplicated().any():
        raise ValueError("Dataset contains duplicate rows")
    for column in FEATURE_ORDER:
        df[column] = pd.to_numeric(df[column], errors="raise")
        if not np.isfinite(df[column]).all():
            raise ValueError(f"{column} contains invalid numeric values")
    if not df["ph"].between(0, 14).all():
        raise ValueError("pH values must be between 0 and 14")
    if (df["rainfall"] < 0).any():
        raise ValueError("Rainfall cannot be negative")
    if (df[FEATURE_ORDER] < 0).any().any():
        raise ValueError("Nutrient and climate values cannot be negative")
    if (df["label"].astype(str).str.strip() == "").any():
        raise ValueError("Dataset contains empty crop labels")
    return df


def _metrics(model, X_train, y_train, X_test, y_test, folds: int) -> dict:
    train_pred, test_pred = model.predict(X_train), model.predict(X_test)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, test_pred, average="weighted", zero_division=0
    )
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, np.vstack((X_train, X_test)), np.hstack((y_train, y_test)), cv=cv)
    return {"training_accuracy": float(accuracy_score(y_train, train_pred)),
            "testing_accuracy": float(accuracy_score(y_test, test_pred)),
            "cross_validation_accuracy": float(scores.mean()),
            "cross_validation_std": float(scores.std()), "precision": float(precision),
            "recall": float(recall), "f1_score": float(f1),
            "confusion_matrix": confusion_matrix(y_test, test_pred).tolist()}


def train_and_save() -> dict:
    df = load_and_validate()
    X, labels = df[FEATURE_ORDER].to_numpy(), df["label"].to_numpy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    # With 3 examples/class, 1 test record/class keeps stratification valid and interpretable.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 / 3, random_state=42, stratify=y
    )
    candidates = {
        "Decision Tree": Pipeline([("scaler", StandardScaler()), ("model", DecisionTreeClassifier(random_state=42))]),
        "Gaussian Naive Bayes": Pipeline([("scaler", StandardScaler()), ("model", GaussianNB())]),
        "SVM": Pipeline([("scaler", StandardScaler()), ("model", SVC(probability=True, random_state=42))]),
        "Random Forest": Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=300, random_state=42))]),
    }
    folds = min(3, int(pd.Series(y).value_counts().min()))
    comparisons = {}
    for name, candidate in candidates.items():
        candidate.fit(X_train, y_train)
        comparisons[name] = _metrics(candidate, X_train, y_train, X_test, y_test, folds)
    # Random Forest is the deployed model.  Its validation metrics are calculated
    # before fitting the final artifact, so the held-out records are never used to
    # report performance.  Once that evaluation is complete, refit on every
    # validated row: deployment should make use of the entire authoritative CSV.
    evaluation_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=500, random_state=42)),
    ])
    evaluation_pipeline.fit(X_train, y_train)
    deployed = _metrics(evaluation_pipeline, X_train, y_train, X_test, y_test, folds)

    scaler = StandardScaler().fit(X)
    model = RandomForestClassifier(n_estimators=500, random_state=42)
    model.fit(scaler.transform(X), y)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "rf_crop_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(encoder, MODEL_DIR / "label_encoder.pkl")
    report = {"trained_at": datetime.now(timezone.utc).isoformat(), "feature_order": FEATURE_ORDER,
              "dataset": {"records": len(df), "crop_classes": int(df.label.nunique()), "records_per_crop": df.label.value_counts().sort_index().to_dict()},
              "deployed_model": "Random Forest", "final_training_records": len(X),
              "metrics": deployed, "comparisons": comparisons}
    (MODEL_DIR / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def ensure_model() -> dict:
    metrics = MODEL_DIR / "metrics.json"
    return json.loads(metrics.read_text(encoding="utf-8")) if metrics.exists() else train_and_save()


def predict(values: dict) -> tuple[str, float]:
    crop, confidence = predict_top(values, limit=1)[0]
    return crop, confidence


def predict_top(values: dict, limit: int = 3) -> list[tuple[str, float]]:
    """Return the highest-probability crop matches in descending order."""
    ensure_model()
    features = [[values[key] for key in FEATURE_ORDER]]
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    model = joblib.load(MODEL_DIR / "rf_crop_model.pkl")
    encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    probabilities = model.predict_proba(scaler.transform(features))[0]
    indices = np.argsort(probabilities)[::-1][:limit]
    labels = encoder.inverse_transform(model.classes_[indices])
    return [(str(label), float(probabilities[index])) for label, index in zip(labels, indices)]
