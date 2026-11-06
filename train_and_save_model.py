import io
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DEFAULT_BUCKET = "gh-actions-lab-j4roqu-models"
DATASET_NAME = "wine"
RANDOM_STATE = 42
CV_FOLDS = 5


def download_data():
    """Load the Wine dataset from scikit-learn (178 samples, 13 features, 3 classes)."""
    wine = load_wine()
    features = pd.DataFrame(wine.data, columns=wine.feature_names)
    target = pd.Series(wine.target)
    return features, target


def preprocess_data(X, y):
    """Stratified 80/20 split with a fixed seed for reproducibility."""
    return train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )


def candidate_models():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def select_model(X_train, y_train):
    """Score every candidate with stratified k-fold CV and return the best one, refit on all training data."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    for name, model in candidate_models().items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        cv_results[name] = {
            "cv_mean": float(np.mean(scores)),
            "cv_std": float(np.std(scores)),
        }

    # Highest mean CV accuracy wins; ties go to the lower std, then to the
    # earlier candidate (max() returns the first of equal keys).
    best_name = max(
        cv_results,
        key=lambda n: (cv_results[n]["cv_mean"], -cv_results[n]["cv_std"]),
    )
    best_model = candidate_models()[best_name]
    best_model.fit(X_train, y_train)
    return best_name, best_model, cv_results


def print_comparison(cv_results, best_name):
    print(f"\n{CV_FOLDS}-fold CV accuracy on training set:")
    print(f"  {'model':<22}{'mean':>8}{'std':>8}")
    for name, r in sorted(cv_results.items(), key=lambda kv: -kv[1]["cv_mean"]):
        marker = "  <- selected" if name == best_name else ""
        print(f"  {name:<22}{r['cv_mean']:>8.4f}{r['cv_std']:>8.4f}{marker}")
    print()


def build_metrics(cv_results, best_name, y_test, y_pred, timestamp):
    return {
        "dataset": DATASET_NAME,
        "timestamp_utc": timestamp,
        "cv_folds": CV_FOLDS,
        "cv_results": cv_results,
        "chosen_model": best_name,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def save_model_to_gcs(model, bucket_name, blob_name):
    """Serialize the model in memory and upload it to GCS.

    Errors are not swallowed, so a failed upload fails the workflow run.
    """
    from google.cloud import storage

    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)

    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    blob.upload_from_file(buffer, content_type="application/octet-stream")
    print(f"Model uploaded to gs://{bucket_name}/{blob_name}")


def main():
    bucket_name = os.getenv("GCS_BUCKET_NAME", DEFAULT_BUCKET)
    skip_upload = os.getenv("SKIP_GCS_UPLOAD", "").lower() in ("1", "true", "yes")

    X, y = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    best_name, model, cv_results = select_model(X_train, y_train)
    print_comparison(cv_results, best_name)

    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%d_%H%M%S")
    metrics = build_metrics(cv_results, best_name, y_test, model.predict(X_test), now.isoformat())
    print(f"Chosen model:  {best_name}")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test macro F1: {metrics['test_macro_f1']:.4f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")

    blob_name = f"models/model_{ts}.joblib"

    if skip_upload:
        joblib.dump(model, "model.joblib")
        print(f"SKIP_GCS_UPLOAD set: saved locally to model.joblib "
              f"(would upload to gs://{bucket_name}/{blob_name})")
        return

    save_model_to_gcs(model, bucket_name, blob_name)


if __name__ == "__main__":
    main()
