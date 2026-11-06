import io
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

DEFAULT_BUCKET = "gh-actions-lab-j4roqu-models"


def download_data():
    """Load the Iris dataset from scikit-learn."""
    iris = load_iris()
    features = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = pd.Series(iris.target)
    return features, target


def preprocess_data(X, y):
    """Split into train and test sets (fixed seed for reproducibility)."""
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


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
    model = train_model(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {accuracy:.4f}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    blob_name = f"models/model_{timestamp}.joblib"

    if skip_upload:
        joblib.dump(model, "model.joblib")
        print(f"SKIP_GCS_UPLOAD set: saved locally to model.joblib "
              f"(would upload to gs://{bucket_name}/{blob_name})")
        return

    save_model_to_gcs(model, bucket_name, blob_name)


if __name__ == "__main__":
    main()
