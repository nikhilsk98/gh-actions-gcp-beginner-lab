# Lab 3: GitHub Actions and GCP Connections

## Overview

In this lab I automated a small ML training pipeline with GitHub Actions and Google Cloud Platform. A workflow trains a `RandomForestClassifier` on the Iris dataset, authenticates to GCP with a service account, and uploads the trained model to a Google Cloud Storage bucket. It runs on a daily schedule and can also be triggered manually.

Based on the MLOps course lab: [Labs/Github_Labs/Lab3](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab3), using the [template repository](https://github.com/AshyScripts/github-actions-gcp-beginner-lab) as the starting point.

## My Setup

| Item | Value |
|---|---|
| GCP project ID | `<PROJECT_ID>` |
| Service account | `gh-actions-sa@<PROJECT_ID>.iam.gserviceaccount.com` |
| Service account role | `roles/storage.admin` |
| GCS bucket | `gh-actions-lab-j4roqu-models` |
| Bucket location | `us-east1` (single region, uniform bucket-level access) |
| GitHub secret | `GCP_SA_KEY` (service account JSON key) |
| Workflow file | `.github/workflows/train-and-upload.yml` |
| Python (CI) | 3.10 |

## Architecture

```
 daily cron (00:00 UTC) / manual workflow_dispatch
                 |
                 v
   GitHub Actions runner (ubuntu-latest)
                 |
   checkout -> setup Python 3.10 -> restore pip cache -> install deps
                 |
                 v
   google-github-actions/auth@v2  (uses secrets.GCP_SA_KEY)
                 |
                 v
   train_and_save_model.py
     - load Iris, 80/20 split, train RandomForest
     - print test accuracy
     - serialize model in memory with joblib
                 |
                 v
   gs://gh-actions-lab-j4roqu-models/models/model_YYYYMMDD_HHMMSS.joblib
```

## Project Structure

```
.
├── .github/workflows/train-and-upload.yml   # CI workflow
├── train_and_save_model.py                  # train, evaluate, upload
├── requirements.txt
└── .gitignore
```

## Steps to Reproduce

1. **Clone and set up the environment**
   ```bash
   git clone https://github.com/nikhilsk98/gh-actions-gcp-beginner-lab.git
   cd gh-actions-gcp-beginner-lab
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Test locally without uploading**
   ```bash
   SKIP_GCS_UPLOAD=1 python train_and_save_model.py
   ```

3. **Create GCP resources** (billing must be linked to the project)
   ```bash
   PROJECT_ID=<PROJECT_ID>
   BUCKET=<PROJECT_ID>-models
   gcloud config set project $PROJECT_ID
   gcloud services enable storage.googleapis.com iam.googleapis.com

   gcloud iam service-accounts create gh-actions-sa --display-name="GitHub Actions SA"
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:gh-actions-sa@$PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/storage.admin"

   gcloud storage buckets create gs://$BUCKET --location=us-east1 --uniform-bucket-level-access

   mkdir -p ~/gcp-keys
   gcloud iam service-accounts keys create ~/gcp-keys/gh-actions-sa.json \
     --iam-account=gh-actions-sa@$PROJECT_ID.iam.gserviceaccount.com
   ```

4. **Add the key as a GitHub secret** (the key file stays outside the repo)
   ```bash
   gh secret set GCP_SA_KEY < ~/gcp-keys/gh-actions-sa.json
   gh secret list
   ```

5. **Point the workflow at the bucket.** Set `GCS_BUCKET_NAME` in the `env:` block of `.github/workflows/train-and-upload.yml`.

6. **Run and verify**
   ```bash
   gh workflow run train-and-upload.yml
   gh run watch
   gcloud storage ls gs://$BUCKET/models/
   ```

## What I Changed from the Original Lab

- **Bucket name from an environment variable.** The template hardcoded the bucket. The script now reads `GCS_BUCKET_NAME` and falls back to a default, and the workflow sets it in its `env:` block.
- **Timestamped model versions.** Every run uploads to `models/model_YYYYMMDD_HHMMSS.joblib` (UTC), so earlier models are never overwritten.
- **Accuracy in the logs.** The script prints `Test accuracy: X.XXXX` so each run's result shows up in the Actions log.
- **In-memory upload.** The model is serialized to a `BytesIO` buffer and uploaded directly, so no temporary file is written on the runner.
- **Failed uploads fail the run.** Upload exceptions are not caught, so a GCS problem turns the workflow red instead of passing silently.
- **Local test mode.** `SKIP_GCS_UPLOAD=1` trains and saves the model locally without needing GCP credentials.
- **Workflow matches the lab README.** I renamed the file to `train-and-upload.yml` and re-enabled the daily cron trigger, which was commented out in the template.
- **Broader `.gitignore`.** It now covers JSON keys, `.env` files, and `*.joblib` / `*.pkl` model artifacts.

## Results

The manually triggered run passed every step. On the fixed 80/20 split (`random_state=42`) the model reached a test accuracy of **1.0000**, and the model was uploaded to:

```
gs://gh-actions-lab-j4roqu-models/models/model_20261106_054628.joblib
```

## Screenshots

### Successful GitHub Actions run
![Successful Actions run](screenshots/actions-run.png)

### GCS bucket contents
![GCS bucket contents](screenshots/gcs-bucket.png)

### GitHub secret configured (name only)
![GitHub secret](screenshots/github-secret.png)

## Security Notes

- The service account key is stored only as the encrypted GitHub secret `GCP_SA_KEY` and in a local folder outside the repository. `*.json` is git-ignored as an extra safeguard.
- A better long-term approach is Workload Identity Federation (keyless auth from GitHub OIDC), which removes long-lived JSON keys entirely.
