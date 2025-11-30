# Customer Churn MLOps pipeline

## Purpose
Complete workflow for predicting customer churn in a telecom context: data preparation, model training, experiment tracking, API service, UI, and containerization.

## Features

### Data & Modeling
- Data preprocessing and feature engineering.
- XGBoost classifier trained for churn prediction.
- Experiments tracked with MLflow (metrics, parameters, artifacts).

### Model Tracking
- MLflow experiment storing all runs.
- Logged serialized model and feature metadata.

### API Service (FastAPI)
- `POST /predict` endpoint for model inference.
- `GET /` health check endpoint.

### Web UI (Gradio)
- Simple interface for manually testing predictions.
- Mounted at `/ui`.

### Containerization (Docker)
- Dockerized FastAPI application with Uvicorn.
- Entrypoint: `src.app.app:app`.
- Exposes port `8000`.

### CI/CD (GitHub Actions)
- Pipeline builds the Docker image.
- Pushes the image to Docker Hub.

## Benefits
- Predicts customers likely to churn.
- Model accessible without notebooks (API + UI).
- Reproducible experiments with MLflow.
- Consistent deployments using Docker and CI/CD.


## Commands

### Training Pipeline
```bash
# Run the complete ML training pipeline
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn

# Prepare processed data only
python scripts/prepare_processed_data.py
```

### Testing
```bash
# Test data processing and feature engineering
python scripts/test_pipeline_phase1_data_features.py

# Test model training and evaluation
python scripts/test_pipeline_phase2_modeling.py

# Test FastAPI endpoints
python scripts/test_fastapi.py
```

### Local Development
```bash

# Alternative app entry point
python -m uvicorn src.app.app:app --host 0.0.0.0 --port 8000
Then access the UI at: http://localhost:8000/ui or http://127.0.0.1:8000/ui/
```

### Docker
```bash
# Build and run the containerized application
docker build -t telco-churn-app .
docker run -p 8000:8000 telco-churn-app
Then access the UI at: http://localhost:8000/ui or http://127.0.0.1:8000/ui/
```
