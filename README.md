Sentiment Analysis Pipeline with CI/CD and Cloud Deployment
Project Overview

This project implements a production-oriented sentiment analysis system that classifies Google Play Store user reviews into positive, neutral, and negative sentiments.

The system combines classical NLP techniques with modern MLOps practices, demonstrating how a machine learning model can move from experimentation to a reproducible, deployable service.

Key components include:

TF-IDF feature extraction

Logistic Regression classifier

Automated model training and evaluation

REST API deployment using FastAPI

Interactive browser-based UI

Docker containerization

CI/CD automation using GitHub Actions

Cloud deployment using Azure Container Apps

The final pipeline demonstrates how AI systems can be built, validated, deployed, and operated in a cloud environment.

System Architecture

The system follows a modular architecture separating training, inference, deployment, and automation.

Dataset
   │
   ▼
Data Preprocessing
(cleaning + label generation)
   │
   ▼
TF-IDF Vectorization
   │
   ▼
Logistic Regression Model
   │
   ▼
Model Artifacts
tfidf.pkl
sentiment_model.pkl
metrics.json
   │
   ▼
FastAPI Inference API
   │
   ├── Browser UI
   │
   └── REST Endpoint (/predict)
   │
   ▼
Docker Container
   │
   ▼
Azure Container Apps Deployment
   │
   ▼
Public Sentiment Prediction Service
Repository Structure
AI510-Project
│
├── dataset/
│   └── GooglePlay_App_Data.csv
│
├── model/
│   ├── artifacts/
│   ├── predict.py
│   ├── train.py
│   └── requirements.txt
│
├── ui/
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
├── app.py
├── Dockerfile
├── README.md
│
└── .github/
    └── workflows/
        ├── ci.yml
        └── azure-deploy.yml
Dataset

The dataset consists of Google Play Store user reviews.

Each record includes:

review title

review description

rating (1–5)

Sentiment labels are derived from ratings:

Rating	Sentiment
≤ 2	Negative
2.5 – 3.5	Neutral
> 3.5	Positive

Dataset source:

https://www.kaggle.com/datasets/lava18/google-play-store-apps

Model Training

To train the sentiment model locally:

python model/train.py \
  --data_path dataset/GooglePlay_App_Data.csv \
  --text_col review_description \
  --title_col review_title \
  --use_rating \
  --rating_col rating

Training will generate artifacts:

model/artifacts/tfidf.pkl
model/artifacts/sentiment_model.pkl
model/artifacts/metrics.json

The metrics file includes:

accuracy

macro F1 score

weighted F1 score

confusion matrix

dataset statistics

Running the API Locally

Start the FastAPI server:

uvicorn app:app --reload --host 0.0.0.0 --port 8000

Open browser:

http://localhost:8000
API Endpoints
Health Check
GET /health

Example response:

{
  "status": "ok",
  "artifacts_exist": true,
  "model_loaded": true
}
Model Info
GET /info
Sentiment Prediction
POST /predict

Request example:

{
  "text": "This app works perfectly!"
}

Response example:

{
  "sentiment": "positive",
  "source": "model",
  "confidence": 0.82
}
Web Interface

The system includes a browser-based interface allowing real-time predictions.

Features:

review text input

sentiment prediction display

confidence score

UI color indicators

Access:

http://localhost:8000
Hybrid Inference Mechanism

The system uses a hybrid prediction strategy to improve reliability.

Keyword override

Strong sentiment keywords trigger deterministic predictions.

Example:

"worst app ever" → negative

Confidence gating

If model confidence is below a threshold:

confidence < 0.55

Prediction defaults to neutral.

This reduces unstable predictions caused by dataset imbalance.

Docker Deployment

Build container:

docker build -t sentiment-api .

Run container:

docker run -p 8000:8000 sentiment-api

Open:

http://localhost:8000
CI/CD Pipeline

The project integrates GitHub Actions for automated validation.

The CI workflow performs:

Dependency installation

Model retraining

Artifact verification

Performance validation

Docker build testing

The pipeline enforces a minimum accuracy threshold to prevent model degradation.

Azure Cloud Deployment

The project deploys automatically to Azure Container Apps.

Deployment pipeline:

GitHub Push
     │
     ▼
GitHub Actions CI
     │
Train Model
Validate Metrics
Build Docker Image
     │
     ▼
Azure Container Apps Deployment
     │
     ▼
Public API Endpoint

Environment variables used in Azure:

PORT=8000
ARTIFACTS_DIR=model/artifacts
MIN_CONFIDENCE=0.55
APP_VERSION=1.3

After deployment the application becomes publicly accessible through the Azure endpoint.

Reproducibility

The pipeline ensures reproducibility through:

deterministic label generation

serialized model artifacts

CI validation

containerized execution environment

automated deployment pipeline

This guarantees consistent behavior across development, testing, and production environments.

Technologies Used

Python

Scikit-learn

FastAPI

Docker

GitHub Actions

Azure Container Apps

Pandas

Joblib

Future Improvements

Potential extensions include:

transformer-based models (BERT)

dataset balancing techniques

model monitoring and drift detection

experiment tracking using MLflow

multilingual sentiment analysis

automated retraining pipelines

Authors

Ashwin Shastry Paturi
Jheyne de Oliveira Panta Cordeiro
Salvador Eng Deng
Shagun Sharma Tamta

AI 510 – Artificial Intelligence in Cloud Computing
City University of Seattle

Final Note

This project demonstrates how machine learning systems can transition from experimentation to cloud deployment using MLOps practices, combining classical NLP models with modern automation and containerization workflows.