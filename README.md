# IA-510-Project
# Automated Sentiment Analysis Pipeline with CI/CD Deployment

This project implements an end-to-end sentiment analysis pipeline:
- Data cleaning + labeling (from ratings)
- TF-IDF feature extraction
- Logistic Regression training
- Artifact export (vectorizer + model + metrics)
- FastAPI inference service
- Docker containerization
- GitHub Actions CI to train + validate model quality

## 1) Install dependencies
```bash
pip install -r model/requirements.txt

## Model Training Result
