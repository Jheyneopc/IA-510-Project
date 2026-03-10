📘 AI510 Project
Automated Sentiment Analysis Pipeline with CI/CD and Web UI
📌 Overview

This project implements a complete end-to-end sentiment analysis system using Google Play Store reviews. The system integrates:

Data preprocessing and label generation

TF-IDF feature extraction

Logistic Regression classifier

Model artifact serialization

REST API using FastAPI

Vibrant web-based UI (browser input → browser output)

Docker containerization

GitHub Actions CI/CD pipeline with accuracy validation

The project demonstrates how a machine learning model transitions from experimental training to a production-style, automated, deployable system.

🏗 Project Architecture
AI510-Project/
│
├── .github/workflows/ci.yml      # CI/CD automation
│
├── dataset/
│   └── GooglePlay_App_Data.csv
│
├── model/
│   ├── train.py                  # Training pipeline
│   ├── predict.py                # CLI prediction
│   ├── requirements.txt          # Dependencies
│   └── artifacts/                # Generated model files
│
├── ui/                           # Vibrant web UI
│   ├── index.html
│   ├── style.css
│   └── app.js
│
├── app.py                        # FastAPI backend + UI serving
├── Dockerfile                    # Container configuration
├── README.md
└── .gitignore
🧠 Model Description

The system uses:

TF-IDF vectorization (1–2 grams, max 5000 features)

Logistic Regression classifier

Class balancing enabled

Confidence gating mechanism

Keyword-based override for demo stability

Sentiment classes:

Positive

Neutral

Negative

For live demo stability:

Strong positive/negative phrases trigger keyword override.

Low-confidence predictions default to neutral.

🚀 Running in GitHub Codespaces (Recommended Demo)
1️⃣ Install Dependencies
pip install -r model/requirements.txt
2️⃣ Train the Model
python model/train.py \
  --data_path dataset/GooglePlay_App_Data.csv \
  --text_col review_description \
  --title_col review_title \
  --use_rating \
  --rating_col rating

Artifacts created in:

model/artifacts/
├── tfidf.pkl
├── sentiment_model.pkl
└── metrics.json
3️⃣ Start the API + UI Server

⚠️ Important for Codespaces: Use 0.0.0.0

uvicorn app:app --reload --host 0.0.0.0 --port 8000
4️⃣ Open the Web UI

Go to Ports tab in Codespaces

Find port 8000

Click Open in Browser

Or manually open:

https://<your-codespace-url>/ 
🎨 Web UI Features

The UI allows:

Direct review text input

Instant sentiment prediction

Confidence score display

Source explanation (model / keyword override / confidence gate)

Adjustable neutral sensitivity slider

Clean, vibrant visual feedback

No terminal commands required during demo.

🧪 Demo Inputs (Guaranteed Distinct Outputs)

Use these for safe presentation:

✅ Positive
This app is amazing and super useful. Great experience!
⚖️ Neutral
It is okay, nothing special. Works sometimes.
❌ Negative
Worst app ever. Keeps crashing and freezing. Totally broken.
🖥 CLI Usage (Optional)
Train
python model/train.py --data_path dataset/GooglePlay_App_Data.csv \
  --text_col review_description \
  --title_col review_title \
  --use_rating \
  --rating_col rating
Predict
python model/predict.py --text "This app is amazing!"
🔌 API Endpoints
Root (UI)
GET /
Health Check
GET /health
Predict
POST /predict

Example:

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Worst app ever. Keeps crashing."}'
🐳 Docker Deployment
Build
docker build -t sentiment-api .
Run (Avoid Port Conflict in Codespaces)

If port 8000 is already in use:

docker run -e PORT=8000 -p 8001:8000 sentiment-api

Then open port 8001 in Codespaces.

🔁 CI/CD Workflow (GitHub Actions)

Workflow file:

.github/workflows/ci.yml

Automated steps:

Checkout repository

Install dependencies

Train model

Verify artifacts exist

Validate minimum accuracy threshold

Build Docker image

If accuracy drops below threshold, CI fails automatically.

📊 Evaluation Metrics

Generated in:

model/artifacts/metrics.json

Includes:

Accuracy

Precision

Recall

F1-score

Confusion matrix

Label distribution

Model parameters

⚙️ Technical Highlights

Classical NLP baseline (TF-IDF + Logistic Regression)

Class imbalance handling

Confidence-based neutral gating

Keyword override safety mechanism

Modular architecture

Production-style containerization

Automated validation pipeline

🛡 Ethical AI Considerations

Deterministic rating-to-label mapping

No personal data usage

Transparent evaluation metrics

Class imbalance explicitly reported

Explainable linear model architecture

🧩 Future Improvements

Transformer-based models (e.g., BERT)

Cloud deployment (AWS/Azure)

Model monitoring & drift detection

Expanded dataset for better class balance

Multi-language sentiment support

👥 Team Notes

For a full demo:

Train model

Run server

Open UI

Enter demo inputs

Show confidence + explanation

Optionally show CI passing in GitHub

Optionally show Docker container running

No manual terminal prediction is required during presentation.

📄 License

## Azure Container Apps Deployment

This project supports cloud deployment using Azure Container Apps with Azure Container Registry.

### Required GitHub Secrets
Add these in **GitHub → Settings → Secrets and variables → Actions**:

- `AZURE_CREDENTIALS`
- `AZURE_RG`
- `AZURE_ACR_NAME`
- `AZURE_CONTAINERAPP_ENV`
- `AZURE_CONTAINERAPP_NAME`

### Deployment Behavior
On every push to `main`, GitHub Actions will:

- Build the Docker image from `AI510-Project/`
- Push it through the Azure deploy action
- Deploy the application to Azure Container Apps

### Runtime Environment Variables
The deployment uses:

- `PORT=8000`
- `ARTIFACTS_DIR=model/artifacts`
- `MIN_CONFIDENCE=0.55`

### Cloud Endpoints
After deployment, the Azure URL supports:

- `/` → Web UI
- `/predict` → Sentiment prediction API
- `/health` → Health check
- `/docs` → Swagger API documentation

Academic project for AI 510 coursework.

✅ Final Status

✔ Training works
✔ Prediction works
✔ UI works
✔ Docker works
✔ CI works
✔ Codespaces compatible
✔ Demo-safe outputs