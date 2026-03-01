import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "model/artifacts")
TFIDF_PATH = os.path.join(ARTIFACTS_DIR, "tfidf.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "sentiment_model.pkl")

def load_artifacts():
    if not os.path.exists(TFIDF_PATH) or not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Artifacts not found. Train the model first.\n"
            f"Expected:\n- {TFIDF_PATH}\n- {MODEL_PATH}"
        )
    tfidf = joblib.load(TFIDF_PATH)
    model = joblib.load(MODEL_PATH)
    return tfidf, model

tfidf, model = load_artifacts()
app = FastAPI(title="Sentiment Analysis API", version="1.0")

class PredictRequest(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    text = (req.text or "").strip()
    if not text:
        return {"error": "Empty text", "sentiment": None}

    X = tfidf.transform([text])
    pred = model.predict(X)[0]
    return {"sentiment": pred}