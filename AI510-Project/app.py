import os
import re
import joblib
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "model/artifacts")
DEFAULT_MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.55"))

TFIDF_PATH = os.path.join(ARTIFACTS_DIR, "tfidf.pkl")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "sentiment_model.pkl")

POSITIVE_HINTS = {
    "love", "great", "amazing", "awesome", "excellent", "fantastic", "perfect",
    "good", "wonderful", "smooth", "helpful", "best"
}
NEGATIVE_HINTS = {
    "hate", "worst", "bad", "terrible", "awful", "crash", "crashing", "bug",
    "broken", "useless", "scam", "error", "doesn't work", "does not work",
    "slow", "freeze", "freezing", "lag", "laggy"
}

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())

def keyword_override(text: str):
    t = normalize(text)
    for phrase in ["doesn't work", "does not work"]:
        if phrase in t:
            return "negative"
    tokens = set(re.findall(r"[a-z']+", t))
    if tokens & NEGATIVE_HINTS:
        return "negative"
    if tokens & POSITIVE_HINTS:
        return "positive"
    return None

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

app = FastAPI(title="Sentiment Analysis API", version="1.2")

# Serve UI static files at /ui, and main UI at /
app.mount("/ui", StaticFiles(directory="ui"), name="ui")

class PredictRequest(BaseModel):
    text: str
    min_confidence: Optional[float] = None

@app.get("/")
def ui_home():
    return FileResponse("ui/index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    text = (req.text or "").strip()
    if not text:
        return {"sentiment": None, "error": "Empty text"}

    # 1) Keyword override for demo stability
    override = keyword_override(text)
    if override is not None:
        return {"sentiment": override, "source": "keyword_override"}

    # 2) Confidence gate (low confidence -> neutral)
    min_conf = req.min_confidence if req.min_confidence is not None else DEFAULT_MIN_CONFIDENCE
    try:
        min_conf = float(min_conf)
    except Exception:
        min_conf = DEFAULT_MIN_CONFIDENCE
    min_conf = max(0.0, min(1.0, min_conf))

    X = tfidf.transform([text])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = list(model.classes_)
        best_idx = int(probs.argmax())
        best_label = classes[best_idx]
        best_conf = float(probs[best_idx])

        if best_conf < min_conf:
            return {"sentiment": "neutral", "source": "confidence_gate", "confidence": best_conf}
        return {"sentiment": best_label, "source": "model", "confidence": best_conf}

    # Fallback
    pred = model.predict(X)[0]
    return {"sentiment": pred, "source": "model_no_proba"}