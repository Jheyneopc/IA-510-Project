import os
import re
import joblib
import uvicorn
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "model/artifacts")
DEFAULT_MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.55"))
APP_VERSION = os.getenv("APP_VERSION", "1.3")
DEPLOY_ENV = os.getenv("DEPLOY_ENV", "local")
CLOUD_PROVIDER = os.getenv("CLOUD_PROVIDER", "none")

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

app = FastAPI(title="Sentiment Analysis API", version=APP_VERSION)

app.mount("/ui", StaticFiles(directory="ui"), name="ui")

tfidf = None
model = None


class PredictRequest(BaseModel):
    text: str
    min_confidence: Optional[float] = None


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
    global tfidf, model

    if not os.path.exists(TFIDF_PATH) or not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Artifacts not found.\n"
            f"Expected:\n- {TFIDF_PATH}\n- {MODEL_PATH}"
        )

    tfidf = joblib.load(TFIDF_PATH)
    model = joblib.load(MODEL_PATH)


@app.on_event("startup")
def startup_event():
    load_artifacts()


@app.get("/")
def ui_home():
    return FileResponse("ui/index.html")


@app.get("/health")
def health():
    artifacts_exist = os.path.exists(TFIDF_PATH) and os.path.exists(MODEL_PATH)
    model_loaded = tfidf is not None and model is not None

    return {
        "status": "ok" if artifacts_exist and model_loaded else "degraded",
        "artifacts_exist": artifacts_exist,
        "model_loaded": model_loaded
    }


@app.get("/info")
def info():
    return {
        "app": "Sentiment Analysis API",
        "version": APP_VERSION,
        "artifacts_dir": ARTIFACTS_DIR,
        "default_min_confidence": DEFAULT_MIN_CONFIDENCE,
        "ui_available": True,
        "deploy_env": DEPLOY_ENV,
        "cloud_provider": CLOUD_PROVIDER
    }


@app.post("/predict")
def predict(req: PredictRequest):
    global tfidf, model

    if tfidf is None or model is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text input.")

    override = keyword_override(text)
    if override is not None:
        return {
            "sentiment": override,
            "source": "keyword_override"
        }

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
            return {
                "sentiment": "neutral",
                "source": "confidence_gate",
                "confidence": best_conf
            }

        return {
            "sentiment": best_label,
            "source": "model",
            "confidence": best_conf
        }

    pred = model.predict(X)[0]
    return {
        "sentiment": pred,
        "source": "model_no_proba"
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)