import argparse
import os
import re
import joblib

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

def main():
    parser = argparse.ArgumentParser(description="Run sentiment prediction using saved artifacts (demo-safe)")
    parser.add_argument("--text", required=True, help="Input text to classify")
    parser.add_argument("--artifacts_dir", default="model/artifacts", help="Path to artifacts directory")
    parser.add_argument("--min_confidence", type=float, default=0.55,
                        help="If model confidence is below this, return 'neutral' (default: 0.55)")
    args = parser.parse_args()

    tfidf_path = os.path.join(args.artifacts_dir, "tfidf.pkl")
    model_path = os.path.join(args.artifacts_dir, "sentiment_model.pkl")

    if not os.path.exists(tfidf_path):
        raise FileNotFoundError(f"Missing: {tfidf_path}. Train first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing: {model_path}. Train first.")

    text = args.text.strip()
    if not text:
        print({"sentiment": None, "error": "Empty text"})
        return

    override = keyword_override(text)
    if override is not None:
        print({"sentiment": override, "source": "keyword_override"})
        return

    tfidf = joblib.load(tfidf_path)
    model = joblib.load(model_path)

    X = tfidf.transform([text])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = list(model.classes_)
        best_idx = int(probs.argmax())
        best_label = classes[best_idx]
        best_conf = float(probs[best_idx])

        if best_conf < args.min_confidence:
            print({"sentiment": "neutral", "source": "confidence_gate", "confidence": best_conf})
        else:
            print({"sentiment": best_label, "source": "model", "confidence": best_conf})
    else:
        pred = model.predict(X)[0]
        print({"sentiment": pred, "source": "model_no_proba"})

if __name__ == "__main__":
    main()