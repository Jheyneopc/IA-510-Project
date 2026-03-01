import argparse
import os
import joblib

def main():
    parser = argparse.ArgumentParser(description="Run sentiment prediction using saved artifacts")
    parser.add_argument("--text", required=True, help="Input text to classify")
    parser.add_argument("--artifacts_dir", default="model/artifacts", help="Path to artifacts directory")
    args = parser.parse_args()

    tfidf_path = os.path.join(args.artifacts_dir, "tfidf.pkl")
    model_path = os.path.join(args.artifacts_dir, "sentiment_model.pkl")

    if not os.path.exists(tfidf_path):
        raise FileNotFoundError(f"Missing: {tfidf_path}. Train first.")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing: {model_path}. Train first.")

    tfidf = joblib.load(tfidf_path)
    model = joblib.load(model_path)

    X = tfidf.transform([args.text])
    pred = model.predict(X)[0]

    print({"sentiment": pred})

if __name__ == "__main__":
    main()