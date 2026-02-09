import os
import re
import argparse
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def label_from_rating(rating):
    try:
        r = float(rating)
    except Exception:
        return None

    if r <= 2:
        return "negative"
    if r == 3:
        return "neutral"
    return "positive"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--text_col", type=str, required=True)
    parser.add_argument("--label_col", type=str, default="sentiment")
    parser.add_argument("--use_rating", action="store_true")
    parser.add_argument("--rating_col", type=str, default="rating")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"File not found: {args.data_path}")

    df = pd.read_csv(args.data_path)

    if args.text_col not in df.columns:
        raise ValueError(
            f"Text column '{args.text_col}' not found. Available columns: {list(df.columns)}"
        )

    if args.use_rating:
        if args.rating_col not in df.columns:
            raise ValueError(
                f"Rating column '{args.rating_col}' not found. Available columns: {list(df.columns)}"
            )
        df["label"] = df[args.rating_col].apply(label_from_rating)
    else:
        if args.label_col not in df.columns:
            raise ValueError(
                f"Label column '{args.label_col}' not found. "
                f"Either set --label_col correctly or use --use_rating."
            )
        df["label"] = df[args.label_col].astype(str).str.lower().str.strip()

    df["text"] = df[args.text_col].apply(clean_text)

    df = df[(df["text"].str.len() > 0) & (df["label"].notna())]
    df = df[df["label"].isin(["positive", "negative", "neutral"])]

    if len(df) < 50:
        raise ValueError("Too few rows after cleaning. Check your column names and label values.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"]
    )

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = LogisticRegression(max_iter=300, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)

    print("\n=== RESULTS ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, zero_division=0))

    os.makedirs("model/artifacts", exist_ok=True)
    joblib.dump(tfidf, "model/artifacts/tfidf.pkl")
    joblib.dump(model, "model/artifacts/sentiment_model.pkl")

    print("\nSaved artifacts:")
    print(" - model/artifacts/tfidf.pkl")
    print(" - model/artifacts/sentiment_model.pkl")


if __name__ == "__main__":
    main()
