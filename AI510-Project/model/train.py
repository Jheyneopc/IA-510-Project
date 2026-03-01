import os
import re
import json
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
    if 2.5 <= r <= 3.5:
        return "neutral"
    return "positive"

def safe_str(x) -> str:
    return x if isinstance(x, str) else ""

def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF + Logistic Regression sentiment model")

    # Data
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--text_col", type=str, required=True, help="Main text column (e.g., review_description)")
    parser.add_argument("--title_col", type=str, default=None, help="Optional title column to concatenate (e.g., review_title)")

    # Labels
    parser.add_argument("--label_col", type=str, default="sentiment", help="Label column if not using rating")
    parser.add_argument("--use_rating", action="store_true", help="Create sentiment label from rating column")
    parser.add_argument("--rating_col", type=str, default="rating", help="Rating column name if --use_rating")

    # Split
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    # Model parameters
    parser.add_argument("--max_features", type=int, default=5000)
    parser.add_argument("--ngram_min", type=int, default=1)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--max_iter", type=int, default=300)

    # Output
    parser.add_argument("--out_dir", type=str, default="model/artifacts", help="Directory to save artifacts")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"File not found: {args.data_path}")

    df = pd.read_csv(args.data_path)

    # Validate columns
    if args.text_col not in df.columns:
        raise ValueError(f"Text column '{args.text_col}' not found. Available: {list(df.columns)}")

    if args.title_col is not None and args.title_col not in df.columns:
        raise ValueError(f"Title column '{args.title_col}' not found. Available: {list(df.columns)}")

    # Build label
    if args.use_rating:
        if args.rating_col not in df.columns:
            raise ValueError(f"Rating column '{args.rating_col}' not found. Available: {list(df.columns)}")
        df["label"] = df[args.rating_col].apply(label_from_rating)
    else:
        if args.label_col not in df.columns:
            raise ValueError(
                f"Label column '{args.label_col}' not found. Either set --label_col correctly or use --use_rating."
            )
        df["label"] = df[args.label_col].astype(str).str.lower().str.strip()

    # Build combined text (optional title + description)
    if args.title_col:
        combined = df[args.title_col].apply(safe_str) + " " + df[args.text_col].apply(safe_str)
        df["text_raw"] = combined.str.strip()
    else:
        df["text_raw"] = df[args.text_col].apply(safe_str).str.strip()

    df["text"] = df["text_raw"].apply(clean_text)

    # Filter valid rows
    df = df[(df["text"].str.len() > 0) & (df["label"].notna())].copy()
    df = df[df["label"].isin(["positive", "negative", "neutral"])].copy()

    if len(df) < 50:
        raise ValueError("Too few rows after cleaning. Check column names and label values.")

    # Report dataset stats
    label_counts = df["label"].value_counts().to_dict()
    print("\n=== DATASET SUMMARY ===")
    print(f"Rows after cleaning: {len(df)}")
    print("Label distribution:", label_counts)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"],
    )

    # Vectorize
    tfidf = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(args.ngram_min, args.ngram_max),
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    # Train model
    model = LogisticRegression(
        max_iter=args.max_iter,
        class_weight="balanced",
        solver="lbfgs",
    )
    model.fit(X_train_vec, y_train)

    # Evaluate
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)

    labels_order = ["negative", "neutral", "positive"]
    cm = confusion_matrix(y_test, preds, labels=labels_order)
    report_text = classification_report(y_test, preds, zero_division=0)
    report_dict = classification_report(y_test, preds, zero_division=0, output_dict=True)

    print("\n=== RESULTS ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report_text)
    print("\nConfusion Matrix (rows=true, cols=pred) labels:", labels_order)
    print(cm)

    # Save artifacts
    os.makedirs(args.out_dir, exist_ok=True)
    tfidf_path = os.path.join(args.out_dir, "tfidf.pkl")
    model_path = os.path.join(args.out_dir, "sentiment_model.pkl")
    metrics_path = os.path.join(args.out_dir, "metrics.json")

    joblib.dump(tfidf, tfidf_path)
    joblib.dump(model, model_path)

    metrics = {
        "accuracy": float(acc),
        "labels_order": labels_order,
        "confusion_matrix": cm.tolist(),
        "report": report_dict,
        "rows_after_cleaning": int(len(df)),
        "label_distribution": label_counts,
        "test_size": float(args.test_size),
        "random_state": int(args.random_state),
        "data": {
            "data_path": args.data_path,
            "text_col": args.text_col,
            "title_col": args.title_col,
            "use_rating": bool(args.use_rating),
            "rating_col": args.rating_col,
            "label_col": args.label_col,
        },
        "tfidf": {
            "max_features": int(args.max_features),
            "ngram_range": [int(args.ngram_min), int(args.ngram_max)],
        },
        "model": {
            "type": "LogisticRegression",
            "max_iter": int(args.max_iter),
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved artifacts:")
    print(f" - {tfidf_path}")
    print(f" - {model_path}")
    print(f" - {metrics_path}")

if __name__ == "__main__":
    main()