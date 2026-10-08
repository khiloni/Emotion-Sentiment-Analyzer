"""
train.py
--------
Reproduces the training pipeline from sentiment_emotion_analysis.ipynb and
saves everything the API server needs to make predictions, as versioned
artifacts under backend/artifacts/:

    word_vectorizer.joblib   TF-IDF (word 1-2 grams)
    char_vectorizer.joblib   TF-IDF (char 3-5 grams)
    model.joblib             OneVsRestClassifier(LogisticRegression)
    meta.json                emotion_cols, confidence threshold, metrics

Run this once (needs Kaggle credentials configured for `kagglehub`, exactly
like the original notebook):

    python train.py

It will take a few minutes on a laptop CPU. Re-run any time you want to
retrain (e.g. after tweaking hyperparameters below).
"""

import json
import os
import warnings

import joblib
import kagglehub
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, hamming_loss
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from nlp_pipeline import preprocess_text

warnings.filterwarnings("ignore")

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

ALL_POSSIBLE_EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]


def load_dataset() -> pd.DataFrame:
    """Download GoEmotions via kagglehub and locate the labelled CSV."""
    path = kagglehub.dataset_download("debarshichanda/goemotions")
    print(f"Dataset downloaded to: {path}")

    csv_file = None
    for root, _dirs, files in os.walk(path):
        for f in files:
            if f.startswith("goemotions_") and f.endswith(".csv"):
                csv_file = os.path.join(root, f)
                break
        if csv_file:
            break

    if not csv_file:
        for root, _dirs, files in os.walk(path):
            for f in files:
                if f.endswith(".csv"):
                    candidate = os.path.join(root, f)
                    sample = pd.read_csv(candidate, nrows=2)
                    if "text" in sample.columns or "Text" in sample.columns:
                        csv_file = candidate
                        break
            if csv_file:
                break

    if not csv_file:
        raise FileNotFoundError("Could not locate a GoEmotions CSV inside the downloaded dataset.")

    df = pd.read_csv(csv_file)
    df.columns = [c.lower() for c in df.columns]
    return df


def main():
    df = load_dataset()

    emotion_cols = [c for c in ALL_POSSIBLE_EMOTIONS if c in df.columns]
    print(f"Emotion columns found ({len(emotion_cols)}): {emotion_cols}")

    # --- Preprocess (spell correction OFF for training speed) ---
    print("Preprocessing dataset...")
    df["cleaned_text"] = df["text"].apply(lambda t: preprocess_text(t, apply_spell_correction=False))

    # --- Feature extraction ---
    word_vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), max_features=25000,
        sublinear_tf=True, min_df=2, token_pattern=r"(?u)\b\w\w+\b|__\w+__",
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), max_features=10000,
        sublinear_tf=True, min_df=3,
    )

    print("Fitting TF-IDF vectorizers...")
    X_word = word_vectorizer.fit_transform(df["cleaned_text"])
    X_char = char_vectorizer.fit_transform(df["cleaned_text"])
    X = hstack([X_word, X_char]).tocsr()
    Y = df[emotion_cols].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train.sort_indices()
    X_test.sort_indices()

    # --- Train ---
    print("Training model...")
    base_classifier = LogisticRegression(
        solver="liblinear", C=1.0, class_weight="balanced", max_iter=500, random_state=42,
    )
    model = OneVsRestClassifier(base_classifier, n_jobs=-1)
    model.fit(X_train, Y_train)

    # --- Evaluate ---
    Y_pred = model.predict(X_test)
    metrics = {
        "exact_match_accuracy": accuracy_score(Y_test, Y_pred),
        "hamming_loss": hamming_loss(Y_test, Y_pred),
        "f1_macro": f1_score(Y_test, Y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(Y_test, Y_pred, average="micro", zero_division=0),
        "f1_weighted": f1_score(Y_test, Y_pred, average="weighted", zero_division=0),
    }
    print("=" * 58)
    print("MODEL EVALUATION RESULTS")
    print("=" * 58)
    for k, v in metrics.items():
        print(f"{k:22s}: {v:.4f}")
    print()
    print(classification_report(Y_test, Y_pred, target_names=emotion_cols, zero_division=0))

    # --- Data-driven confidence threshold ---
    Y_test_proba = model.predict_proba(X_test)
    max_probs = np.max(Y_test_proba, axis=1)
    confidence_threshold = float(np.percentile(max_probs, 25))
    confidence_threshold = max(0.10, min(confidence_threshold, 0.35))
    secondary_gap_ratio = 0.75
    print(f"Confidence threshold: {confidence_threshold:.3f}")

    # --- Save artifacts ---
    joblib.dump(word_vectorizer, os.path.join(ARTIFACT_DIR, "word_vectorizer.joblib"))
    joblib.dump(char_vectorizer, os.path.join(ARTIFACT_DIR, "char_vectorizer.joblib"))
    joblib.dump(model, os.path.join(ARTIFACT_DIR, "model.joblib"))

    meta = {
        "emotion_cols": emotion_cols,
        "confidence_threshold": confidence_threshold,
        "secondary_gap_ratio": secondary_gap_ratio,
        "metrics": metrics,
    }
    with open(os.path.join(ARTIFACT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nArtifacts saved to {ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
