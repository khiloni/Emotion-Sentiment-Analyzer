"""
predict.py
----------
Loads the artifacts produced by train.py and exposes
`predict_emotion_and_sentiment`, matching the notebook's prediction logic
exactly (confidence-gated primary emotion, optional secondary emotion,
sentiment derived from the primary emotion).

Artifacts are loaded once as module-level singletons so repeated API calls
don't reload the model from disk.
"""

import json
import os

import joblib
from scipy.sparse import hstack

from nlp_pipeline import get_sentiment, preprocess_text

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

_word_vectorizer = None
_char_vectorizer = None
_model = None
_meta = None


class ModelNotTrainedError(RuntimeError):
    pass


def _load():
    global _word_vectorizer, _char_vectorizer, _model, _meta
    if _model is not None:
        return

    required = ["word_vectorizer.joblib", "char_vectorizer.joblib", "model.joblib", "meta.json"]
    missing = [f for f in required if not os.path.exists(os.path.join(ARTIFACT_DIR, f))]
    if missing:
        raise ModelNotTrainedError(
            f"Missing artifacts {missing} in {ARTIFACT_DIR}/. Run `python train.py` first."
        )

    _word_vectorizer = joblib.load(os.path.join(ARTIFACT_DIR, "word_vectorizer.joblib"))
    _char_vectorizer = joblib.load(os.path.join(ARTIFACT_DIR, "char_vectorizer.joblib"))
    _model = joblib.load(os.path.join(ARTIFACT_DIR, "model.joblib"))
    with open(os.path.join(ARTIFACT_DIR, "meta.json")) as f:
        _meta = json.load(f)


def is_ready() -> bool:
    try:
        _load()
        return True
    except ModelNotTrainedError:
        return False


def predict_emotion_and_sentiment(text_input: str, verbose: bool = False) -> dict:
    """
    Predict primary (and optional secondary) emotion and overall sentiment
    for a single piece of raw text.

    Returns a dict with keys: emotions (list[str]), sentiment (str),
    confidence (float), scores (dict[str, float] of every emotion — for
    charts in the frontend), and optionally note (low-confidence explainer).
    """
    _load()
    emotion_cols = _meta["emotion_cols"]
    confidence_threshold = _meta["confidence_threshold"]
    secondary_gap_ratio = _meta["secondary_gap_ratio"]

    cleaned = preprocess_text(text_input, apply_spell_correction=True)

    if not cleaned.strip():
        return {
            "emotions": ["neutral"],
            "sentiment": "Neutral",
            "confidence": 0.0,
            "scores": {e: 0.0 for e in emotion_cols},
            "cleaned_text": cleaned,
        }

    x_word = _word_vectorizer.transform([cleaned])
    x_char = _char_vectorizer.transform([cleaned])
    x_combined = hstack([x_word, x_char])

    probs = _model.predict_proba(x_combined)[0]
    scored = sorted(zip(emotion_cols, probs), key=lambda x: x[1], reverse=True)
    scores = {emo: float(sc) for emo, sc in scored}

    if verbose:
        print(f"[Cleaned input]: {cleaned!r}")
        for emo, sc in scored:
            print(f"  {emo:18s} {sc:.3f}  {'=' * int(sc * 40)}")

    primary_emo, primary_score = scored[0]
    primary_score = float(primary_score)

    if primary_score < confidence_threshold:
        return {
            "emotions": ["neutral"],
            "sentiment": "Neutral",
            "confidence": primary_score,
            "scores": scores,
            "cleaned_text": cleaned,
            "note": (
                f"Low confidence (max={primary_score:.3f} < "
                f"threshold={confidence_threshold:.3f}) — returning Neutral"
            ),
        }

    detected_emotions = [primary_emo]

    if len(scored) > 1:
        second_emo, second_score = scored[1]
        second_score = float(second_score)
        if second_score >= confidence_threshold and second_score >= secondary_gap_ratio * primary_score:
            detected_emotions.append(second_emo)

    sentiment = get_sentiment(primary_emo)

    return {
        "emotions": detected_emotions,
        "sentiment": sentiment,
        "confidence": primary_score,
        "scores": scores,
        "cleaned_text": cleaned,
    }
