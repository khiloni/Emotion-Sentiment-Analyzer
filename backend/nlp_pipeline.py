"""
nlp_pipeline.py
----------------
Emotion-preserving text preprocessing, shared verbatim between training
(train.py) and serving (predict.py) so that inference-time cleaning always
matches what the model was trained on.

Ported directly from the project notebook (sentiment_emotion_analysis.ipynb).
"""

import os
import re

import nltk
from nltk.stem import WordNetLemmatizer
from symspellpy import SymSpell, Verbosity
import symspellpy

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()

# ---------------------------------------------------------------------------
# Contractions
# ---------------------------------------------------------------------------
CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "didn't": "did not", "isn't": "is not", "aren't": "are not",
    "wasn't": "was not", "weren't": "were not", "hasn't": "has not",
    "haven't": "have not", "hadn't": "had not", "shouldn't": "should not",
    "wouldn't": "would not", "couldn't": "could not", "mustn't": "must not",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "you're": "you are", "you've": "you have", "you'll": "you will",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "we're": "we are", "we've": "we have", "we'll": "we will",
    "they're": "they are", "they've": "they have", "they'll": "they will",
    "that's": "that is", "there's": "there is", "here's": "here is",
    "let's": "let us", "who's": "who is", "what's": "what is",
    "ain't": "is not",
}

# Abbreviations / slang we do NOT want spell-corrected
_SKIP_TOKENS = {
    "omg", "lol", "lmao", "wtf", "tbh", "imo", "imho", "smh",
    "brb", "irl", "fml", "ngl", "ikr", "gg", "af", "rn", "ily",
    "yay", "omfg", "rotfl",
}


def _expand_contractions(text: str) -> str:
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
    return text


def _normalize_repeated_chars(text: str) -> str:
    """sooooo -> soo  (max 2 consecutive identical chars)."""
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


# ---------------------------------------------------------------------------
# SymSpell (lazy singleton — loading the dictionary is not free)
# ---------------------------------------------------------------------------
_sym_spell = None


def _get_sym_spell() -> SymSpell:
    global _sym_spell
    if _sym_spell is None:
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        try:
            import pkg_resources
            dict_path = pkg_resources.resource_filename(
                "symspellpy", "frequency_dictionary_en_82_765.txt"
            )
        except Exception:
            dict_path = os.path.join(
                os.path.dirname(symspellpy.__file__),
                "frequency_dictionary_en_82_765.txt",
            )
        sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
        _sym_spell = sym_spell
    return _sym_spell


def correct_spelling(text: str) -> str:
    """Correct spelling word-by-word using SymSpell."""
    sym_spell = _get_sym_spell()
    words = text.split()
    corrected = []
    for word in words:
        if word in _SKIP_TOKENS or len(word) <= 2 or not word.isalpha():
            corrected.append(word)
            continue
        suggestions = sym_spell.lookup(
            word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True
        )
        corrected.append(suggestions[0].term if suggestions else word)
    return " ".join(corrected)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def preprocess_text(text: str, apply_spell_correction: bool = True) -> str:
    """
    Emotion-preserving text preprocessing.

    - lowercases, strips URLs/@mentions, unpacks hashtags
    - expands contractions
    - keeps `!`/`?` as explicit signal tokens (__EXCL__ / __QUES__)
    - collapses letter-repeats ("sooooo" -> "soo")
    - optionally spell-corrects (used at inference time, not training time)
    - lemmatizes
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)  # hashtag -> word
    text = _expand_contractions(text)
    text = re.sub(r"!+", " __EXCL__ ", text)  # keep ! signal
    text = re.sub(r"\?+", " __QUES__ ", text)  # keep ? signal
    text = _normalize_repeated_chars(text)
    text = re.sub(r"[^a-z0-9\s'_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if apply_spell_correction and text:
        words = text.split()
        alpha_words = [w for w in words if not (w.startswith("__") and w.endswith("__"))]
        special_idx = [i for i, w in enumerate(words) if (w.startswith("__") and w.endswith("__"))]
        corrected_alpha = correct_spelling(" ".join(alpha_words)).split()
        result_words = []
        ai = 0
        for i, w in enumerate(words):
            if i in special_idx:
                result_words.append(w)
            else:
                if ai < len(corrected_alpha):
                    result_words.append(corrected_alpha[ai])
                    ai += 1
                else:
                    result_words.append(w)
        text = " ".join(result_words)

    words = text.split()
    text = " ".join(
        w if (w.startswith("__") and w.endswith("__")) else lemmatizer.lemmatize(w)
        for w in words
    )
    return text


# ---------------------------------------------------------------------------
# Emotion -> Sentiment mapping (GoEmotions paper coarse groupings)
# ---------------------------------------------------------------------------
SENTIMENT_MAP = {
    "admiration": "Positive", "amusement": "Positive", "approval": "Positive",
    "caring": "Positive", "excitement": "Positive", "gratitude": "Positive",
    "joy": "Positive", "love": "Positive", "optimism": "Positive",
    "pride": "Positive", "relief": "Positive",
    "anger": "Negative", "annoyance": "Negative", "disappointment": "Negative",
    "disapproval": "Negative", "disgust": "Negative", "embarrassment": "Negative",
    "fear": "Negative", "grief": "Negative", "nervousness": "Negative",
    "remorse": "Negative", "sadness": "Negative",
    "confusion": "Neutral", "curiosity": "Neutral", "desire": "Neutral",
    "realization": "Neutral", "surprise": "Neutral", "neutral": "Neutral",
}


def get_sentiment(emotion_name: str) -> str:
    """Map an emotion name to Positive / Negative / Neutral."""
    return SENTIMENT_MAP.get(emotion_name.lower(), "Neutral")
