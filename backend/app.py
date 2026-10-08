"""
app.py
------
Flask API for the emotion & sentiment classifier.

Endpoints:
    GET  /api/health            -> {"status": "ok", "model_ready": bool}
    POST /api/predict           -> {"text": "..."}  -> prediction dict
    POST /api/predict/batch     -> {"texts": ["...", "..."]} -> list of predictions

Run:
    python app.py
    (serves on http://localhost:5000)

CORS is enabled for all origins by default so the static frontend (opened
as a local file or served separately) can call this API directly. Lock
`CORS(app, ...)` down to your real frontend origin before deploying.
"""

import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from predict import ModelNotTrainedError, predict_emotion_and_sentiment, is_ready

app = Flask(__name__)
CORS(app)

MAX_TEXT_LENGTH = 2000
MAX_BATCH_SIZE = 50


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "model_ready": is_ready()})


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Field 'text' is required and must be a non-empty string."}), 400
    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({"error": f"Text exceeds {MAX_TEXT_LENGTH} characters."}), 400

    try:
        result = predict_emotion_and_sentiment(text)
    except ModelNotTrainedError as e:
        return jsonify({"error": str(e)}), 503

    result["text"] = text
    return jsonify(result)


@app.post("/api/predict/batch")
def predict_batch():
    payload = request.get_json(silent=True) or {}
    texts = payload.get("texts", [])

    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "Field 'texts' is required and must be a non-empty list."}), 400
    if len(texts) > MAX_BATCH_SIZE:
        return jsonify({"error": f"Batch size exceeds {MAX_BATCH_SIZE}."}), 400

    try:
        results = []
        for text in texts:
            if not isinstance(text, str) or not text.strip():
                results.append({"error": "empty or invalid text"})
                continue
            r = predict_emotion_and_sentiment(text[:MAX_TEXT_LENGTH])
            r["text"] = text
            results.append(r)
    except ModelNotTrainedError as e:
        return jsonify({"error": str(e)}), 503

    return jsonify({"results": results})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
