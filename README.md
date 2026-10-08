# Emotion \& Sentiment Analyzer — full stack

Turns `sentiment\_emotion\_analysis.ipynb` into a real app: a Flask API serving
the GoEmotions model, and a static frontend that talks to it.

```
project/
├── backend/
│   ├── nlp\_pipeline.py   # preprocessing, shared by train + serve
│   ├── train.py          # reproduces the notebook, saves model artifacts
│   ├── predict.py        # loads artifacts, runs inference
│   ├── app.py             # Flask API (/api/predict, /api/health)
│   ├── requirements.txt
│   └── artifacts/         # created by train.py (empty until you train)
└── frontend/
    └── index.html          # single-file UI, no build step
```

## 1\. Set up the backend

```bash
cd backend
python -m venv venv 
.\\myenv\\Scripts\\Activate.ps1

venv\\Scripts\\activate   # optional but recommended
pip install -r requirements.txt
```

`train.py` downloads GoEmotions via `kagglehub`, exactly like the notebook —
you'll need Kaggle credentials configured the same way you had them working
in Colab/Jupyter (`\~/.kaggle/kaggle.json` or the `kagglehub` login flow).

```bash
python train.py
```

This trains the TF-IDF + OneVsRestClassifier(LogisticRegression) pipeline,
prints the same evaluation report as the notebook, and saves four files into
`backend/artifacts/`: `word\_vectorizer.joblib`, `char\_vectorizer.joblib`,
`model.joblib`, `meta.json`. Takes a few minutes on a laptop CPU. Re-run it
any time you change the hyperparameters in `train.py`.

Then start the API:

```bash
python app.py
```

It serves on `http://localhost:5000`. Quick check:

```bash
curl -X POST http://localhost:5000/api/predict \\
  -H "Content-Type: application/json" \\
  -d '{"text": "the trip was amaizing"}'
```

## 2\. Open the frontend

No build step — just open `frontend/index.html` in a browser (double-click
it, or `python -m http.server` from the `frontend/` folder and visit it).
It talks to `http://localhost:5000` by default; change that under
"connection settings" at the bottom of the page if you deploy the backend
elsewhere. A status dot there tells you if it can reach the API and whether
the model has been trained yet.

## API contract

`POST /api/predict`

```json
{"text": "I am sooo happy!!!"}
```

→

```json
{
  "emotions": \["joy"],
  "sentiment": "Positive",
  "confidence": 0.82,
  "scores": {"joy": 0.82, "gratitude": 0.41, "...": "..."},
  "cleaned\_text": "i am so happy \_\_EXCL\_\_",
  "text": "I am sooo happy!!!"
}
```

Low-confidence inputs come back as `{"emotions": \["neutral"], "sentiment": "Neutral", "note": "..."}`,
matching the notebook's confidence-gating logic exactly.

`GET /api/health` → `{"status": "ok", "model\_ready": true|false}`

`POST /api/predict/batch` → `{"texts": \[...]}` → `{"results": \[...]}`, capped
at 50 texts per call.

## Deploying for real (optional next step)

* Backend: `gunicorn app:app` behind a reverse proxy (nginx/Caddy), or any
container host. Lock down `CORS(app)` in `app.py` to your real frontend
origin instead of allowing all origins.
* Frontend: it's a static file — any static host (GitHub Pages, Netlify,
S3) works. Just point "connection settings" at your deployed backend URL.

## If you want to redesign the frontend with another AI tool

The backend's API contract above is the only thing that has to stay fixed.
If you want to hand the frontend off to a different tool (v0, Lovable,
Cursor, or another Claude conversation) to restyle it, a prompt like this
will get you a compatible result:

> Build a single-page web app (plain HTML/CSS/JS, no build step) called
> "Undertone" that analyzes the emotion and sentiment of text the user
> types in. It calls a backend at `http://localhost:5000` (configurable):
> `POST /api/predict` with `{"text": "..."}`, returning `{"emotions":
> \[...], "sentiment": "Positive"|"Negative"|"Neutral", "confidence": 0-1,
> "scores": {emotion: score, ...}, "cleaned\_text": "..."}`, and `GET
> /api/health` returning `{"status": "ok", "model\_ready": bool}`. Show:
> a text input, the detected emotion(s) and sentiment prominently, a bar
> per emotion showing its score out of the full 28-label GoEmotions set,
> a few clickable example sentences, a small history of recent queries,
> and a visible connection status if the backend is unreachable or the
> model isn't trained yet. Handle empty/failed responses gracefully.
> Design it with a distinct point of view — avoid generic SaaS-card
> styling — the subject is close reading of emotional language, so lean
> into that in the visual identity rather than a default dashboard look.

