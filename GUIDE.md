# Getting Started Guide

## Project Structure

```
aviation/
├── .env                     ← Gemini API key (never commit this)
├── requirements.txt
├── GUIDE.md                 ← you are here
│
├── app/
│   └── app.py               ← Streamlit web app (run this last)
│
├── src/                     ← core library modules
│   ├── data_loader.py       ← dataset loading, scaling, sequence creation
│   ├── model.py             ← LSTM autoencoder architecture + inference
│   ├── expert_diagnosis.py  ← rule-based fault diagnosis + NLP log search
│   ├── gemini_api.py        ← Gemini API wrapper
│   └── report_generator.py  ← PDF report builder
│
├── scripts/                 ← training scripts (run once to produce models/)
│   ├── run_combined.py      ← trains master model on all 4 datasets ← USE THIS
│   ├── run_project_fd001.py ← trains model on FD001 only
│   ├── run_project_fd002.py ← trains model on FD002 only
│   ├── run_project_fd003.py ← trains model on FD003 only
│   └── run_project_fd004.py ← trains model on FD004 only
│
├── data/                    ← NASA C-MAPSS dataset (already present)
├── models/                  ← saved model artefacts (created after training)
└── reports/                 ← generated PDF reports (created on demand)
```

---

## Step 1 — Set up the Python environment

> **Important:** TensorFlow requires **Python 3.12** (not 3.13 or 3.14).
> Your machine has Python 3.12 at:
> `C:\Users\nopei\AppData\Local\Programs\Python\Python312\python.exe`

Everything is run from the **project root** (`aviation/`).

```powershell
# Delete the old venv if you created one with Python 3.14
Remove-Item -Recurse -Force venv

# Create a new venv explicitly using Python 3.12
C:\Users\nopei\AppData\Local\Programs\Python\Python312\python.exe -m venv venv

# Activate it
.\venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt
```

> If you get an execution policy error on the activate step, run:
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

## Step 2 — Download the spaCy language model

The NLP maintenance-log search uses spaCy's small English model.

```powershell
python -m spacy download en_core_web_sm
```

---

## Step 3 — Train the model

The Streamlit app uses the **combined master model** trained on all four datasets.
This is a one-time step — the artefacts are saved to `models/` and reused on every app start.

```powershell
python scripts/run_combined.py
```

The combined training run evaluates the anomaly detector on the held-out
C-MAPSS test engines. Since C-MAPSS supplies test RUL values rather than binary
fault labels, the reported binary metrics use an explicit proxy: **positive =
true RUL ≤ 30 cycles**. Precision, recall, false-positive rate, and TP/FP/TN/FN
counts are calculated per dataset from the engine's final 50-cycle window.
These measure detection of near-end-of-life engines under that chosen cutoff;
they are not metrics against direct fault annotations. To recompute metrics
with the saved model and thresholds without retraining, run:

```powershell
python scripts/evaluate_test.py
```

Expected output (condensed):
```
Loading and merging all datasets...
  Combined training sequences : (XXXX, 50, 14)
  ...
Epoch 1/60 ...
...
Anomaly threshold (95th pct on healthy data): 0.XXXXXX

Per-dataset test evaluation:
  FD001: mean=0.0X  flagged=XX/100
  FD002: mean=0.0X  flagged=XX/259
  ...

Saved artefacts to models/ with prefix 'combined_'
```

Training takes roughly **5–20 minutes** depending on your hardware (CPU vs GPU).

> **Want faster results for testing?** Train on a single dataset first:
> ```powershell
> python scripts/run_project_fd001.py
> ```
> The app won't load FD001-only models, but you can verify training works.

---

## Step 4 — Launch the web app

```powershell
streamlit run app/app.py
```

Your browser will open automatically at `http://localhost:8501`.

---

## Using the app

### Tab 1 — Pre-loaded Test Data
1. Pick a dataset (FD001–FD004) from the dropdown.
2. Enter an engine ID (any number from 1 up to the max shown).
3. Optionally tick **Include maintenance log** and paste free-text notes.
4. Click **Run Diagnosis**.
5. View the anomaly gauge, sensor error chart, timeline, expert diagnosis, and Gemini narrative.
6. Click **Download Diagnostic Report (PDF)** to save the full report.

### Tab 2 — Live Sensor Input
1. Copy-paste rows from any `test_FDxxx.txt` file (or real sensor data) — at least 50 rows.
2. Optionally paste maintenance log text.
3. Click **Analyse**.

Column format accepted: `unit_id  cycle  op1  op2  op3  s1…s21` (26 values per row).  
24-column (no id/cycle) and 25-column (no id) formats are also accepted.

### Tab 3 — About
Architecture overview and sensor-to-component mapping table.

---

## Environment variables

The `.env` file at the project root holds your Gemini API key:

```
GEMINI_API_KEY=your_key_here
```

If the key is missing or invalid, the app falls back to a template-based narrative
and shows a warning — everything else (anomaly detection, expert rules, PDF) still works.

---

## Re-training

To retrain from scratch (e.g. after changing hyperparameters), just re-run:

```powershell
python scripts/run_combined.py
```

New artefacts overwrite the old ones in `models/`.

---

## Common issues

| Problem | Fix |
|---|---|
| `Could not find a version that satisfies tensorflow` | Your venv is using Python 3.13 or 3.14 — TensorFlow only supports up to 3.12. Delete `venv/` and recreate it with the Python 3.12 path shown in Step 1. |
| `ModuleNotFoundError` | Make sure your venv is activated and `pip install -r requirements.txt` completed |
| `Model not found` in the app | Run `python scripts/run_combined.py` first |
| Gemini returns an error | Check your API key in `.env`; the app will fall back to stub text |
| spaCy model not found | Run `python -m spacy download en_core_web_sm` |
| TensorFlow GPU not detected | Install the GPU build of TF separately; CPU training still works |
