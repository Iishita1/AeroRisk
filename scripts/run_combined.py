"""
scripts/run_combined.py
-----------------------
Train ONE master LSTM autoencoder on all four FD datasets combined.
This is the model the Streamlit app uses.

Run from the project root:
    python scripts/run_combined.py

Artefacts saved to models/
    combined_autoencoder.keras       ← native Keras format
    combined_scaler.pkl
    combined_threshold.npy           ← global threshold (95th pct, all data)
    combined_thresholds_per_ds.pkl   ← per-dataset thresholds (used by app)
    combined_features.pkl
    combined_rul_regressor.pkl       ← sklearn Pipeline: anomaly_score → RUL
    combined_metadata.json           ← training info (date, hyperparams, scores)
"""

import os, sys, pickle, json, datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.random.set_seed(42)
np.random.seed(42)

from src.data_loader import (
    prepare_combined, get_healthy_sequences,
    make_sequences_with_rul, apply_scaler, select_features,
    load_dataset, SEQUENCE_LENGTH,
)
from src.model import (
    build_autoencoder, train_autoencoder,
    compute_reconstruction_error, compute_threshold,
    fit_rul_regressor,
)

MODEL_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
EPOCHS     = 60
BATCH_SIZE = 128

os.makedirs(MODEL_DIR, exist_ok=True)
prefix = os.path.join(MODEL_DIR, "combined")

print(f"\n{'='*60}")
print("  Training MASTER Autoencoder — Combined FD001-FD004")
print(f"{'='*60}\n")

# ── 1. Load and prepare data ──────────────────────────────────────────────────
print("Loading and merging all datasets...")
combined     = prepare_combined()
train_seqs   = combined["train_seqs"]
feature_cols = combined["feature_cols"]
scaler       = combined["scaler"]
test_data    = combined["test_data"]
train_df     = combined["train_df"]
n_features   = train_seqs.shape[2]

# Pre-shuffle so the 10% val split is a random cross-section of all datasets,
# preventing val_loss spikes caused by homogeneous operating-condition clusters.
rng  = np.random.default_rng(42)
perm = rng.permutation(len(train_seqs))
train_seqs = train_seqs[perm]

print(f"  Combined training sequences : {train_seqs.shape}")
print(f"  Features ({n_features})           : {', '.join(feature_cols)}\n")

# ── 2. Build and train the autoencoder ────────────────────────────────────────
model = build_autoencoder(seq_len=SEQUENCE_LENGTH, n_features=n_features)
model.summary()

history = train_autoencoder(model, train_seqs, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
actual_epochs = len(history.history["loss"])
best_val_loss = float(min(history.history["val_loss"]))

# ── 3. Anomaly thresholds ─────────────────────────────────────────────────────
global_threshold = compute_threshold(model, train_seqs, percentile=95.0)
print(f"\nGlobal anomaly threshold (95th pct): {global_threshold:.6f}")

print("\nComputing per-dataset thresholds...")
per_ds_thresholds: dict[str, float] = {}
offsets = {"FD001": 0, "FD002": 1000, "FD003": 2000, "FD004": 3000}

for ds_id, ds in test_data.items():
    offset   = offsets[ds_id]
    ds_train = train_df[
        (train_df["unit_id"] > offset) & (train_df["unit_id"] <= offset + 999)
    ].copy()
    if len(ds_train) == 0:
        per_ds_thresholds[ds_id] = global_threshold
        continue
    ds_seqs, _ = get_healthy_sequences(ds_train, feature_cols)
    if len(ds_seqs) == 0:
        per_ds_thresholds[ds_id] = global_threshold
        continue
    t = compute_threshold(model, ds_seqs, percentile=95.0)
    per_ds_thresholds[ds_id] = t
    print(f"  {ds_id}: {t:.6f}")

# ── 4. Evaluate test sets ─────────────────────────────────────────────────────
print("\nPer-dataset test evaluation (per-dataset thresholds):")
test_eval = {}
for ds_id, ds in test_data.items():
    errors  = compute_reconstruction_error(model, ds["test_seqs"])
    t       = per_ds_thresholds[ds_id]
    flagged = int((errors > t).sum())
    test_eval[ds_id] = {
        "threshold": round(t, 6),
        "mean_error": round(float(errors.mean()), 6),
        "max_error":  round(float(errors.max()),  6),
        "flagged":    flagged,
        "total":      len(errors),
    }
    print(f"  {ds_id}: threshold={t:.6f}  mean={errors.mean():.6f}  "
          f"max={errors.max():.6f}  flagged={flagged}/{len(errors)}")

# ── 5. Fit RUL regressor ──────────────────────────────────────────────────────
# Build (sequence, RUL) pairs from the full training data (not just healthy),
# because we need the degradation range to calibrate score → RUL.
# We merge all 4 datasets' raw training DataFrames, scale them, then call
# make_sequences_with_rul which computes RUL from cycle position.
print("\nFitting RUL regressor (anomaly_score → RUL)...")

all_labeled_seqs  = []
all_labeled_ruls  = []

for ds_id, offset in offsets.items():
    raw = load_dataset(ds_id)
    train_raw = select_features(raw["train"])
    train_raw = train_raw.copy()
    train_raw["unit_id"] += offset          # keep IDs unique across datasets
    train_sc = apply_scaler(train_raw, scaler, feature_cols)
    seqs, ruls = make_sequences_with_rul(train_sc, feature_cols)
    all_labeled_seqs.append(seqs)
    all_labeled_ruls.append(ruls)
    print(f"  {ds_id}: {len(seqs)} labelled sequences, "
          f"RUL range [{ruls.min():.0f}, {ruls.max():.0f}]")

labeled_seqs = np.concatenate(all_labeled_seqs, axis=0)
labeled_ruls = np.concatenate(all_labeled_ruls, axis=0)

# Shuffle to avoid ordering effects in the regression fit
perm2 = rng.permutation(len(labeled_seqs))
labeled_seqs = labeled_seqs[perm2]
labeled_ruls = labeled_ruls[perm2]

rul_regressor = fit_rul_regressor(model, labeled_seqs, labeled_ruls, degree=2)

# Quick evaluation: MAE on the labelled set
from sklearn.metrics import mean_absolute_error
pred_errors = compute_reconstruction_error(model, labeled_seqs)
rul_preds   = rul_regressor.predict(pred_errors.reshape(-1, 1))
rul_mae     = float(mean_absolute_error(labeled_ruls, rul_preds))
print(f"  RUL regressor training MAE: {rul_mae:.1f} cycles")

# ── 6. Save all artefacts ─────────────────────────────────────────────────────
model.save(f"{prefix}_autoencoder.keras")

with open(f"{prefix}_scaler.pkl",              "wb") as f: pickle.dump(scaler, f)
with open(f"{prefix}_features.pkl",            "wb") as f: pickle.dump(feature_cols, f)
with open(f"{prefix}_thresholds_per_ds.pkl",   "wb") as f: pickle.dump(per_ds_thresholds, f)
with open(f"{prefix}_rul_regressor.pkl",       "wb") as f: pickle.dump(rul_regressor, f)
np.save(f"{prefix}_threshold.npy", global_threshold)

# Save human-readable metadata JSON
metadata = {
    "trained_at"         : datetime.datetime.now().isoformat(timespec="seconds"),
    "epochs_run"         : actual_epochs,
    "best_val_loss"      : round(best_val_loss, 6),
    "batch_size"         : BATCH_SIZE,
    "sequence_length"    : SEQUENCE_LENGTH,
    "n_features"         : n_features,
    "feature_cols"       : feature_cols,
    "global_threshold"   : round(global_threshold, 6),
    "per_ds_thresholds"  : {k: round(v, 6) for k, v in per_ds_thresholds.items()},
    "rul_regressor_mae"  : round(rul_mae, 1),
    "test_evaluation"    : test_eval,
}
with open(f"{prefix}_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved artefacts:")
for name in [
    "combined_autoencoder.keras",
    "combined_scaler.pkl",
    "combined_threshold.npy",
    "combined_thresholds_per_ds.pkl",
    "combined_rul_regressor.pkl",
    "combined_features.pkl",
    "combined_metadata.json",
]:
    print(f"  models/{name}")

print(f"\nGlobal threshold : {global_threshold:.6f}")
print(f"RUL regressor MAE: {rul_mae:.1f} cycles")
print("\nCombined model training complete.")
