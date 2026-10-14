"""
scripts/run_project_fd002.py
----------------------------
Train and evaluate the LSTM autoencoder on FD002 only.
FD002 has 6 operating conditions.

Run from the project root:
    python scripts/run_project_fd002.py
"""

import os, sys, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from src.data_loader import prepare_data, SEQUENCE_LENGTH
from src.model import (
    build_autoencoder, train_autoencoder,
    compute_reconstruction_error, compute_threshold,
)

DATASET_ID = "FD002"
MODEL_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
EPOCHS     = 50
BATCH_SIZE = 64

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"\n{'='*60}")
print(f"  Training LSTM Autoencoder — {DATASET_ID}")
print(f"{'='*60}\n")

data         = prepare_data(DATASET_ID)
train_seqs   = data["train_seqs"]
test_seqs    = data["test_seqs"]
feature_cols = data["feature_cols"]
scaler       = data["scaler"]
n_features   = train_seqs.shape[2]

print(f"  Training sequences : {train_seqs.shape}")
print(f"  Test sequences     : {test_seqs.shape}")

model = build_autoencoder(seq_len=SEQUENCE_LENGTH, n_features=n_features)
model.summary()
train_autoencoder(model, train_seqs, epochs=EPOCHS, batch_size=BATCH_SIZE)

threshold   = compute_threshold(model, train_seqs, percentile=95.0)
test_errors = compute_reconstruction_error(model, test_seqs)

print(f"\nThreshold: {threshold:.6f}")
print(f"Test — mean: {test_errors.mean():.6f}  flagged: {(test_errors > threshold).sum()}/{len(test_errors)}")

prefix = os.path.join(MODEL_DIR, DATASET_ID.lower())
model.save(f"{prefix}_autoencoder.keras")
with open(f"{prefix}_scaler.pkl", "wb")   as f: pickle.dump(scaler, f)
np.save(f"{prefix}_threshold.npy", threshold)
with open(f"{prefix}_features.pkl", "wb") as f: pickle.dump(feature_cols, f)

print(f"\nSaved artefacts to models/ with prefix '{DATASET_ID.lower()}_'")
