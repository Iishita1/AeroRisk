"""Evaluate the saved combined model on held-out C-MAPSS test engines.

Run from the project root:
    python scripts/evaluate_test.py

The C-MAPSS test split has true RUL labels rather than binary anomaly labels.
For anomaly metrics this script uses the documented proxy: an engine is
positive when its true RUL is <= 30 cycles.
"""

import os
import json
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

from src.data_loader import (
    DATA_DIR,
    apply_scaler,
    load_dataset,
    make_last_sequence,
    select_features,
)
from src.evaluation import evaluate_test_sets, TEST_RUL_CUTOFF

MODEL_DIR = os.path.join(ROOT, "models")


def main():
    keras_path = os.path.join(MODEL_DIR, "combined_autoencoder.keras")
    h5_path = os.path.join(MODEL_DIR, "combined_autoencoder.h5")
    model_path = keras_path if os.path.exists(keras_path) else h5_path
    if not os.path.exists(model_path):
        raise FileNotFoundError("Train the combined model first: python scripts/run_combined.py")

    model = tf.keras.models.load_model(model_path)
    with open(os.path.join(MODEL_DIR, "combined_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "combined_features.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    thresholds_path = os.path.join(MODEL_DIR, "combined_thresholds_per_ds.pkl")
    if os.path.exists(thresholds_path):
        with open(thresholds_path, "rb") as f:
            thresholds = pickle.load(f)
    else:
        global_threshold = float(np.load(os.path.join(MODEL_DIR, "combined_threshold.npy")))
        thresholds = {f"FD00{i}": global_threshold for i in range(1, 5)}

    test_data = {}
    for ds_id in ("FD001", "FD002", "FD003", "FD004"):
        raw = load_dataset(ds_id, DATA_DIR)
        test_df = select_features(raw["test"])
        test_scaled = apply_scaler(test_df, scaler, feature_cols)
        sequences, _ = make_last_sequence(test_scaled, feature_cols)
        test_data[ds_id] = {"test_seqs": sequences, "rul": raw["rul"]}

    metrics = evaluate_test_sets(model, test_data, thresholds)
    print(f"Held-out test evaluation; positive label: true RUL <= {TEST_RUL_CUTOFF} cycles")
    print("Dataset  Precision  Recall  FPR     TP  FP  TN  FN")
    for ds_id, row in metrics.items():
        print(
            f"{ds_id:<8} {row['precision']:<10.3f} {row['recall']:<7.3f} "
            f"{row['false_positive_rate']:<7.3f} {row['tp']:<3} {row['fp']:<3} "
            f"{row['tn']:<3} {row['fn']:<3}"
        )

    # Persist the held-out results so the Streamlit About tab can display them
    # without requiring a costly model retrain.
    metadata_path = os.path.join(MODEL_DIR, "combined_metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
    metadata["test_evaluation"] = metrics
    metadata["test_anomaly_label"] = f"true RUL <= {TEST_RUL_CUTOFF} cycles"
    metadata["test_rul_cutoff"] = TEST_RUL_CUTOFF
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved held-out metrics to {metadata_path}")


if __name__ == "__main__":
    main()
