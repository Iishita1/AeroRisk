"""
src/data_loader.py
------------------
Handles loading, preprocessing, and sequence creation for the NASA C-MAPSS dataset.

Column layout (space-delimited, no header):
  [0]  unit_id
  [1]  cycle
  [2..4] op1, op2, op3   (3 operational settings)
  [5..25] s1..s21        (21 sensor readings)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ── Column definitions ────────────────────────────────────────────────────────

COLUMN_NAMES = (
    ["unit_id", "cycle"]
    + [f"op{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

# Sensors with near-zero variance across all conditions — dropping them is
# standard practice for C-MAPSS.
DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]

SENSOR_COLS = [c for c in COLUMN_NAMES if c.startswith("s") and c not in DROP_SENSORS]
OP_COLS     = ["op1", "op2", "op3"]

# Only the first HEALTH_CUTOFF_FRACTION of each engine's life is used for
# autoencoder training — these cycles represent "healthy" behaviour.
HEALTH_CUTOFF_FRACTION = 0.60   # first 60 %
SEQUENCE_LENGTH        = 50     # sliding-window length

# Default data directory (relative to project root, i.e. one level up from src/)
_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.dirname(_HERE)
DATA_DIR  = os.path.join(_ROOT, "data")


# ── Low-level I/O ─────────────────────────────────────────────────────────────

def load_raw(filepath: str) -> pd.DataFrame:
    """Read a C-MAPSS text file into a DataFrame with named columns."""
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)
    df = df.dropna(axis=1, how="all")
    df.columns = COLUMN_NAMES[: len(df.columns)]
    return df


def load_dataset(dataset_id: str, data_dir: str = DATA_DIR) -> dict:
    """
    Load train + test + RUL files for one FD dataset.

    Parameters
    ----------
    dataset_id : str   One of 'FD001', 'FD002', 'FD003', 'FD004'.
    data_dir   : str   Directory containing the C-MAPSS txt files.

    Returns
    -------
    dict  {train: DataFrame, test: DataFrame, rul: ndarray}
    """
    base  = os.path.join(data_dir, "{split}_{id}.txt")
    train = load_raw(base.format(split="train", id=dataset_id))
    test  = load_raw(base.format(split="test",  id=dataset_id))
    rul   = np.loadtxt(os.path.join(data_dir, f"RUL_{dataset_id}.txt"), dtype=float)
    return {"train": train, "test": test, "rul": rul}


# ── Feature engineering ───────────────────────────────────────────────────────

def select_features(df: pd.DataFrame, include_ops: bool = False) -> pd.DataFrame:
    """Keep only informative sensor columns (and optionally op-settings)."""
    cols = SENSOR_COLS + (OP_COLS if include_ops else [])
    return df[["unit_id", "cycle"] + cols].copy()


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Append a 'RUL' column computed from training data (max_cycle - cycle)."""
    df = df.copy()
    max_cycles = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycles, on="unit_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df


# ── Scaling ───────────────────────────────────────────────────────────────────

def fit_scaler(df: pd.DataFrame, feature_cols: list) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df[feature_cols].values)
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: MinMaxScaler, feature_cols: list) -> pd.DataFrame:
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols].values)
    return df


# ── Sequence creation ─────────────────────────────────────────────────────────

def make_sequences(df: pd.DataFrame, feature_cols: list, seq_len: int = SEQUENCE_LENGTH):
    """
    Build sliding-window sequences from per-engine time series.

    Returns
    -------
    sequences : ndarray  (N, seq_len, n_features)
    meta      : list of (unit_id, last_cycle)
    """
    sequences, meta = [], []
    for uid, grp in df.groupby("unit_id", sort=False):
        data = grp[feature_cols].values
        n = len(data)
        for i in range(n - seq_len + 1):
            sequences.append(data[i : i + seq_len])
            meta.append((uid, grp["cycle"].iloc[i + seq_len - 1]))
    return np.array(sequences, dtype=np.float32), meta


def make_last_sequence(df: pd.DataFrame, feature_cols: list, seq_len: int = SEQUENCE_LENGTH):
    """
    Return only the LAST window for every engine.
    Used when scoring test engines (one anomaly score per engine).

    Returns
    -------
    sequences : ndarray  (N_engines, seq_len, n_features)
    unit_ids  : list
    """
    sequences, unit_ids = [], []
    for uid, grp in df.groupby("unit_id", sort=False):
        data = grp[feature_cols].values
        if len(data) < seq_len:
            pad  = np.tile(data[0], (seq_len - len(data), 1))
            data = np.vstack([pad, data])
        sequences.append(data[-seq_len:])
        unit_ids.append(uid)
    return np.array(sequences, dtype=np.float32), unit_ids


# ── "Healthy" training subset ─────────────────────────────────────────────────

def get_healthy_sequences(df: pd.DataFrame, feature_cols: list, seq_len: int = SEQUENCE_LENGTH):
    """Extract sequences from the early (healthy) portion of each engine's life."""
    healthy_frames = []
    for uid, grp in df.groupby("unit_id", sort=False):
        max_cycle = grp["cycle"].max()
        cutoff    = max_cycle * HEALTH_CUTOFF_FRACTION
        healthy_frames.append(grp[grp["cycle"] <= cutoff])
    healthy_df = pd.concat(healthy_frames, ignore_index=True)
    return make_sequences(healthy_df, feature_cols, seq_len)


# ── Full pipeline convenience function ───────────────────────────────────────

def prepare_data(dataset_id: str, data_dir: str = DATA_DIR, include_ops: bool = False):
    """
    End-to-end preparation for a single FD dataset.

    Returns
    -------
    dict  {train_seqs, test_seqs, test_unit_ids, scaler, feature_cols, raw,
           train_df, test_df}
    """
    raw   = load_dataset(dataset_id, data_dir)
    train = select_features(raw["train"], include_ops=include_ops)
    test  = select_features(raw["test"],  include_ops=include_ops)

    feature_cols = SENSOR_COLS + (OP_COLS if include_ops else [])

    scaler   = fit_scaler(train, feature_cols)
    train_sc = apply_scaler(train, scaler, feature_cols)
    test_sc  = apply_scaler(test,  scaler, feature_cols)

    train_seqs, _         = get_healthy_sequences(train_sc, feature_cols)
    test_seqs, test_units = make_last_sequence(test_sc, feature_cols)

    return {
        "train_seqs"    : train_seqs,
        "test_seqs"     : test_seqs,
        "test_unit_ids" : test_units,
        "scaler"        : scaler,
        "feature_cols"  : feature_cols,
        "raw"           : raw,
        "train_df"      : train_sc,
        "test_df"       : test_sc,
    }


def prepare_combined(data_dir: str = DATA_DIR, include_ops: bool = False):
    """
    Merge all four FD datasets into one training corpus.
    Unit IDs are offset (FD002 +1000, FD003 +2000, FD004 +3000) to stay unique.

    Returns same structure as prepare_data plus 'test_data' per dataset.
    """
    offsets      = {"FD001": 0, "FD002": 1000, "FD003": 2000, "FD004": 3000}
    feature_cols = SENSOR_COLS + (OP_COLS if include_ops else [])

    all_train_frames = []
    per_dataset      = {}

    for ds_id, offset in offsets.items():
        raw   = load_dataset(ds_id, data_dir)
        train = select_features(raw["train"], include_ops=include_ops)
        test  = select_features(raw["test"],  include_ops=include_ops)

        train = train.copy(); train["unit_id"] += offset
        test  = test.copy();  test["unit_id"]  += offset

        per_dataset[ds_id] = {"raw": raw, "train": train, "test": test, "offset": offset}
        all_train_frames.append(train)

    combined_train    = pd.concat(all_train_frames, ignore_index=True)
    scaler            = fit_scaler(combined_train, feature_cols)
    combined_train_sc = apply_scaler(combined_train, scaler, feature_cols)
    train_seqs, _     = get_healthy_sequences(combined_train_sc, feature_cols)

    test_data = {}
    for ds_id, d in per_dataset.items():
        test_sc = apply_scaler(d["test"], scaler, feature_cols)
        seqs, uids = make_last_sequence(test_sc, feature_cols)
        test_data[ds_id] = {
            "test_seqs" : seqs,
            "unit_ids"  : uids,
            "test_df"   : test_sc,
            "rul"       : d["raw"]["rul"],
        }

    return {
        "train_seqs"  : train_seqs,
        "scaler"      : scaler,
        "feature_cols": feature_cols,
        "test_data"   : test_data,
        "train_df"    : combined_train_sc,
    }
