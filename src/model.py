"""
src/model.py
------------
LSTM Autoencoder for time-series anomaly detection on C-MAPSS sensor data.

Architecture
------------
Encoder  : LSTM(128) → LSTM(64)  → bottleneck vector (64-dim)
Bridge   : RepeatVector(seq_len)  — spreads bottleneck across time steps
Decoder  : LSTM(64, ret_seq)  → LSTM(128, ret_seq) → TimeDistributed(Dense)

Train on *healthy* sequences only.  At inference, high MAE = anomaly.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    LSTM, Dense, RepeatVector, TimeDistributed, Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

# Seeds are set here as a default; the training scripts also set them
# before importing this module for full reproducibility.
tf.random.set_seed(42)
np.random.seed(42)


# ── Model factory ─────────────────────────────────────────────────────────────

def build_autoencoder(seq_len: int, n_features: int, latent_dim: int = 64) -> Model:
    """
    Build and compile the LSTM autoencoder.

    Parameters
    ----------
    seq_len     : number of time steps per sequence (e.g. 50)
    n_features  : number of sensor channels per step
    latent_dim  : bottleneck size (inner LSTM hidden units)
    """
    inputs = Input(shape=(seq_len, n_features), name="encoder_input")

    # Encoder
    x = LSTM(128, return_sequences=True,  name="enc_lstm1")(inputs)
    x = Dropout(0.2,                      name="enc_drop1")(x)
    x = LSTM(latent_dim, return_sequences=False, name="enc_lstm2")(x)

    # Bridge
    x = RepeatVector(seq_len, name="repeat_vector")(x)

    # Decoder
    x = LSTM(latent_dim, return_sequences=True, name="dec_lstm1")(x)
    x = Dropout(0.2,                            name="dec_drop1")(x)
    x = LSTM(128,        return_sequences=True, name="dec_lstm2")(x)
    outputs = TimeDistributed(
        Dense(n_features, activation="sigmoid"), name="reconstruction"
    )(x)

    model = Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mae")
    return model


# ── Training helper ───────────────────────────────────────────────────────────

def train_autoencoder(
    model: Model,
    train_sequences: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    validation_split: float = 0.10,
    verbose: int = 1,
):
    """
    Fit the autoencoder (target == input — that's the autoencoder objective).

    Returns keras History object.
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=7,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-6, verbose=1,
        ),
    ]
    return model.fit(
        train_sequences, train_sequences,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        shuffle=True,
        verbose=verbose,
    )


# ── Inference helpers ─────────────────────────────────────────────────────────

def compute_reconstruction_error(
    model: Model,
    sequences: np.ndarray,
    per_sensor: bool = False,
) -> np.ndarray:
    """
    Mean Absolute Error between input and reconstruction.

    per_sensor=False → shape (N,)             scalar MAE per sample
    per_sensor=True  → shape (N, n_features)  MAE per sensor per sample
    """
    reconstructions = model.predict(sequences, verbose=0)
    errors = np.abs(sequences - reconstructions).mean(axis=1)   # (N, n_features)
    if per_sensor:
        return errors
    return errors.mean(axis=1)                                   # (N,)


def compute_threshold(
    model: Model,
    healthy_sequences: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """
    Derive an anomaly threshold from reconstruction errors on healthy sequences.
    Set at `percentile` of the error distribution — signals above it are anomalous.
    """
    errors = compute_reconstruction_error(model, healthy_sequences)
    return float(np.percentile(errors, percentile))


def anomaly_score_for_engine(
    model: Model,
    sequence: np.ndarray,
    feature_cols: list,
    threshold: float,
) -> dict:
    """
    Full per-engine diagnostic bundle.

    Parameters
    ----------
    sequence : ndarray  shape (seq_len, n_features)

    Returns
    -------
    dict  {anomaly_score, is_anomaly, per_sensor_errors, top3_sensors, threshold}
    """
    seq_batch  = sequence[np.newaxis, ...]                   # (1, seq_len, n_features)
    per_sensor = compute_reconstruction_error(model, seq_batch, per_sensor=True)[0]
    overall    = float(per_sensor.mean())

    sensor_errors = {col: float(per_sensor[i]) for i, col in enumerate(feature_cols)}
    top3 = sorted(sensor_errors, key=sensor_errors.get, reverse=True)[:3]

    return {
        "anomaly_score"     : overall,
        "is_anomaly"        : overall > threshold,
        "per_sensor_errors" : sensor_errors,
        "top3_sensors"      : top3,
        "threshold"         : threshold,
    }


# ── RUL estimation ────────────────────────────────────────────────────────────

def fit_rul_regressor(
    model: Model,
    sequences: np.ndarray,
    rul_labels: np.ndarray,
    degree: int = 2,
) -> Pipeline:
    """
    Fit a polynomial Ridge regression that maps anomaly_score → RUL.

    The autoencoder already captures health degradation in its reconstruction
    error; we just need to calibrate the monotonic relationship between that
    score and remaining useful life using the labelled training sequences.

    Parameters
    ----------
    model       : trained LSTM autoencoder
    sequences   : ndarray (N, seq_len, n_features)  — one window per sample
    rul_labels  : ndarray (N,)  — true RUL for each sample
    degree      : polynomial degree (2 = quadratic, works well in practice)

    Returns
    -------
    sklearn Pipeline (PolynomialFeatures → Ridge)
    """
    errors = compute_reconstruction_error(model, sequences)   # (N,)
    X = errors.reshape(-1, 1)
    y = np.clip(rul_labels, 0, None).astype(float)            # clip to >= 0

    pipe = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
        ("ridge", Ridge(alpha=1.0)),
    ])
    pipe.fit(X, y)
    return pipe


def estimate_rul(
    rul_regressor: Pipeline,
    anomaly_score: float,
    min_rul: float = 0.0,
) -> dict:
    """
    Predict RUL from a single anomaly score.

    Parameters
    ----------
    rul_regressor : fitted sklearn Pipeline from fit_rul_regressor()
    anomaly_score : float
    min_rul       : floor value (RUL can't be negative)

    Returns
    -------
    dict  {rul_cycles, health_pct, confidence}
        rul_cycles  : predicted RUL in engine cycles (int, >= 0)
        health_pct  : rough engine health percentage (100 = new, 0 = at limit)
        confidence  : 'high' | 'medium' | 'low'  based on score magnitude
    """
    raw = float(rul_regressor.predict([[anomaly_score]])[0])
    rul = max(min_rul, round(raw))

    # Health percentage: interpolated from the regressor's training range.
    # We use the prediction at score=0 as the "maximum expected RUL" baseline.
    max_rul_est = float(rul_regressor.predict([[0.0]])[0])
    if max_rul_est > 0:
        health_pct = min(100.0, max(0.0, (rul / max_rul_est) * 100))
    else:
        health_pct = 0.0

    # Confidence based on how far the input score is from the training data
    # regime. Low scores (near 0) are well-calibrated; very high scores
    # extrapolate beyond the training distribution.
    if anomaly_score < 0.05:
        confidence = "high"
    elif anomaly_score < 0.15:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "rul_cycles" : int(rul),
        "health_pct" : round(health_pct, 1),
        "confidence" : confidence,
    }



# ── Model factory ─────────────────────────────────────────────────────────────

def build_autoencoder(seq_len: int, n_features: int, latent_dim: int = 64) -> Model:
    """
    Build and compile the LSTM autoencoder.

    Parameters
    ----------
    seq_len     : number of time steps per sequence (e.g. 50)
    n_features  : number of sensor channels per step
    latent_dim  : bottleneck size (inner LSTM hidden units)
    """
    inputs = Input(shape=(seq_len, n_features), name="encoder_input")

    # Encoder
    x = LSTM(128, return_sequences=True,  name="enc_lstm1")(inputs)
    x = Dropout(0.2,                      name="enc_drop1")(x)
    x = LSTM(latent_dim, return_sequences=False, name="enc_lstm2")(x)

    # Bridge
    x = RepeatVector(seq_len, name="repeat_vector")(x)

    # Decoder
    x = LSTM(latent_dim, return_sequences=True, name="dec_lstm1")(x)
    x = Dropout(0.2,                            name="dec_drop1")(x)
    x = LSTM(128,        return_sequences=True, name="dec_lstm2")(x)
    outputs = TimeDistributed(
        Dense(n_features, activation="sigmoid"), name="reconstruction"
    )(x)

    model = Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mae")
    return model


# ── Training helper ───────────────────────────────────────────────────────────

def train_autoencoder(
    model: Model,
    train_sequences: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    validation_split: float = 0.10,
    verbose: int = 1,
):
    """
    Fit the autoencoder (target == input — that's the autoencoder objective).

    Returns keras History object.
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=7,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-6, verbose=1,
        ),
    ]
    return model.fit(
        train_sequences, train_sequences,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        shuffle=True,
        verbose=verbose,
    )


# ── Inference helpers ─────────────────────────────────────────────────────────

def compute_reconstruction_error(
    model: Model,
    sequences: np.ndarray,
    per_sensor: bool = False,
) -> np.ndarray:
    """
    Mean Absolute Error between input and reconstruction.

    per_sensor=False → shape (N,)             scalar MAE per sample
    per_sensor=True  → shape (N, n_features)  MAE per sensor per sample
    """
    reconstructions = model.predict(sequences, verbose=0)
    errors = np.abs(sequences - reconstructions).mean(axis=1)   # (N, n_features)
    if per_sensor:
        return errors
    return errors.mean(axis=1)                                   # (N,)


def compute_threshold(
    model: Model,
    healthy_sequences: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """
    Derive an anomaly threshold from reconstruction errors on healthy sequences.
    Set at `percentile` of the error distribution — signals above it are anomalous.
    """
    errors = compute_reconstruction_error(model, healthy_sequences)
    return float(np.percentile(errors, percentile))


def anomaly_score_for_engine(
    model: Model,
    sequence: np.ndarray,
    feature_cols: list,
    threshold: float,
) -> dict:
    """
    Full per-engine diagnostic bundle.

    Parameters
    ----------
    sequence : ndarray  shape (seq_len, n_features)

    Returns
    -------
    dict  {anomaly_score, is_anomaly, per_sensor_errors, top3_sensors, threshold}
    """
    seq_batch  = sequence[np.newaxis, ...]                   # (1, seq_len, n_features)
    per_sensor = compute_reconstruction_error(model, seq_batch, per_sensor=True)[0]
    overall    = float(per_sensor.mean())

    sensor_errors = {col: float(per_sensor[i]) for i, col in enumerate(feature_cols)}
    top3 = sorted(sensor_errors, key=sensor_errors.get, reverse=True)[:3]

    return {
        "anomaly_score"     : overall,
        "is_anomaly"        : overall > threshold,
        "per_sensor_errors" : sensor_errors,
        "top3_sensors"      : top3,
        "threshold"         : threshold,
    }
