"""
app/app.py
----------
Streamlit front-end for the Aviation Engine Health Monitoring System.

Run from the project root:
    streamlit run app/app.py

Tabs
----
1. Pre-loaded Test Data  — pick dataset + engine, run full pipeline
2. Live Sensor Input     — paste raw sensor rows, get instant diagnosis
3. About                 — architecture overview + model metadata
"""

from __future__ import annotations
import json
import os
import pickle
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Allow imports from project root regardless of working directory
_APP_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_APP_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Aviation Engine Health Monitor",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.data_loader import (
    load_dataset, select_features, apply_scaler,
    make_sequences, SEQUENCE_LENGTH, DATA_DIR,
    COLUMN_NAMES, DROP_SENSORS,
)
from src.expert_diagnosis import full_diagnosis, SENSOR_COMPONENT_MAP
from src.gemini_api import generate_narrative
from src.report_generator import report_to_bytes
from src.model import (
    anomaly_score_for_engine,
    compute_reconstruction_error,
    estimate_rul,
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_DIR       = os.path.join(_ROOT_DIR, "models")
DATASETS        = ["FD001", "FD002", "FD003", "FD004"]
SEVERITY_COLORS = {"HIGH": "#c62828", "MEDIUM": "#ef6c00", "LOW": "#2e7d32"}

# Health-band thresholds for colour coding
_HEALTH_GREEN  = 70   # >= 70 % → green
_HEALTH_ORANGE = 40   # 40–70 % → orange


# ── Model loading (cached) ────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model artefacts…")
def load_model_artefacts():
    try:
        import tensorflow as tf

        keras_path = os.path.join(MODEL_DIR, "combined_autoencoder.keras")
        h5_path    = os.path.join(MODEL_DIR, "combined_autoencoder.h5")
        if os.path.exists(keras_path):
            model = tf.keras.models.load_model(keras_path)
        elif os.path.exists(h5_path):
            model = tf.keras.models.load_model(h5_path)
        else:
            raise FileNotFoundError(
                "No model file found. Run: python scripts/run_combined.py"
            )

        global_threshold = float(np.load(
            os.path.join(MODEL_DIR, "combined_threshold.npy")
        ))

        with open(os.path.join(MODEL_DIR, "combined_scaler.pkl"),   "rb") as f:
            scaler = pickle.load(f)
        with open(os.path.join(MODEL_DIR, "combined_features.pkl"), "rb") as f:
            feature_cols = pickle.load(f)

        per_ds_path = os.path.join(MODEL_DIR, "combined_thresholds_per_ds.pkl")
        if os.path.exists(per_ds_path):
            with open(per_ds_path, "rb") as f:
                per_ds_thresholds = pickle.load(f)
        else:
            per_ds_thresholds = {ds: global_threshold for ds in DATASETS}

        # RUL regressor (optional — only present after re-training)
        rul_regressor = None
        rul_path = os.path.join(MODEL_DIR, "combined_rul_regressor.pkl")
        if os.path.exists(rul_path):
            with open(rul_path, "rb") as f:
                rul_regressor = pickle.load(f)

        # Metadata JSON (optional)
        metadata = {}
        meta_path = os.path.join(MODEL_DIR, "combined_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                metadata = json.load(f)

        return (
            model, scaler, global_threshold, feature_cols,
            per_ds_thresholds, rul_regressor, metadata, None,
        )
    except Exception as exc:
        return None, None, None, None, {}, None, {}, str(exc)


@st.cache_data(show_spinner="Loading dataset…")
def load_cached_dataset(dataset_id: str):
    return load_dataset(dataset_id, DATA_DIR)


# ── Inference helpers ─────────────────────────────────────────────────────────

def score_engine(model, threshold, feature_cols, seq: np.ndarray) -> dict:
    return anomaly_score_for_engine(model, seq, feature_cols, threshold)


def compute_cycle_scores(model, feature_cols, engine_df: pd.DataFrame) -> list[float]:
    seqs, _ = make_sequences(engine_df, feature_cols, SEQUENCE_LENGTH)
    if len(seqs) == 0:
        return []
    errors = compute_reconstruction_error(model, seqs)
    # Pad first (SEQUENCE_LENGTH - 1) positions so index aligns with cycles
    pad = [float(errors[0])] * (SEQUENCE_LENGTH - 1)
    return pad + [float(e) for e in errors]


def get_rul_estimate(rul_regressor, anomaly_score: float) -> dict | None:
    """Return RUL dict or None if no regressor is available."""
    if rul_regressor is None:
        return None
    return estimate_rul(rul_regressor, anomaly_score)


# ── Plotly helpers ────────────────────────────────────────────────────────────

def plotly_sensor_bars(per_sensor_errors: dict) -> go.Figure:
    sorted_items = sorted(per_sensor_errors.items(), key=lambda x: x[1], reverse=True)
    sensors = [s for s, _ in sorted_items]
    errors  = [e for _, e in sorted_items]
    colors  = ["#c62828" if i < 3 else "#1565c0" for i in range(len(sensors))]

    fig = go.Figure(go.Bar(
        x=errors[::-1], y=sensors[::-1], orientation="h",
        marker_color=colors[::-1],
        text=[f"{e:.5f}" for e in errors[::-1]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Sensor Reconstruction Errors", xaxis_title="MAE",
        height=420, margin=dict(l=10, r=80, t=40, b=30),
        plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#eeeeee")
    return fig


def plotly_timeline(cycle_scores: list, threshold: float, engine_id) -> go.Figure:
    cycles = list(range(1, len(cycle_scores) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cycles, y=cycle_scores, mode="lines", name="Anomaly Score",
        line=dict(color="#1565c0", width=1.5),
        fill="tozeroy", fillcolor="rgba(21,101,192,0.1)",
    ))
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#c62828",
        annotation_text=f"Threshold ({threshold:.5f})",
        annotation_position="top right",
    )
    anom_y = [s if s > threshold else None for s in cycle_scores]
    fig.add_trace(go.Scatter(
        x=cycles, y=anom_y, mode="lines", name="Anomalous",
        line=dict(color="#c62828", width=0),
        fill="tozeroy", fillcolor="rgba(198,40,40,0.18)",
    ))
    fig.update_layout(
        title=f"Anomaly Score Timeline — Engine {engine_id}",
        xaxis_title="Cycle", yaxis_title="MAE", height=350,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        plot_bgcolor="white", margin=dict(l=10, r=10, t=40, b=30),
    )
    fig.update_xaxes(gridcolor="#eeeeee")
    fig.update_yaxes(gridcolor="#eeeeee")
    return fig


def plotly_anomaly_gauge(score: float, threshold: float) -> go.Figure:
    max_val = max(score * 1.5, threshold * 1.5, 0.05)
    color   = "#c62828" if score > threshold else "#2e7d32"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score, 6),
        number={"font": {"size": 26, "color": color}},
        gauge={
            "axis"     : {"range": [0, max_val], "tickformat": ".4f"},
            "bar"      : {"color": color},
            "bgcolor"  : "white",
            "steps"    : [
                {"range": [0, threshold],  "color": "rgba(46,125,50,0.15)"},
                {"range": [threshold, max_val], "color": "rgba(198,40,40,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#333", "width": 3},
                "thickness": 0.75, "value": threshold,
            },
        },
        title={"text": "Anomaly Score", "font": {"size": 13}},
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=10))
    return fig


def plotly_health_gauge(health_pct: float) -> go.Figure:
    """Circular gauge showing engine health 0–100 %."""
    if health_pct >= _HEALTH_GREEN:
        color = "#2e7d32"
    elif health_pct >= _HEALTH_ORANGE:
        color = "#ef6c00"
    else:
        color = "#c62828"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(health_pct, 1),
        number={"suffix": "%", "font": {"size": 26, "color": color}},
        gauge={
            "axis"   : {"range": [0, 100]},
            "bar"    : {"color": color},
            "bgcolor": "white",
            "steps"  : [
                {"range": [0,  _HEALTH_ORANGE], "color": "rgba(198,40,40,0.12)"},
                {"range": [_HEALTH_ORANGE, _HEALTH_GREEN], "color": "rgba(239,108,0,0.12)"},
                {"range": [_HEALTH_GREEN, 100], "color": "rgba(46,125,50,0.12)"},
            ],
        },
        title={"text": "Engine Health", "font": {"size": 13}},
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=30, b=10))
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(model_ok: bool, metadata: dict, model_error: str | None):
    with st.sidebar:
        st.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e2/"
            "Jet_engine.svg/320px-Jet_engine.svg.png",
            width=200,
        )
        st.title("✈ Engine Health Monitor")
        st.markdown("---")

        if model_ok:
            st.success("Model loaded")
            if metadata:
                trained_at = metadata.get("trained_at", "")
                if trained_at:
                    st.caption(f"Trained: {trained_at[:10]}")
                rul_mae = metadata.get("rul_regressor_mae")
                if rul_mae is not None:
                    st.caption(f"RUL regressor MAE: ±{rul_mae:.0f} cycles")
        else:
            st.error("Model not found")
            st.info(
                "**Train the model first:**\n\n"
                "```\npython scripts/run_combined.py\n```\n\n"
                "This saves the autoencoder, scaler, thresholds, "
                "RUL regressor, and metadata to `models/`."
            )
            if model_error:
                with st.expander("Error details"):
                    st.code(model_error)

        st.markdown("---")
        st.markdown(
            "LSTM Autoencoder anomaly detection on NASA C-MAPSS turbofan data.\n\n"
            "Reconstruction error → expert rules → RUL estimate → Gemini narrative → PDF."
        )
        st.caption("NASA C-MAPSS · Gemini 2.0 Flash")


# ── Tab 1: Pre-loaded test data ───────────────────────────────────────────────

def tab_preloaded(model, scaler, global_threshold, feature_cols,
                  per_ds_thresholds, rul_regressor):
    st.header("Pre-loaded Test Data Analysis")

    col1, col2 = st.columns([1, 2])
    with col1:
        dataset_id = st.selectbox("Dataset", DATASETS)
        raw        = load_cached_dataset(dataset_id)
        max_engine = int(raw["test"]["unit_id"].max())
        engine_id  = st.number_input(
            f"Engine ID (1–{max_engine})",
            min_value=1, max_value=max_engine, value=1,
        )
        use_log  = st.checkbox("Include maintenance log")
        log_text = ""
        if use_log:
            log_text = st.text_area(
                "Maintenance log text",
                placeholder="e.g. 'Borescope inspection noted HPC blade erosion ...'",
                height=120,
            )
        run_btn = st.button("Run Diagnosis", type="primary", use_container_width=True)

    with col2:
        engine_df_raw = raw["test"][raw["test"]["unit_id"] == engine_id]
        n_cycles      = int(engine_df_raw["cycle"].max())
        st.metric("Engine cycles in test set", n_cycles)
        st.caption("1 operating condition" if dataset_id in ("FD001", "FD003")
                   else "6 operating conditions")
        engine_idx = engine_id - 1
        if engine_idx < len(raw["rul"]):
            true_rul = int(raw["rul"][engine_idx])
            st.metric("True RUL (ground truth)", true_rul)

    if not run_btn:
        st.info("Select a dataset and engine, then click **Run Diagnosis**.")
        return

    if model is None:
        st.error("No model loaded. Run `python scripts/run_combined.py` first.")
        return

    threshold = per_ds_thresholds.get(dataset_id, global_threshold)

    with st.spinner("Analysing engine data…"):
        engine_df_sc = apply_scaler(
            select_features(engine_df_raw.copy()), scaler, feature_cols
        )

        last_seq = engine_df_sc[feature_cols].values
        if len(last_seq) < SEQUENCE_LENGTH:
            pad      = np.tile(last_seq[0], (SEQUENCE_LENGTH - len(last_seq), 1))
            last_seq = np.vstack([pad, last_seq])
        last_seq = last_seq[-SEQUENCE_LENGTH:].astype(np.float32)

        result       = score_engine(model, threshold, feature_cols, last_seq)
        cycle_scores = compute_cycle_scores(model, feature_cols, engine_df_sc)
        rul_est      = get_rul_estimate(rul_regressor, result["anomaly_score"])
        diag         = full_diagnosis(
            top_sensors       = result["top3_sensors"],
            per_sensor_errors = result["per_sensor_errors"],
            anomaly_score     = result["anomaly_score"],
            threshold         = threshold,
            log_text          = log_text if use_log else None,
        )

    true_rul_val = int(raw["rul"][engine_id - 1]) if engine_id - 1 < len(raw["rul"]) else None
    _render_results(
        result, diag, cycle_scores, threshold,
        engine_id, dataset_id,
        rul_est      = rul_est,
        true_rul     = true_rul_val,
        log_text     = log_text if use_log else "",
    )


# ── Shared result renderer ────────────────────────────────────────────────────

def _render_results(
    result, diag, cycle_scores, threshold,
    engine_id, dataset_id,
    rul_est: dict | None = None,
    true_rul: int | None = None,
    log_text: str = "",
):
    is_anom  = result["is_anomaly"]
    severity = diag["severity"]

    # ── Status banner ──────────────────────────────────────────────────────
    if is_anom:
        st.error(
            f"⚠️  ANOMALY DETECTED  |  Severity: {severity}  |  "
            f"Fault: {diag['fault_name']}"
        )
    else:
        st.success("✅  Engine status: NOMINAL — reconstruction error within healthy baseline")

    # ── Top metric cards ───────────────────────────────────────────────────
    cols = st.columns(5 if rul_est else 4)
    cols[0].metric("Anomaly Score",   f"{result['anomaly_score']:.6f}")
    cols[1].metric("Threshold",       f"{threshold:.6f}")
    cols[2].metric("Suspected Fault", diag["fault_name"])
    cols[3].metric("Top Sensor",      result["top3_sensors"][0] if result["top3_sensors"] else "—")
    if rul_est:
        conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(rul_est["confidence"], "")
        cols[4].metric(
            "Est. RUL",
            f"{rul_est['rul_cycles']} cycles",
            delta=f"vs true: {rul_est['rul_cycles'] - true_rul:+d}" if true_rul is not None else None,
            help=f"Confidence: {rul_est['confidence']} {conf_icon}. "
                 "Based on polynomial regression of anomaly score on labelled training data. "
                 "±uncertainty increases for high anomaly scores.",
        )

    # ── Gauges row ─────────────────────────────────────────────────────────
    if rul_est:
        g1, g2, bc = st.columns([1, 1, 2])
        with g1:
            st.plotly_chart(
                plotly_anomaly_gauge(result["anomaly_score"], threshold),
                use_container_width=True,
            )
        with g2:
            st.plotly_chart(
                plotly_health_gauge(rul_est["health_pct"]),
                use_container_width=True,
            )
        with bc:
            st.plotly_chart(
                plotly_sensor_bars(result["per_sensor_errors"]),
                use_container_width=True,
            )
    else:
        gc, bc = st.columns([1, 2])
        with gc:
            st.plotly_chart(
                plotly_anomaly_gauge(result["anomaly_score"], threshold),
                use_container_width=True,
            )
        with bc:
            st.plotly_chart(
                plotly_sensor_bars(result["per_sensor_errors"]),
                use_container_width=True,
            )

    # ── RUL detail card ────────────────────────────────────────────────────
    if rul_est:
        conf      = rul_est["confidence"]
        conf_col  = {"high": "green", "medium": "orange", "low": "red"}.get(conf, "grey")
        health    = rul_est["health_pct"]

        with st.container(border=True):
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Predicted RUL", f"{rul_est['rul_cycles']} cycles")
            rc2.metric("Engine Health", f"{health:.1f}%")
            rc3.metric("Estimate Confidence", conf.capitalize())

            if true_rul is not None:
                error_cycles = abs(rul_est["rul_cycles"] - true_rul)
                st.caption(
                    f"Ground-truth RUL: **{true_rul}** cycles  |  "
                    f"Prediction error: **{error_cycles}** cycles  |  "
                    f"Confidence: :{conf_col}[{conf}]"
                )
            else:
                st.caption(
                    f"Confidence: :{conf_col}[{conf}] — "
                    "confidence is low when the anomaly score is far from the healthy training range."
                )

    # ── Timeline ───────────────────────────────────────────────────────────
    if cycle_scores:
        st.plotly_chart(
            plotly_timeline(cycle_scores, threshold, engine_id),
            use_container_width=True,
        )

    # ── Expert diagnosis expander ──────────────────────────────────────────
    with st.expander("Expert Rule-Based Diagnosis", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Fault:** {diag['fault_name']}")
            st.markdown(f"**Severity:** {severity}")
            st.markdown(f"**Components:** {', '.join(diag['components'])}")
        with c2:
            st.markdown(f"**Top 3 sensors:** {', '.join(diag['top3_sensors'])}")
            rule = ", ".join(sorted(diag["matched_rule"])) if diag["matched_rule"] else "Fallback (catch-all)"
            st.markdown(f"**Matched rule:** {rule}")
        st.info(diag["description"])

    # ── Gemini narrative ───────────────────────────────────────────────────
    with st.expander("AI-Generated Narrative (Gemini)", expanded=True):
        with st.spinner("Generating narrative…"):
            narrative = generate_narrative(diag)
        if narrative["error"]:
            st.warning(f"Gemini error: {narrative['error']}")
            if narrative.get("narrative"):
                st.write(narrative["narrative"])
        else:
            st.markdown("**Expert Diagnosis**")
            st.write(narrative["expert_diag"])
            st.markdown("**Probable Root Cause & Recommended Action**")
            st.write(narrative["root_cause"])
            st.caption(f"Generated by {narrative['model']}")

    # ── Maintenance log evidence ───────────────────────────────────────────
    if log_text and diag.get("log_evidence"):
        with st.expander("Maintenance Log Evidence"):
            log_ev = diag["log_evidence"]
            st.metric("Evidence Score", f"{log_ev['evidence_score']:.0%}")
            if log_ev["corroborating"]:
                st.success(f"Keywords found: {', '.join(log_ev['found_keywords'])}")
            else:
                st.warning("No matching keywords found in log text.")
            st.markdown("**Highlighted excerpt:**")
            st.markdown(log_ev["highlighted_text"].replace("**", "`"))

    # ── PDF download ───────────────────────────────────────────────────────
    st.markdown("---")
    with st.spinner("Building PDF report…"):
        pdf_bytes = report_to_bytes(
            engine_id    = engine_id,
            dataset_id   = dataset_id,
            diagnosis    = diag,
            narrative    = narrative,
            cycle_scores = cycle_scores,
            rul_est      = rul_est,
            true_rul     = true_rul,
        )
    st.download_button(
        label              = "⬇ Download Diagnostic Report (PDF)",
        data               = pdf_bytes,
        file_name          = f"engine_{engine_id}_{dataset_id}_report.pdf",
        mime               = "application/pdf",
        use_container_width= True,
    )


# ── Tab 2: Live sensor input ──────────────────────────────────────────────────

def tab_live(model, scaler, global_threshold, feature_cols, rul_regressor):
    st.header("Live Sensor Reading Analysis")
    st.markdown(
        f"Paste space-separated sensor rows — one cycle per row, "
        f"at least **{SEQUENCE_LENGTH} rows** needed.  \n"
        "Column order: `unit_id  cycle  op1  op2  op3  s1…s21` (26 values).  \n"
        "The first two columns are optional; 24- or 25-column rows are also accepted."
    )

    raw_text     = st.text_area("Sensor data", height=180,
                                placeholder="Paste rows from test_FDxxx.txt here…")
    engine_label = st.text_input("Engine label", value="LIVE-001")
    log_text     = st.text_area("Maintenance log (optional)", height=80,
                                placeholder="Paste maintenance notes here…")

    if not st.button("Analyse", type="primary", use_container_width=True):
        st.info("Paste sensor data above and click **Analyse**.")
        return

    if model is None:
        st.error("No model loaded. Run `python scripts/run_combined.py` first.")
        return

    threshold = global_threshold   # no dataset context for live input

    lines = [
        ln.strip() for ln in raw_text.strip().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    if not lines:
        st.error("No data found — please paste sensor rows.")
        return

    try:
        rows = [[float(v) for v in ln.split()] for ln in lines]
        arr  = np.array(rows)
    except ValueError as exc:
        st.error(f"Parse error: {exc}")
        return

    n_cols = arr.shape[1]
    if n_cols == 26:
        df = pd.DataFrame(arr, columns=COLUMN_NAMES)
    elif n_cols == 24:
        df = pd.DataFrame(arr, columns=COLUMN_NAMES[2:])
        df.insert(0, "cycle",   range(1, len(df) + 1))
        df.insert(0, "unit_id", 1)
    elif n_cols == 25:
        df = pd.DataFrame(arr, columns=COLUMN_NAMES[1:])
        df.insert(0, "unit_id", 1)
    else:
        st.error(f"Expected 24–26 columns per row, got {n_cols}.")
        return

    df = df.drop(columns=[c for c in DROP_SENSORS if c in df.columns], errors="ignore")

    with st.spinner("Running analysis…"):
        df_sc    = apply_scaler(df, scaler, feature_cols)
        last_seq = df_sc[feature_cols].values
        if len(last_seq) < SEQUENCE_LENGTH:
            st.warning(f"Only {len(last_seq)} cycles provided — padding to {SEQUENCE_LENGTH}.")
            pad      = np.tile(last_seq[0], (SEQUENCE_LENGTH - len(last_seq), 1))
            last_seq = np.vstack([pad, last_seq])
        last_seq = last_seq[-SEQUENCE_LENGTH:].astype(np.float32)

        result = score_engine(model, threshold, feature_cols, last_seq)

        cycle_scores: list[float] = []
        if len(df_sc) >= SEQUENCE_LENGTH:
            cycle_scores = compute_cycle_scores(model, feature_cols, df_sc)

        rul_est = get_rul_estimate(rul_regressor, result["anomaly_score"])

        diag = full_diagnosis(
            top_sensors       = result["top3_sensors"],
            per_sensor_errors = result["per_sensor_errors"],
            anomaly_score     = result["anomaly_score"],
            threshold         = threshold,
            log_text          = log_text if log_text.strip() else None,
        )

    _render_results(
        result, diag, cycle_scores, threshold,
        engine_id    = engine_label,
        dataset_id   = "LIVE",
        rul_est      = rul_est,
        true_rul     = None,
        log_text     = log_text,
    )


# ── Tab 3: About ──────────────────────────────────────────────────────────────

def tab_about(metadata: dict):
    st.header("About this System")
    st.markdown("""
### Pipeline Architecture

```
Raw sensor data → MinMax scaling → 50-cycle sliding windows
      ↓
LSTM Autoencoder → reconstruction error (anomaly score)
      ↓
Polynomial Ridge regressor → Estimated RUL + Engine Health %
      ↓
per-sensor error → top-3 deviating sensors
      ↓
Expert rule base → fault hypothesis + affected component
      ↓
(optional) maintenance log NLP → corroborating evidence
      ↓
Gemini 2.0 Flash → natural-language diagnosis
      ↓
PDF / Streamlit → user-facing report
```

### Layer 1 — LSTM Autoencoder
Trained **only on healthy (early-life) cycles**. At inference, degraded engines produce
higher reconstruction error (MAE) because the model never saw degraded patterns during training.

### Layer 1b — RUL Estimator
A **polynomial Ridge regression** is fitted post-training, mapping anomaly score → RUL
using all four datasets' labelled training data. This gives a coarse but calibrated
remaining-life estimate without needing a separate supervised model.

### Layer 2 — Expert Rule-Based Diagnosis
Maps per-sensor reconstruction errors → physical components → fault hypotheses using
C-MAPSS domain knowledge encoded as deterministic rules. Optional spaCy NLP searches
maintenance logs for corroborating keywords.

### Layer 3 — LLM Narrative (Gemini)
Turns the structured diagnostic output into readable prose.
The LLM **does not** perform diagnostic reasoning — it only narrates what Layers 1 & 2 found.

### Dataset
NASA C-MAPSS — 4 sub-datasets (FD001–FD004), 21 sensors, 3 op-settings.
FD001/FD003: single operating condition.  FD002/FD004: 6 operating conditions.
""")

    # ── Model metadata card ────────────────────────────────────────────────
    if metadata:
        with st.expander("Model Training Metadata", expanded=True):
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Trained at",     metadata.get("trained_at", "—")[:19].replace("T", " "))
            mc2.metric("Epochs run",     metadata.get("epochs_run", "—"))
            mc3.metric("Best val loss",  f"{metadata.get('best_val_loss', 0):.5f}")

            mc4, mc5, mc6 = st.columns(3)
            mc4.metric("Features",       metadata.get("n_features", "—"))
            mc5.metric("Sequence len",   metadata.get("sequence_length", "—"))
            mc6.metric("RUL MAE",        f"±{metadata.get('rul_regressor_mae', '—')} cycles"
                                          if metadata.get("rul_regressor_mae") else "—")

            st.markdown("**Per-dataset thresholds:**")
            thresh_data = [
                {"Dataset": k, "Threshold": f"{v:.6f}"}
                for k, v in metadata.get("per_ds_thresholds", {}).items()
            ]
            if thresh_data:
                st.dataframe(pd.DataFrame(thresh_data), use_container_width=True, hide_index=True)

            st.markdown("**Test set evaluation:**")
            eval_rows = [
                {
                    "Dataset" : ds,
                    "Threshold": f"{v['threshold']:.6f}",
                    "Mean error": f"{v['mean_error']:.6f}",
                    "Max error":  f"{v['max_error']:.6f}",
                    "Flagged":    f"{v['flagged']}/{v['total']}",
                }
                for ds, v in metadata.get("test_evaluation", {}).items()
            ]
            if eval_rows:
                st.dataframe(pd.DataFrame(eval_rows), use_container_width=True, hide_index=True)
    else:
        st.info(
            "No metadata found. Retrain with the latest script to generate "
            "`models/combined_metadata.json`."
        )

    with st.expander("Sensor → Component Mapping"):
        rows = [{"Sensor": k, "Component": v} for k, v in SENSOR_COMPONENT_MAP.items()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("""
### Project Structure
```
aviation/
├── app/            ← Streamlit UI  (this file)
├── src/            ← core library modules
│   ├── data_loader.py       loading, scaling, sequence creation
│   ├── model.py             LSTM autoencoder + RUL regressor
│   ├── expert_diagnosis.py  rule-based fault diagnosis + NLP log search
│   ├── gemini_api.py        Gemini REST API wrapper
│   └── report_generator.py  PDF assembly
├── scripts/
│   ├── run_combined.py      ← train master model (run this)
│   └── run_project_fd00x.py ← per-dataset training (optional)
├── data/           ← NASA C-MAPSS dataset files
├── models/         ← saved artefacts after training
└── reports/        ← generated PDFs
```
""")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    (
        model, scaler, global_threshold, feature_cols,
        per_ds_thresholds, rul_regressor, metadata, model_error,
    ) = load_model_artefacts()

    render_sidebar(model is not None, metadata, model_error)

    st.title("✈ Aviation Engine Health Monitoring System")
    st.caption("LSTM Autoencoder · RUL Estimator · Expert Rules · Gemini AI · NASA C-MAPSS")

    tab1, tab2, tab3 = st.tabs([
        "📂 Pre-loaded Test Data",
        "🔴 Live Sensor Input",
        "ℹ️ About",
    ])

    with tab1:
        tab_preloaded(
            model, scaler, global_threshold, feature_cols,
            per_ds_thresholds, rul_regressor,
        )
    with tab2:
        tab_live(model, scaler, global_threshold, feature_cols, rul_regressor)
    with tab3:
        tab_about(metadata)


if __name__ == "__main__":
    main()
