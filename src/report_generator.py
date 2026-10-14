"""
src/report_generator.py
-----------------------
Assembles a downloadable PDF diagnostic report using FPDF2 and matplotlib.

Report sections
---------------
1. Cover / header
2. Anomaly summary
3. Expert diagnosis
4. LLM narrative (Gemini)
5. Sensor error bar chart
6. Anomaly score timeline chart
7. Maintenance log evidence (optional)
8. Full sensor error table
"""

from __future__ import annotations
import io
import os
import datetime
import tempfile
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF

# Reports are written to <project_root>/reports/
_SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR     = os.path.dirname(_SRC_DIR)
DEFAULT_OUTPUT_DIR = os.path.join(_ROOT_DIR, "reports")


# ── PDF class ─────────────────────────────────────────────────────────────────

class DiagnosticReport(FPDF):
    BRAND_BLUE  = (13,  71, 161)
    BRAND_RED   = (198, 40,  40)
    BRAND_GREEN = (27,  94,  32)
    BRAND_GREY  = (97,  97,  97)

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*self.BRAND_BLUE)
        self.cell(0, 8, "AIRCRAFT ENGINE HEALTH MONITORING SYSTEM", align="C", ln=True)
        self.set_draw_color(*self.BRAND_BLUE)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*self.BRAND_GREY)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        self.cell(0, 10, f"Page {self.page_no()} | Generated: {ts}", align="C")

    def section_title(self, title: str):
        self.ln(4)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*self.BRAND_BLUE)
        self.set_fill_color(232, 240, 254)
        self.cell(0, 8, f"  {title}", ln=True, fill=True)
        self.ln(2)

    def kv_row(self, key: str, value: str, value_color=None):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(50, 50, 50)
        self.cell(60, 7, key + ":", ln=False)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(*(value_color if value_color else (20, 20, 20)))
        self.cell(0, 7, str(value), ln=True)

    def body_text(self, text: str):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def embed_image_from_bytes(self, img_bytes: bytes, w: int = 180):
        # Write to a process-unique temp file so concurrent report generations
        # (multiple Streamlit users) don't collide on the same path.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_f:
            tmp_path = tmp_f.name
            tmp_f.write(img_bytes)
        try:
            self.image(tmp_path, x=15, w=w)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ── Chart generators ──────────────────────────────────────────────────────────

def _sensor_error_chart(per_sensor_errors: dict, top_n: int = 10) -> bytes:
    sorted_sensors = sorted(per_sensor_errors.items(), key=lambda x: x[1], reverse=True)
    sensors = [s for s, _ in sorted_sensors[:top_n]]
    errors  = [e for _, e in sorted_sensors[:top_n]]
    colors  = ["#c62828" if i < 3 else "#1565c0" for i in range(len(sensors))]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(sensors[::-1], errors[::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Reconstruction Error (MAE)", fontsize=9)
    ax.set_title(f"Top {top_n} Sensor Reconstruction Errors", fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, errors[::-1]):
        ax.text(bar.get_width() + 0.0001, bar.get_y() + bar.get_height() / 2,
                f"{val:.5f}", va="center", fontsize=7, color="#333333")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _anomaly_timeline_chart(
    cycle_scores: Optional[list],
    threshold: float,
    engine_id,
) -> bytes:
    if not cycle_scores:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "Cycle-level data not available",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.axis("off")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        return buf.getvalue()

    cycles = list(range(1, len(cycle_scores) + 1))
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(cycles, cycle_scores, color="#1565c0", linewidth=1.2, label="Anomaly Score")
    ax.axhline(threshold, color="#c62828", linewidth=1.2, linestyle="--",
               label=f"Threshold ({threshold:.5f})")
    ax.fill_between(cycles, cycle_scores, threshold,
                    where=[s > threshold for s in cycle_scores],
                    alpha=0.25, color="#c62828", label="Anomalous Region")
    ax.set_xlabel("Cycle", fontsize=9)
    ax.set_ylabel("MAE", fontsize=9)
    ax.set_title(f"Anomaly Score Timeline — Engine {engine_id}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ── Main report function ──────────────────────────────────────────────────────

def generate_report(
    engine_id,
    dataset_id: str,
    diagnosis: dict,
    narrative: dict,
    cycle_scores: Optional[list] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    rul_est: Optional[dict] = None,
    true_rul: Optional[int] = None,
) -> str:
    """Build and save a PDF report.  Returns the absolute path to the saved file."""
    os.makedirs(output_dir, exist_ok=True)
    ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"engine_{engine_id}_{dataset_id}_{ts}.pdf")

    pdf = DiagnosticReport(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # 1. Cover
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*DiagnosticReport.BRAND_BLUE)
    pdf.cell(0, 12, "Engine Diagnostic Report", align="C", ln=True)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*DiagnosticReport.BRAND_GREY)
    pdf.cell(0, 8, f"Dataset: {dataset_id}   |   Engine Unit: {engine_id}", align="C", ln=True)
    pdf.cell(0, 7, datetime.datetime.now().strftime("%A, %d %B %Y  %H:%M UTC"), align="C", ln=True)
    pdf.ln(4)

    # 2. Anomaly summary
    score     = diagnosis["anomaly_score"]
    threshold = diagnosis["threshold"]
    is_anom   = score > threshold
    status_color = DiagnosticReport.BRAND_RED if is_anom else DiagnosticReport.BRAND_GREEN

    pdf.section_title("1. Anomaly Detection Summary")
    pdf.kv_row("Engine ID",     str(engine_id))
    pdf.kv_row("Dataset",       dataset_id)
    pdf.kv_row("Anomaly Score", f"{score:.6f}")
    pdf.kv_row("Threshold",     f"{threshold:.6f}")
    pdf.kv_row("Status",        "ANOMALY DETECTED" if is_anom else "NOMINAL",
               value_color=status_color)
    pdf.kv_row("Severity",      diagnosis.get("severity", "N/A"),
               value_color=DiagnosticReport.BRAND_RED
               if diagnosis.get("severity") == "HIGH" else None)
    pdf.kv_row("Top 3 Sensors", ", ".join(diagnosis.get("top3_sensors", [])))

    # RUL estimate block (only when regressor is available)
    if rul_est:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*DiagnosticReport.BRAND_BLUE)
        pdf.set_fill_color(232, 240, 254)
        pdf.cell(0, 7, "  Remaining Useful Life Estimate", ln=True, fill=True)
        pdf.ln(1)
        pdf.kv_row("Predicted RUL",    f"{rul_est['rul_cycles']} cycles")
        pdf.kv_row("Engine Health",    f"{rul_est['health_pct']:.1f}%")
        conf = rul_est['confidence'].capitalize()
        conf_color = (
            DiagnosticReport.BRAND_GREEN  if rul_est['confidence'] == "high"   else
            (239, 108,  0)                if rul_est['confidence'] == "medium" else
            DiagnosticReport.BRAND_RED
        )
        pdf.kv_row("Est. Confidence",  conf, value_color=conf_color)
        if true_rul is not None:
            error_cycles = abs(rul_est["rul_cycles"] - true_rul)
            pdf.kv_row("True RUL",     f"{true_rul} cycles (ground truth)")
            pdf.kv_row("Error",        f"{error_cycles} cycles")

    # 3. Expert diagnosis
    pdf.section_title("2. Rule-Based Expert Diagnosis")
    pdf.kv_row("Suspected Fault",     diagnosis.get("fault_name", "N/A"))
    pdf.kv_row("Affected Components", ", ".join(diagnosis.get("components", [])))
    pdf.kv_row("Severity",            diagnosis.get("severity", "N/A"))
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(50, 50, 50)
    pdf.cell(0, 6, "Diagnostic Reasoning:", ln=True)
    pdf.body_text(diagnosis.get("description", ""))

    # 4. LLM narrative
    pdf.section_title("3. AI-Generated Expert Narrative")
    if narrative.get("error"):
        pdf.body_text(f"[Gemini API error: {narrative['error']}]")
    else:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.cell(0, 6, "Expert Diagnosis:", ln=True)
        pdf.body_text(narrative.get("expert_diag", ""))
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 6, "Probable Root Cause & Recommended Action:", ln=True)
        pdf.body_text(narrative.get("root_cause", ""))
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(*DiagnosticReport.BRAND_GREY)
        pdf.cell(0, 5, f"Narrative generated by {narrative.get('model', 'LLM')}", ln=True)

    # 5. Sensor error chart
    pdf.section_title("4. Sensor Reconstruction Error Analysis")
    pdf.embed_image_from_bytes(_sensor_error_chart(diagnosis.get("per_sensor_errors", {})), w=170)

    # 6. Anomaly timeline
    pdf.section_title("5. Anomaly Score Timeline")
    pdf.embed_image_from_bytes(_anomaly_timeline_chart(cycle_scores, threshold, engine_id), w=170)

    # 7. Maintenance log evidence
    log_ev = diagnosis.get("log_evidence")
    if log_ev:
        pdf.section_title("6. Maintenance Log Evidence")
        pdf.kv_row("Keywords Found",  ", ".join(log_ev.get("found_keywords", [])) or "None")
        pdf.kv_row("Evidence Score",  f"{log_ev.get('evidence_score', 0):.0%}")
        pdf.kv_row("Corroborating",   "Yes" if log_ev.get("corroborating") else "No")
        if log_ev.get("highlighted_text"):
            excerpt = log_ev["highlighted_text"][:800]
            if len(log_ev["highlighted_text"]) > 800:
                excerpt += "  [truncated]"
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 5, "Log Excerpt:", ln=True)
            pdf.body_text(excerpt)

    # 8. Sensor table
    pdf.section_title("7. Full Sensor Error Table")
    # Support both `import src.expert_diagnosis` and direct `import expert_diagnosis`
    try:
        from src.expert_diagnosis import SENSOR_COMPONENT_MAP
    except ImportError:
        from expert_diagnosis import SENSOR_COMPONENT_MAP
    sorted_errors = sorted(
        diagnosis.get("per_sensor_errors", {}).items(),
        key=lambda x: x[1], reverse=True,
    )

    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(*DiagnosticReport.BRAND_BLUE)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(40, 6, "Sensor",            border=1, fill=True)
    pdf.cell(60, 6, "Recon. Error (MAE)", border=1, fill=True)
    pdf.cell(60, 6, "Component",          border=1, fill=True, ln=True)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(20, 20, 20)
    for i, (sensor, error) in enumerate(sorted_errors):
        fill = i % 2 == 0
        pdf.set_fill_color(240, 244, 255) if fill else pdf.set_fill_color(255, 255, 255)
        component = SENSOR_COMPONENT_MAP.get(sensor, "—")
        pdf.cell(40, 6, sensor,         border=1, fill=fill)
        pdf.cell(60, 6, f"{error:.6f}", border=1, fill=fill)
        pdf.cell(60, 6, component,      border=1, fill=fill, ln=True)

    pdf.output(filepath)
    return os.path.abspath(filepath)


def report_to_bytes(
    engine_id,
    dataset_id: str,
    diagnosis: dict,
    narrative: dict,
    cycle_scores: Optional[list] = None,
    rul_est: Optional[dict] = None,
    true_rul: Optional[int] = None,
) -> bytes:
    """Return the PDF as bytes (for Streamlit download buttons)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = generate_report(
            engine_id, dataset_id, diagnosis, narrative,
            cycle_scores=cycle_scores, output_dir=tmpdir,
            rul_est=rul_est, true_rul=true_rul,
        )
        with open(path, "rb") as f:
            return f.read()
