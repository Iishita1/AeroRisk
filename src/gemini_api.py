"""
src/gemini_api.py
-----------------
Wraps the Google Gemini REST API to turn structured anomaly data into
human-readable narrative.  Layer 3 of the pipeline.

The LLM narrates what Layers 1 & 2 already determined — it does NOT perform
independent diagnostic reasoning.
"""

import os
import json
import requests
from dotenv import load_dotenv

# .env is at the project root (one level above src/)
_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
load_dotenv(os.path.join(_ROOT_DIR, ".env"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)
TIMEOUT_SECONDS = 30


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_prompt(diagnosis: dict) -> str:
    score     = diagnosis.get("anomaly_score", 0.0)
    threshold = diagnosis.get("threshold", 0.0)
    fault     = diagnosis.get("fault_name", "Unknown")
    severity  = diagnosis.get("severity", "UNKNOWN")
    desc      = diagnosis.get("description", "")
    components= ", ".join(diagnosis.get("components", []))
    top3      = ", ".join(diagnosis.get("top3_sensors", []))

    sensor_lines = "\n".join(
        f"  - {k}: {v:.5f}"
        for k, v in sorted(
            diagnosis.get("per_sensor_errors", {}).items(),
            key=lambda x: x[1], reverse=True,
        )[:6]
    )

    log_block = ""
    if diagnosis.get("log_evidence") and diagnosis["log_evidence"]["corroborating"]:
        kws = ", ".join(diagnosis["log_evidence"]["found_keywords"])
        log_block = (
            f"\n\nMaintenance Log Evidence (keywords found): {kws}\n"
            f"Evidence score: {diagnosis['log_evidence']['evidence_score']:.0%}"
        )

    return f"""You are a senior aircraft engine health-monitoring engineer writing a concise,
professional diagnostic report. Use technical but accessible language.

--- ANOMALY DETECTION RESULTS ---
Overall Anomaly Score   : {score:.5f}
Anomaly Threshold       : {threshold:.5f}
Status                  : {"ANOMALY DETECTED" if score > threshold else "NOMINAL"}
Severity                : {severity}

--- EXPERT RULE-BASED DIAGNOSIS ---
Suspected Fault         : {fault}
Affected Components     : {components}
Top 3 Deviating Sensors : {top3}
Diagnostic Logic        : {desc}

--- TOP SENSOR RECONSTRUCTION ERRORS ---
{sensor_lines}
{log_block}

--- YOUR TASK ---
Write exactly two paragraphs:

1. **Expert Diagnosis** (3-5 sentences): Describe what the anomaly detection
   system found, which sensors deviated and what that physically means, and which
   engine component is most likely affected.

2. **Probable Root Cause & Recommended Action** (3-5 sentences): Based on the
   fault pattern identified, explain the most probable root cause, the potential
   consequences if left unaddressed, and what maintenance action should be taken.

Be specific about sensor names and component names. State conclusions clearly.
Do not repeat the raw numbers verbatim."""


# ── API call ──────────────────────────────────────────────────────────────────

def generate_narrative(diagnosis: dict) -> dict:
    """
    Call Gemini and return the generated narrative.

    Returns dict {narrative, expert_diag, root_cause, model, error}
    """
    if not GEMINI_API_KEY:
        return _stub_narrative(diagnosis)

    payload = {
        "contents": [{"parts": [{"text": _build_prompt(diagnosis)}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 512},
    }
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY,
    }

    try:
        resp = requests.post(
            GEMINI_ENDPOINT,
            headers=headers,
            data=json.dumps(payload),
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        narrative = (
            resp.json()
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )
        return _split_narrative(narrative, model="gemini-2.0-flash")

    except requests.exceptions.RequestException as exc:
        return {
            "narrative": "", "expert_diag": "", "root_cause": "",
            "model": "gemini-2.0-flash", "error": str(exc),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _split_narrative(narrative: str, model: str) -> dict:
    paragraphs  = [p.strip() for p in narrative.split("\n\n") if p.strip()]
    expert_diag = paragraphs[0] if paragraphs else narrative
    root_cause  = paragraphs[1] if len(paragraphs) >= 2 else ""

    # Strip markdown bold headers if Gemini added them
    expert_diag = expert_diag.lstrip("**Expert Diagnosis**").strip().lstrip(":").strip()
    root_cause  = (root_cause
                   .lstrip("**Probable Root Cause").strip()
                   .lstrip("& Recommended Action**").strip()
                   .lstrip(":").strip())

    return {
        "narrative"  : narrative,
        "expert_diag": expert_diag,
        "root_cause" : root_cause,
        "model"      : model,
        "error"      : None,
    }


def _stub_narrative(diagnosis: dict) -> dict:
    """Template-based fallback when no API key is present."""
    fault    = diagnosis.get("fault_name", "Unknown Fault")
    score    = diagnosis.get("anomaly_score", 0.0)
    top3     = ", ".join(diagnosis.get("top3_sensors", []))
    comps    = ", ".join(diagnosis.get("components", []))
    severity = diagnosis.get("severity", "UNKNOWN")

    expert_diag = (
        f"The LSTM autoencoder detected an anomaly with a reconstruction error "
        f"of {score:.5f}, exceeding the established healthy baseline threshold. "
        f"The sensors showing the highest reconstruction error are {top3}, "
        f"which correspond to the {comps}. "
        f"The expert rule-base identifies this pattern as consistent with {fault} "
        f"({severity} severity)."
    )
    root_cause = (
        f"The most probable root cause is {fault.lower()} in the affected component(s). "
        f"If not addressed, this degradation may progress to reduced engine efficiency "
        f"and, in high-severity cases, potential in-flight shutdown. "
        f"Recommended action: schedule a borescope inspection of the {comps} at the "
        f"next available maintenance window, and monitor trend data at increased frequency."
    )

    return {
        "narrative"  : expert_diag + "\n\n" + root_cause,
        "expert_diag": expert_diag,
        "root_cause" : root_cause,
        "model"      : "stub (no API key)",
        "error"      : None,
    }
