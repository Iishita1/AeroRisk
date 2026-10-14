"""
src/expert_diagnosis.py
-----------------------
Rule-based expert system:
  sensor reconstruction errors  →  component mapping  →  fault hypothesis
  + optional NLP maintenance-log search for corroborating evidence.

Layer 2 of the pipeline — deterministic, no ML.
"""

from __future__ import annotations
import re
from typing import Optional


# ── Sensor → Component mapping ────────────────────────────────────────────────

SENSOR_COMPONENT_MAP: dict[str, str] = {
    "s2":  "Fan / Low-Pressure Compressor",
    "s3":  "High-Pressure Compressor (HPC)",
    "s4":  "High-Pressure Compressor (HPC)",
    "s7":  "High-Pressure Turbine (HPT)",
    "s8":  "Low-Pressure Turbine (LPT)",
    "s9":  "High-Pressure Compressor (HPC)",
    "s11": "Low-Pressure Turbine (LPT)",
    "s12": "High-Pressure Compressor (HPC)",
    "s13": "Low-Pressure Compressor / Fan",
    "s14": "High-Pressure Turbine (HPT)",
    "s15": "High-Pressure Compressor (HPC)",
    "s17": "Low-Pressure Turbine (LPT)",
    "s20": "High-Pressure Turbine (HPT)",
    "s21": "Low-Pressure Turbine (LPT)",
}


# ── Fault rules ────────────────────────────────────────────────────────────────
# (trigger_sensors, fault_name, severity, description)
# Rules are checked in order; first match wins.
# An empty trigger set acts as the catch-all fallback.

_FAULT_RULES: list[tuple[frozenset, str, str, str]] = [
    # HPC faults
    (
        frozenset({"s4", "s9", "s14"}),
        "HPC Stall",
        "HIGH",
        "Simultaneous deviation in HPC discharge temperature (s4), HPC outlet "
        "pressure ratio (s9), and HPT inlet temperature (s14) is a classic "
        "signature of high-pressure compressor stall.",
    ),
    (
        frozenset({"s3", "s4", "s9"}),
        "HPC Efficiency Degradation",
        "MEDIUM",
        "Elevated s3 (HPC inlet), s4 (HPC discharge), and s9 (HPT inlet) "
        "suggest progressive blade-tip erosion or fouling reducing HPC "
        "isentropic efficiency.",
    ),
    (
        frozenset({"s4", "s15"}),
        "HPC Fouling",
        "MEDIUM",
        "Correlated drift in HPC discharge temperature (s4) and HPC pressure "
        "ratio (s15) is consistent with compressor blade fouling.",
    ),
    (
        frozenset({"s3", "s12"}),
        "HPC Inlet Distortion",
        "LOW",
        "Anomalies in fan-exit / LPC-outlet conditions (s3) together with "
        "reduced HPC speed (s12) indicate inlet airflow distortion.",
    ),
    # HPT faults
    (
        frozenset({"s14", "s20"}),
        "HPT Blade Erosion",
        "HIGH",
        "Co-deviation of HPT inlet temperature (s14) and HPT exhaust "
        "temperature (s20) points to turbine blade erosion or tip-clearance growth.",
    ),
    (
        frozenset({"s7", "s14"}),
        "HPT Seal Degradation",
        "MEDIUM",
        "Combined HPT total temperature anomalies (s7, s14) suggest internal "
        "seal wear leading to hot-gas path leakage.",
    ),
    # LPT / Fan faults
    (
        frozenset({"s8", "s11"}),
        "LPT Efficiency Loss",
        "MEDIUM",
        "Coordinated changes in LPT temperature sensors (s8, s11) indicate "
        "progressive low-pressure turbine efficiency loss.",
    ),
    (
        frozenset({"s2", "s13"}),
        "Fan Blade Damage",
        "HIGH",
        "Concurrent anomalies in fan-inlet total temperature (s2) and "
        "fan-exit static pressure (s13) are consistent with fan blade damage "
        "or foreign object ingestion.",
    ),
    (
        frozenset({"s17", "s21"}),
        "LPT Stage Degradation",
        "MEDIUM",
        "Combined LPT stage sensor deviations (s17, s21) suggest stage-level "
        "aerodynamic deterioration.",
    ),
    # Catch-all
    (
        frozenset(),
        "General Sensor Anomaly",
        "LOW",
        "The deviant sensors do not match a specific fault pattern. General "
        "engine health degradation is suspected; further inspection required.",
    ),
]


def diagnose(top_sensors: list[str]) -> dict:
    """
    Apply the rule base to the top-N deviant sensors.

    Returns dict {fault_name, severity, description, components, matched_rule}
    """
    sensor_set    = set(top_sensors)
    matched_fault = None

    for trigger_set, fault_name, severity, description in _FAULT_RULES:
        if not trigger_set:
            if matched_fault is None:
                matched_fault = (trigger_set, fault_name, severity, description)
            continue
        if trigger_set.issubset(sensor_set):
            matched_fault = (trigger_set, fault_name, severity, description)
            break

    if matched_fault is None:
        matched_fault = (frozenset(), "General Sensor Anomaly", "LOW",
                         "No specific fault pattern matched.")

    trigger_set, fault_name, severity, description = matched_fault

    components = sorted(
        {SENSOR_COMPONENT_MAP.get(s, "Unknown Component") for s in sensor_set}
    )

    return {
        "fault_name"   : fault_name,
        "severity"     : severity,
        "description"  : description,
        "components"   : components,
        "matched_rule" : trigger_set,
    }


# ── NLP maintenance-log search ────────────────────────────────────────────────

_KEYWORD_BANK: dict[str, list[str]] = {
    "HPC Stall": [
        "compressor stall", "hpc stall", "surge", "stall margin",
        "compressor surge", "high pressure compressor", "hpc", "blade stall",
    ],
    "HPC Efficiency Degradation": [
        "hpc degradation", "compressor efficiency", "blade erosion",
        "fouling", "efficiency loss", "high pressure compressor",
    ],
    "HPC Fouling": [
        "fouling", "compressor wash", "blade fouling", "hpc wash",
        "deposit", "compressor cleaning",
    ],
    "HPC Inlet Distortion": [
        "inlet distortion", "airflow distortion", "fan exit",
        "lpc outlet", "inlet guide vane", "igv",
    ],
    "HPT Blade Erosion": [
        "turbine blade", "hpt erosion", "tip clearance", "blade wear",
        "high pressure turbine", "hpt", "turbine erosion",
    ],
    "HPT Seal Degradation": [
        "seal wear", "seal degradation", "hpt seal",
        "hot section", "leakage", "turbine seal",
    ],
    "LPT Efficiency Loss": [
        "lpt", "low pressure turbine", "turbine efficiency", "lpt degradation",
    ],
    "Fan Blade Damage": [
        "fan blade", "foreign object", "fod", "bird strike",
        "fan damage", "blade damage", "fan", "ingestion",
    ],
    "LPT Stage Degradation": [
        "lpt stage", "turbine stage", "stage degradation", "lpt efficiency",
    ],
    "General Sensor Anomaly": [
        "anomaly", "unusual reading", "sensor drift", "exceedance",
        "out of limits", "maintenance check", "inspection required",
    ],
}


def search_maintenance_logs(
    log_text: str,
    fault_name: str,
    top_sensors: Optional[list[str]] = None,
    use_spacy: bool = True,
) -> dict:
    """
    Search free-text maintenance log for keywords corroborating `fault_name`.

    Falls back to plain regex if spaCy is unavailable.

    Returns dict {found_keywords, evidence_score, corroborating, highlighted_text}
    """
    keywords = list(_KEYWORD_BANK.get(fault_name, []))
    if top_sensors:
        keywords += [s.lower() for s in top_sensors]
    keywords = list(dict.fromkeys(kw.lower() for kw in keywords))

    search_text = log_text.lower()
    if use_spacy:
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            doc = nlp(log_text)
            search_text = " ".join(tok.lemma_.lower() for tok in doc)
        except Exception:
            pass  # spaCy unavailable — plain text fallback

    found       = []
    highlighted = log_text

    for kw in keywords:
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        if pattern.search(search_text):
            found.append(kw)
            highlighted = pattern.sub(lambda m: f"**{m.group()}**", highlighted)

    return {
        "found_keywords"   : found,
        "evidence_score"   : round(len(found) / max(len(keywords), 1), 3),
        "corroborating"    : len(found) > 0,
        "highlighted_text" : highlighted,
    }


# ── Full diagnostic bundle ────────────────────────────────────────────────────

def full_diagnosis(
    top_sensors: list[str],
    per_sensor_errors: dict,
    anomaly_score: float,
    threshold: float,
    log_text: Optional[str] = None,
) -> dict:
    """
    Combine rule-based diagnosis + optional log search into one report dict.

    Returns all keys from diagnose() plus:
        anomaly_score, threshold, per_sensor_errors, top3_sensors, log_evidence
    """
    diag = diagnose(top_sensors)

    log_evidence = None
    if log_text and log_text.strip():
        log_evidence = search_maintenance_logs(log_text, diag["fault_name"], top_sensors)

    return {
        **diag,
        "anomaly_score"    : anomaly_score,
        "threshold"        : threshold,
        "per_sensor_errors": per_sensor_errors,
        "top3_sensors"     : top_sensors,
        "log_evidence"     : log_evidence,
    }
