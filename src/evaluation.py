"""Held-out evaluation helpers for the C-MAPSS test engines.

The C-MAPSS test files provide an engine-level true RUL value, but no binary
anomaly label. For the reported anomaly metrics, define a positive test label
as near end of life (true RUL <= 30 cycles). This explicit proxy target lets
us evaluate the existing thresholded anomaly detector without claiming that
the benchmark supplies direct fault-state annotations.
"""

from __future__ import annotations

import numpy as np

from src.model import compute_reconstruction_error


TEST_RUL_CUTOFF = 30


def evaluate_test_sets(
    model,
    test_data: dict,
    thresholds: dict[str, float],
    rul_cutoff: int = TEST_RUL_CUTOFF,
) -> dict[str, dict]:
    """Compute engine-level precision, recall and FPR on held-out test sets.

    `test_data` maps dataset IDs to `test_seqs` and ground-truth `rul` arrays.
    The detector prediction is reconstruction error > that dataset's healthy
    threshold. The reference label is true RUL <= `rul_cutoff` cycles.
    """
    evaluation = {}
    for ds_id, dataset in test_data.items():
        scores = compute_reconstruction_error(model, dataset["test_seqs"])
        true_rul = np.asarray(dataset["rul"], dtype=float)
        if len(scores) != len(true_rul):
            raise ValueError(
                f"{ds_id}: got {len(scores)} test scores but "
                f"{len(true_rul)} RUL labels"
            )

        threshold = float(thresholds[ds_id])
        actual_positive = true_rul <= rul_cutoff
        predicted_positive = scores > threshold

        tp = int(np.sum(predicted_positive & actual_positive))
        fp = int(np.sum(predicted_positive & ~actual_positive))
        tn = int(np.sum(~predicted_positive & ~actual_positive))
        fn = int(np.sum(~predicted_positive & actual_positive))

        evaluation[ds_id] = {
            "threshold": round(threshold, 6),
            "mean_error": round(float(scores.mean()), 6),
            "max_error": round(float(scores.max()), 6),
            "flagged": int(predicted_positive.sum()),
            "total": len(scores),
            "rul_cutoff": int(rul_cutoff),
            "positive_test_engines": int(actual_positive.sum()),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 4) if tp + fn else 0.0,
            "false_positive_rate": round(fp / (fp + tn), 4) if fp + tn else 0.0,
        }

    return evaluation
