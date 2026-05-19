"""Metric helpers."""

from __future__ import annotations

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def binary_metrics(labels: list[int], preds: list[int]) -> dict:
    """Compute binary malware-detection metrics."""
    if not labels:
        return {"accuracy": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0, "n": 0}

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "n": len(labels),
    }
