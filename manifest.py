"""Load Manifest metadata produced during static analysis."""

from __future__ import annotations

import json
from pathlib import Path

LIST_FIELDS = ("permissions", "services", "receivers", "activities", "providers")


def normalize_manifest(raw: dict) -> dict:
    """Normalize sidecar JSON into the schema expected by the prompt formatter."""
    manifest = {
        "package": raw.get("package") or "unknown",
        "target_sdk": raw.get("target_sdk"),
    }
    for field in LIST_FIELDS:
        values = raw.get(field, []) or []
        manifest[field] = sorted(set(str(value) for value in values if value))
    return manifest


def infer_manifest_json(static_ir_file: str | Path | None) -> Path | None:
    """Infer the Manifest sidecar JSON path for one static-analysis text file."""
    if not static_ir_file:
        return None
    candidate = Path(static_ir_file).with_suffix(".manifest.json")
    return candidate if candidate.exists() else None


def load_manifest(manifest_json: str | Path | None) -> dict:
    """Load Manifest metadata from the static-analysis sidecar JSON."""
    if manifest_json is None:
        raise FileNotFoundError(
            "Missing Manifest metadata sidecar. Run preprocessing.extract_static_ir "
            "first or pass --manifest_json explicitly."
        )

    path = Path(manifest_json)
    if not path.exists():
        raise FileNotFoundError(f"Manifest metadata JSON not found: {path}")

    return normalize_manifest(json.loads(path.read_text(encoding="utf-8")))
