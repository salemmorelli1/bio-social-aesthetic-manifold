"""Export simulated GPA consensus shapes for the independent AME preview.

SPDX-License-Identifier: MIT

These configurations are synthetic method examples, not empirical population
models, clinical norms, demographic classes, or aesthetic targets. The
existing Python engine remains the source of truth for the simulation and its
full tangent/covariance calculations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.analytics import (  # noqa: E402
    CANONICAL_TEMPLATE,
    CANONICAL_TEMPLATE_SOURCE_COMMIT,
    ENGINE_VERSION,
    _fit_reference_model,
)
from core.frontal_assessment import assess_frontal_landmarks  # noqa: E402

OUTPUT = ROOT / "src/lib/math/simulated-references.json"


def reference_payload() -> dict:
    """Return on-disk examples and Python-generated cross-language checks."""

    references = {}
    for key in ("a", "b", "pooled"):
        model = _fit_reference_model(key)
        references[key] = {
            "kind": "simulated",
            "sample_size": model.sample_size,
            "seed_description": model.seed_description,
            "consensus": model.gpa.consensus.tolist(),
        }

    # Frozen examples let the TypeScript module be checked against both Python
    # engines; they are not in-sample calibration or validation on real faces.
    canonical = CANONICAL_TEMPLATE.tolist()
    frontal = assess_frontal_landmarks(canonical)
    return {
        "schema_version": "ame-simulated-consensus-v1",
        "kind": "simulated",
        "engine_version": ENGINE_VERSION,
        "source": "core.analytics._fit_reference_model",
        "canonical_model_commit": CANONICAL_TEMPLATE_SOURCE_COMMIT,
        "interpretation": (
            "Algorithm-generated example shapes. No empirical cohort, clinical "
            "norm, demographic label, attractiveness target, or population percentile."
        ),
        "references": references,
        "cross_language_check": {
            "input": canonical,
            "frontal_values": {
                metric["key"]: metric["value"] for metric in frontal["metrics"]
            },
            "partial_procrustes": {
                key: float(np.linalg.norm(
                    _aligned_canonical(key) - _fit_reference_model(key).gpa.consensus
                ))
                for key in references
            },
        },
    }


def _aligned_canonical(key: str) -> np.ndarray:
    """Reuse the Python GPA alignment for the independent TypeScript check."""

    from core.analytics import _align_to_reference, _to_preshape

    preshape, _, _ = _to_preshape(CANONICAL_TEMPLATE)
    return _align_to_reference(preshape, _fit_reference_model(key).gpa.consensus)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="regenerate the JSON fixture")
    mode.add_argument("--check", action="store_true", help="verify committed fixture")
    args = parser.parse_args()

    expected = reference_payload()
    if args.write:
        OUTPUT.write_text(json.dumps(expected, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Wrote {OUTPUT.relative_to(ROOT)}")
        return 0

    actual = json.loads(OUTPUT.read_text(encoding="utf-8"))
    for key in ("schema_version", "kind", "engine_version", "source", "canonical_model_commit", "interpretation"):
        if actual.get(key) != expected[key]:
            raise ValueError(f"Simulated-reference provenance changed: {key}.")
    if set(actual.get("references", {})) != set(expected["references"]):
        raise ValueError("Simulated-reference choices changed.")
    for key, item in expected["references"].items():
        observed = actual["references"][key]
        if (observed.get("kind"), observed.get("sample_size"), observed.get("seed_description")) != (
            item["kind"], item["sample_size"], item["seed_description"]
        ):
            raise ValueError(f"Simulated-reference metadata changed: {key}.")
        _compare_numeric(observed.get("consensus"), item["consensus"], f"{key} consensus")
    observed_check = actual.get("cross_language_check", {})
    expected_check = expected["cross_language_check"]
    _compare_numeric(observed_check.get("input"), expected_check["input"], "check input")
    if set(observed_check.get("frontal_values", {})) != set(expected_check["frontal_values"]):
        raise ValueError("Frontal metric keys changed.")
    for key, value in expected_check["frontal_values"].items():
        _compare_numeric(observed_check["frontal_values"][key], value, key)
    if set(observed_check.get("partial_procrustes", {})) != set(expected_check["partial_procrustes"]):
        raise ValueError("Procrustes check keys changed.")
    for key, value in expected_check["partial_procrustes"].items():
        _compare_numeric(observed_check["partial_procrustes"][key], value, key)
    print("Simulated consensus and cross-language checks match the Python engine.")
    return 0


def _compare_numeric(observed: object, expected: object, name: str) -> None:
    try:
        left = np.asarray(observed, dtype=np.float64)
        right = np.asarray(expected, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numerical fixture: {name}.") from error
    if (left.shape != right.shape or not np.all(np.isfinite(left)) or
            not np.allclose(left, right, rtol=0.0, atol=1.0e-12)):
        raise ValueError(f"Numerical fixture changed: {name}.")


if __name__ == "__main__":
    raise SystemExit(main())
