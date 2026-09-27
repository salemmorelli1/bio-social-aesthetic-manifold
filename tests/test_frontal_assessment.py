"""Contracts for the exploratory frontal-landmark assessment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.analytics import CANONICAL_TEMPLATE  # noqa: E402
from core.frontal_assessment import (  # noqa: E402
    assess_frontal_landmarks,
    assess_frontal_landmarks_json,
)


def _values(points):
    return {metric["key"]: metric["value"] for metric in assess_frontal_landmarks(points)["metrics"]}


def test_frontal_metrics_are_similarity_invariant_without_normative_reference():
    baseline = assess_frontal_landmarks(CANONICAL_TEMPLATE)
    angle = 0.91
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = assess_frontal_landmarks(
        CANONICAL_TEMPLATE @ rotation * 103.0 + np.array([200.0, -173.0])
    )
    assert baseline["reference_population"] is None
    assert baseline["attractiveness_score"] is None
    assert baseline["pose_verified"] is False
    assert baseline["image_quality_verified"] is False
    assert len(baseline["metrics"]) == 8
    for original, moved in zip(baseline["metrics"], transformed["metrics"], strict=True):
        assert original["status"] == moved["status"] == "measured"
        assert moved["value"] == pytest.approx(original["value"], abs=1e-12)


def test_frontal_metrics_keep_representable_detail_with_large_coordinate_origin():
    translated = CANONICAL_TEMPLATE + np.array([1.0e10, -1.0e10])
    # Removing the known origin after float64 input quantization is the best
    # attainable reference; no algorithm can restore lost input mantissa bits.
    recentered = translated - np.array([1.0e10, -1.0e10])
    actual = _values(translated)
    expected = _values(recentered)
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value, abs=1e-12), key


def test_frontal_report_discloses_missing_denominators_and_real_discrepancy():
    modified = CANONICAL_TEMPLATE.copy()
    modified[35] = modified[31]
    report = assess_frontal_landmarks(modified)
    metrics = {item["key"]: item for item in report["metrics"]}
    assert metrics["mouth_to_nose"]["value"] is None
    assert metrics["mouth_to_nose"]["status"] == "unavailable"
    assert metrics["inner_eye_to_eye_width"]["status"] == "measured"

    wider_eye = CANONICAL_TEMPLATE.copy()
    wider_eye[36, 0] -= 0.03
    assert _values(wider_eye)["eyes_paired_discrepancy_pct"] > (
        _values(CANONICAL_TEMPLATE)["eyes_paired_discrepancy_pct"]
    )


def test_frontal_bridge_is_strict_json_and_withholds_unobservable_categories():
    payload = assess_frontal_landmarks_json(CANONICAL_TEMPLATE.ravel().tolist())
    assert "NaN" not in payload and "Infinity" not in payload
    result = json.loads(payload)
    assert result["analysis_kind"] == "exploratory_2d_frontal_measurements"
    assert any("hairline" in item.lower() for item in result["not_assessed"])
    assert any("skin" in item.lower() for item in result["not_assessed"])
    assert "error" in json.loads(assess_frontal_landmarks_json([0.0] * 135))


def test_browser_loading_both_python_sources_preserves_core_schema():
    root = Path(__file__).resolve().parents[1]
    namespace = {"__name__": "core.analytics"}
    for path in (root / "core/analytics.py", root / "core/frontal_assessment.py"):
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    coordinates = CANONICAL_TEMPLATE.ravel().tolist()
    assert json.loads(namespace["run_pipeline_from_js"](coordinates))["schema_version"] == "2.0"
    assert json.loads(namespace["assess_frontal_landmarks_json"](coordinates))[
        "schema_version"
    ] == "frontal-landmarks-v1"
