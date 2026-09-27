"""Exploratory frontal measurements from ordered 68-point 2D landmarks.

SPDX-License-Identifier: MIT

This module is deliberately independent of the validated morphometric engine
and its 27-page reproducible report. It computes observable ratios and paired
landmark discrepancies; it has no empirical reference distribution, ideal
range, attractiveness target, or photographic pose estimator.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

FRONTAL_SCHEMA_VERSION = "frontal-landmarks-v1"
MIN_SPAN = 1.0e-12

PAIRS = {
    "eyes": ((36, 45), (37, 44), (38, 43), (39, 42), (40, 47), (41, 46)),
    "jaw": tuple((index, 16 - index) for index in range(8)),
    "mouth": ((48, 54), (49, 53), (50, 52), (55, 59), (56, 58)),
}


def _normalized_landmarks(landmarks: Any) -> FloatArray:
    """Preserve scale and rotation invariance without overflowing on input."""

    if isinstance(landmarks, str):
        landmarks = json.loads(landmarks)
    elif hasattr(landmarks, "to_py"):
        landmarks = landmarks.to_py()
    points = np.asarray(landmarks, dtype=np.float64)
    if points.size != 136:
        raise ValueError("Expected 68 ordered two-dimensional landmarks.")
    points = points.reshape(68, 2)
    if not np.all(np.isfinite(points)):
        raise ValueError("All landmark coordinates must be finite.")
    # Subtract an observed point before scaling. Scaling by the magnitude of
    # the raw coordinates first loses shape detail when the face is offset far
    # from the origin relative to its extent. The fallback handles the rare
    # case where subtraction of finite opposite-signed coordinates overflows.
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = points - points[0]
    if not np.all(np.isfinite(shifted)):
        coordinate_scale = float(np.max(np.abs(points)))
        scaled_points = points / coordinate_scale
        shifted = scaled_points - scaled_points[0]
    scale = float(np.max(np.abs(shifted)))
    if scale == 0.0:
        raise ValueError("Landmark configuration has zero size.")
    scaled = shifted / scale
    centered = scaled - np.mean(scaled, axis=0, keepdims=True)
    size = float(np.linalg.norm(centered))
    if size == 0.0:
        raise ValueError("Landmark configuration has zero centroid size.")
    return centered / size


def _distance(points: FloatArray, a: int, b: int) -> float:
    return float(np.linalg.norm(points[a] - points[b]))


def _metric(
    key: str,
    label: str,
    numerator: float | None,
    denominator: float,
    landmarks: list[int],
    definition: str,
    unit: str = "ratio",
) -> dict[str, Any]:
    available = (
        numerator is not None
        and np.isfinite(numerator)
        and np.isfinite(denominator)
        and denominator > MIN_SPAN
    )
    value = float(numerator / denominator) if available else None
    return {
        "key": key,
        "label": label,
        "value": value,
        "unit": unit,
        "status": "measured" if available else "unavailable",
        "landmark_indices": landmarks,
        "definition": definition,
        "reason": None if available else "Required span is missing or too small.",
    }


def _paired_discrepancy(
    points: FloatArray,
    pairs: tuple[tuple[int, int], ...],
    axis_start: FloatArray,
    axis_direction: FloatArray | None,
) -> float | None:
    """Mean reflected-pair separation, in normalized shape-space units."""

    if axis_direction is None:
        return None
    residuals = []
    for first, second in pairs:
        offset = points[first] - axis_start
        reflected = axis_start + 2.0 * np.dot(offset, axis_direction) * axis_direction - offset
        residuals.append(float(np.linalg.norm(reflected - points[second])))
    return float(np.mean(residuals))


def assess_frontal_landmarks(landmarks: Any) -> dict[str, Any]:
    """Describe measurable frontal geometry without judging its appearance.

    The 68-point ordering is Dlib-style. These ratios are measured in a 2D
    projection and do not estimate true anatomy, a population percentile, or
    the desirability of a feature. The calling UI must identify whether the
    coordinates came from a simulated sample, a file, or a local photo.
    """

    points = _normalized_landmarks(landmarks)
    eye_right = _distance(points, 36, 39)
    eye_left = _distance(points, 42, 45)
    mean_eye_width = (eye_right + eye_left) / 2.0
    inner_eye_span = _distance(points, 39, 42)
    nose_span = _distance(points, 31, 35)
    mouth_span = _distance(points, 48, 54)
    face_span = _distance(points, 0, 16)

    axis_start = points[27]
    axis = points[8] - axis_start
    axis_length = float(np.linalg.norm(axis))
    axis_direction = axis / axis_length if axis_length > MIN_SPAN else None

    lower_fraction = None
    if axis_direction is not None:
        nose_position = float(np.dot(points[33] - axis_start, axis_direction))
        if 0.0 <= nose_position <= axis_length:
            lower_fraction = axis_length - nose_position

    metrics = [
        _metric(
            "inner_eye_to_eye_width", "Inner-eye / eye width",
            inner_eye_span, mean_eye_width, [36, 39, 42, 45],
            "Distance between inner eye corners divided by mean visible eye width.",
        ),
        _metric(
            "nose_to_inner_eye", "Nose / inner-eye span",
            nose_span, inner_eye_span, [31, 35, 39, 42],
            "Visible nose-base width divided by inner eye-corner distance.",
        ),
        _metric(
            "mouth_to_nose", "Mouth / nose width",
            mouth_span, nose_span, [31, 35, 48, 54],
            "Mouth-corner span divided by visible nose-base width.",
        ),
        _metric(
            "mouth_to_jaw", "Mouth / jaw span",
            mouth_span, face_span, [0, 16, 48, 54],
            "Mouth-corner span divided by jaw-outline span in this image.",
        ),
        _metric(
            "lower_visible_face_fraction", "Lower visible-face fraction",
            lower_fraction, axis_length, [8, 27, 33],
            "Projected nose-base-to-chin length divided by nose-bridge-to-chin length. "
            "The hairline is not detected, so this is not a classical facial-third measure.",
        ),
    ]

    for region, pairs in PAIRS.items():
        discrepancy = _paired_discrepancy(points, pairs, axis_start, axis_direction)
        metrics.append(_metric(
            f"{region}_paired_discrepancy_pct",
            f"{region.capitalize()} paired discrepancy",
            None if discrepancy is None else 100.0 * discrepancy,
            face_span,
            [index for pair in pairs for index in pair] + [27, 8, 0, 16],
            "Mean distance after reflecting paired landmarks across the nose-bridge-to-chin "
            "axis, divided by jaw-outline span; percent of image span, not a beauty score.",
            unit="% of jaw span",
        ))

    return {
        "schema_version": FRONTAL_SCHEMA_VERSION,
        "analysis_kind": "exploratory_2d_frontal_measurements",
        "reference_population": None,
        "attractiveness_score": None,
        "pose_verified": False,
        "image_quality_verified": False,
        "metrics": metrics,
        "not_assessed": [
            "Profile depth, projection, and three-dimensional structure",
            "Forehead/hairline thirds (no hairline landmark)",
            "Skin texture, pigmentation, and lighting",
            "Attractiveness, ideal ratios, demographic norms, or treatment advice",
        ],
        "interpretation": (
            "Landmark measurements are descriptive image-space quantities. "
            "Pose, lens distance, expression, occlusion, and landmark accuracy "
            "are not verified; no empirical reference intervals or aesthetic "
            "outcome model were fitted."
        ),
    }


def assess_frontal_landmarks_json(landmarks: Any) -> str:
    """Pyodide-friendly strict JSON entry point with bounded validation errors."""

    try:
        return json.dumps(assess_frontal_landmarks(landmarks), allow_nan=False)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        return json.dumps({"schema_version": FRONTAL_SCHEMA_VERSION, "error": str(error)})
